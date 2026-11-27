//! Synthetic Windows process/Job behavior tests, requiring test-probe.
#![cfg(all(windows, feature = "test-probe"))]

use std::ffi::OsString;
use std::fs;
use std::io;
#[cfg(feature = "containment")]
use std::io::{Read as _, Write as _};
use std::net::{Ipv4Addr, TcpListener};
#[cfg(feature = "containment")]
use std::net::{TcpStream, UdpSocket};
use std::os::windows::ffi::{OsStrExt as _, OsStringExt as _};
use std::os::windows::fs::OpenOptionsExt as _;
use std::os::windows::process::CommandExt as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[cfg(feature = "containment")]
use bonsai_windows_process::{
    MAX_INHERITED_SECTIONS, MAX_WORKER_PRIVATE_STORAGE_BYTES, MAX_WORKER_PRIVATE_STORAGE_ENTRIES,
    ParentProcess, SandboxNetwork, SandboxPolicy, native_sandbox_available,
    native_sandbox_offline_available,
};
use bonsai_windows_process::{
    OwnedProcess, ProcessSpec, ToolProcessSpec, available_memory_bytes, listener_process_id,
};

// Hosted packages retain the exact probe next to their checked test binaries.
// The native qualifier verifies/holds this path through all six controls; normal
// source tests keep Cargo's own compiled executable selection.
fn probe_executable() -> PathBuf {
    std::env::var_os("BONSAI_SANDBOX_PROBE_PATH").map_or_else(
        || PathBuf::from(env!("CARGO_BIN_EXE_bonsai-process-test-probe")),
        PathBuf::from,
    )
}
const TIMEOUT: Duration = Duration::from_secs(10);
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Lane {
    path: PathBuf,
}

impl Lane {
    fn new(name: &str) -> io::Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "bonsai-process-{name}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }

    fn spec(&self, arguments: Vec<OsString>) -> ProcessSpec {
        ProcessSpec {
            executable: probe_executable(),
            arguments,
            working_directory: self.path.clone(),
            stdout: self.path.join("stdout.log"),
            stderr: self.path.join("stderr.log"),
            memory_limit_bytes: 256 * 1024 * 1024,
            active_process_limit: 3,
        }
    }
}

fn until(mut predicate: impl FnMut() -> io::Result<bool>) -> io::Result<()> {
    let start = Instant::now();
    let mut attempt = 0_u32;
    while !predicate()? {
        if start.elapsed() >= TIMEOUT {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "probe condition not reached",
            ));
        }
        if attempt < 4 {
            std::thread::yield_now();
        } else {
            let millis = u64::from(1_u32 << (attempt - 4).min(3)).min(5);
            std::thread::sleep(Duration::from_millis(millis));
        }
        attempt = attempt.saturating_add(1);
    }
    Ok(())
}

fn unused_port() -> io::Result<u16> {
    Ok(TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?
        .local_addr()?
        .port())
}

#[cfg(feature = "containment")]
#[test]
fn sandbox_rejects_invalid_grants_before_creating_evidence() -> io::Result<()> {
    let lane = Lane::new("invalid-sandbox")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let mut spec = lane.spec(vec![OsString::from("sleep")]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    for (policy, reason) in [
        SandboxPolicy {
            readonly_paths: vec![spec.executable.clone(), PathBuf::from("relative-model")],
            writable_directory: scratch.clone(),
        },
        SandboxPolicy {
            readonly_paths: vec![spec.executable.clone()],
            writable_directory: PathBuf::from("relative-scratch"),
        },
        SandboxPolicy {
            readonly_paths: Vec::new(),
            writable_directory: scratch.clone(),
        },
        SandboxPolicy {
            readonly_paths: vec![spec.executable.clone(), scratch.clone()],
            writable_directory: scratch,
        },
    ]
    .into_iter()
    .zip([
        "expanded absolute paths",
        "expanded absolute paths",
        "1..128 explicit read-only grants",
        "exact existing files",
    ]) {
        for piped in [
            None,
            Some(SandboxNetwork::Offline),
            Some(SandboxNetwork::HostLoopback),
        ] {
            let error = piped
                .map_or_else(
                    || OwnedProcess::start_sandboxed(&spec, &policy).err(),
                    |network| OwnedProcess::start_sandboxed_piped(&spec, &policy, network).err(),
                )
                .ok_or_else(|| io::Error::other("invalid sandbox policy launched a worker"))?;
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains(reason), "{error}");
            assert!(!spec.stdout.exists());
            assert!(!spec.stderr.exists());
        }
        let error = OwnedProcess::start_sandboxed_broker_pipe(&spec, &policy)
            .err()
            .ok_or_else(|| {
                io::Error::other("invalid sandbox policy launched a broker-pipe worker")
            })?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains(reason), "{error}");
        assert!(!spec.stdout.exists());
        assert!(!spec.stderr.exists());
    }
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
fn piped_sandbox_rejects_process_count_before_creating_streams() -> io::Result<()> {
    let lane = Lane::new("invalid-piped-count")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let mut spec = lane.spec(vec![OsString::from("sleep")]);
    spec.working_directory.clone_from(&scratch);
    let policy = SandboxPolicy {
        readonly_paths: vec![spec.executable.clone()],
        writable_directory: scratch,
    };
    for network in [SandboxNetwork::Offline, SandboxNetwork::HostLoopback] {
        let error = OwnedProcess::start_sandboxed_piped(&spec, &policy, network)
            .err()
            .ok_or_else(|| io::Error::other("multi-process sandbox policy was accepted"))?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("single-process"));
        assert!(!spec.stdout.exists());
        assert!(!spec.stderr.exists());
    }
    let error = OwnedProcess::start_sandboxed_broker_pipe(&spec, &policy)
        .err()
        .ok_or_else(|| io::Error::other("multi-process broker-pipe sandbox policy was accepted"))?;
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("single-process"));
    assert!(!spec.stdout.exists());
    assert!(!spec.stderr.exists());
    Ok(())
}

#[cfg(feature = "containment")]
#[allow(
    unsafe_code,
    reason = "A disposable inherited-handle sentinel checks the actual launch handle list."
)]
fn inheritable_file_identity(file: &fs::File) -> io::Result<String> {
    use std::os::windows::io::AsRawHandle as _;
    use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };

    let handle = file.as_raw_handle();
    let mut identity = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: The fixture retains the exact file throughout both synchronous
    // calls. The writable identity storage has the native ABI layout.
    let queried = unsafe { GetFileInformationByHandle(handle, &raw mut identity) };
    if queried == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: This live fixture handle alone becomes inheritable; the production
    // explicit handle list must still exclude it from the contained child.
    if unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(format!(
        "{}:{}:{}:{}",
        handle.addr(),
        identity.dwVolumeSerialNumber,
        identity.nFileIndexHigh,
        identity.nFileIndexLow
    ))
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires native Windows PSEC v1.1; exercises actual offline pipes and kernel authority."]
fn native_sandbox_offline_pipes_enforce_authority_and_close_without_logs() -> io::Result<()> {
    let lane = Lane::new("offline-pipes")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let approved = lane.path.join("approved-input.wav");
    let unrelated = lane.path.join("unrelated.key");
    fs::write(&approved, "approved audio input")?;
    fs::write(&unrelated, "unrelated credential")?;
    // Trusted controls prove the same input is writable, the unrelated file is
    // readable and the target listener accepts connections outside containment.
    drop(fs::OpenOptions::new().write(true).open(&approved)?);
    let sentinel = fs::File::open(&unrelated)?;
    assert_eq!(fs::read_to_string(&unrelated)?, "unrelated credential");
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let endpoint = listener.local_addr()?;
    drop(TcpStream::connect_timeout(
        &endpoint,
        Duration::from_secs(1),
    )?);
    drop(listener.accept()?);
    let mut spec = lane.spec(vec![
        OsString::from("sandbox-offline-pipes"),
        approved.clone().into_os_string(),
        unrelated.clone().into_os_string(),
        OsString::from(endpoint.port().to_string()),
        OsString::from(inheritable_file_identity(&sentinel)?),
    ]);
    let control = Command::new(&spec.executable)
        .arg("report")
        .current_dir(&lane.path)
        .creation_flags(0x0800_0000)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    assert!(control.success());
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let (process, mut input, output) = OwnedProcess::start_sandboxed_piped(
        &spec,
        &SandboxPolicy {
            readonly_paths: vec![spec.executable.clone(), approved.clone()],
            writable_directory: scratch.clone(),
        },
        SandboxNetwork::Offline,
    )?;
    assert_eq!(process.active_processes()?, 1);
    let expected = (0..=255_u8).cycle().take(131_079).collect::<Vec<_>>();
    let payload = expected.clone();
    let exchange = std::thread::spawn(move || -> io::Result<Vec<u8>> {
        input.write_all(&payload)?;
        drop(input);
        let mut response = Vec::new();
        output.take(131_080).read_to_end(&mut response)?;
        Ok(response)
    });
    let exit = process.wait_for_exit(TIMEOUT);
    let stopped = process.stop(TIMEOUT);
    stopped?;
    let response = exchange
        .join()
        .map_err(|_| io::Error::other("offline pipe fixture panicked"))?;
    assert_eq!(
        exit?,
        Some(0),
        "{}",
        fs::read_to_string(scratch.join("offline-error.txt")).unwrap_or_default()
    );
    assert_eq!(process.active_processes()?, 0);
    assert_eq!(response?, expected);
    assert_eq!(fs::read_to_string(approved)?, "approved audio input");
    assert_eq!(fs::read_to_string(unrelated)?, "unrelated credential");
    assert!(!spec.stdout.exists());
    assert!(!spec.stderr.exists());
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires a native Windows PSEC v1.1 host with host-loopback ingress support."]
fn native_sandbox_limits_files_and_preserves_loopback_process_identity() -> io::Result<()> {
    native_sandbox_available()
        .map_err(|error| io::Error::new(error.kind(), format!("native preflight: {error}")))?;
    let lane = Lane::new("sandbox-files")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let model = lane.path.join("approved-model.gguf");
    let backend_key = lane.path.join("backend.key");
    let frontend_key = lane.path.join("frontend.key");
    fs::write(&model, "approved model")?;
    fs::write(&backend_key, "backend key")?;
    fs::write(&frontend_key, "unrelated frontend key")?;
    // Match the retained deny-write/delete handles used by production artifact
    // and credential verification while Windows admits the native policy.
    let _model_guard = fs::OpenOptions::new()
        .read(true)
        .share_mode(1)
        .open(&model)?;
    let _key_guard = fs::OpenOptions::new()
        .read(true)
        .share_mode(1)
        .open(&backend_key)?;
    let port = unused_port()?;
    let mut spec = lane.spec(vec![
        OsString::from("sandbox-files"),
        model.clone().into_os_string(),
        backend_key.clone().into_os_string(),
        frontend_key.clone().into_os_string(),
        OsString::from(port.to_string()),
    ]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let mut policy = SandboxPolicy {
        readonly_paths: vec![spec.executable.clone(), model.clone(), backend_key],
        writable_directory: scratch.clone(),
    };
    // The pinned Swift bundle needs 66 exact grants after adding its model, key
    // and adapter. Exercise native admission at that real launch size as well as
    // enforcing file/network/process authority on the resulting worker.
    for index in 0..63 {
        let dependency = lane.path.join(format!("runtime-{index}.bin"));
        fs::write(&dependency, b"verified runtime fixture")?;
        policy.readonly_paths.push(dependency);
    }
    assert_eq!(policy.readonly_paths.len(), 66);
    let process = OwnedProcess::start_sandboxed(&spec, &policy)
        .map_err(|error| io::Error::new(error.kind(), format!("first worker launch: {error}")))?;
    until(|| {
        if scratch.join("sandbox-ready.txt").exists() {
            return Ok(true);
        }
        if let Some(code) = process.try_wait()? {
            return Err(io::Error::other(format!(
                "sandbox file probe exited {code}: {}",
                fs::read_to_string(&spec.stderr)?
            )));
        }
        Ok(false)
    })?;
    assert_eq!(
        fs::read_to_string(scratch.join("sandbox-ready.txt"))?,
        process.id().to_string()
    );
    assert_eq!(listener_process_id(port)?, Some(process.id()));
    let _connection =
        TcpStream::connect_timeout(&(Ipv4Addr::LOCALHOST, port).into(), Duration::from_secs(2))?;
    assert_eq!(
        fs::read_to_string(scratch.join("scratch.txt"))?,
        "private scratch"
    );
    assert_eq!(fs::read_to_string(model)?, "approved model");
    assert_eq!(fs::read_to_string(frontend_key)?, "unrelated frontend key");
    let child_policy = fs::read_to_string(scratch.join("child-policy.txt"))?
        .parse::<u32>()
        .map_err(io::Error::other)?;
    fs::write(
        lane.path.join("observed-child-policy.txt"),
        child_policy.to_string(),
    )?;
    assert_other_worker_temp_denied(&scratch)?;
    assert!(process.memory_usage()?.private_bytes > 0);
    process.stop(TIMEOUT)?;
    assert!(process.try_wait()?.is_some());
    until(|| listener_process_id(port).map(|owner| owner.is_none()))?;
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires a native Windows PSEC v1.1 host with host-loopback ingress support."]
fn native_sandbox_stdout_flood_is_capped_and_stopped() -> io::Result<()> {
    sandbox_output_flood("stdout")
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires a native Windows PSEC v1.1 host with host-loopback ingress support."]
fn native_sandbox_stderr_flood_is_capped_and_stopped() -> io::Result<()> {
    sandbox_output_flood("stderr")
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires a native Windows PSEC v1.1 host with host-loopback ingress support."]
fn native_sandbox_exit_drains_both_logs_and_releases_capture_files() -> io::Result<()> {
    let lane = Lane::new("sandbox-output-tail")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let mut spec = lane.spec(vec![OsString::from("sandbox-output-tail")]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let process = OwnedProcess::start_sandboxed(
        &spec,
        &SandboxPolicy {
            readonly_paths: vec![spec.executable.clone()],
            writable_directory: scratch,
        },
    )?;
    assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
    for (stderr, path) in [(false, &spec.stdout), (true, &spec.stderr)] {
        // Status publication must precede neither tail retention nor release of
        // the parent's no-delete-sharing capture file, even with owner retained.
        let retained = path.with_extension("retained");
        fs::rename(path, &retained)?;
        let bytes = fs::read(retained)?;
        assert_eq!(bytes.len(), 17 * 16_384);
        for (byte, chunk) in (0..17_u8).zip(bytes.as_chunks::<16_384>().0) {
            let expected = if stderr { 255 - byte } else { byte };
            assert!(chunk.iter().all(|value| *value == expected));
        }
    }
    process.stop(TIMEOUT)?;
    drop(process);
    fs::remove_dir_all(lane.path)
}

#[cfg(feature = "containment")]
fn sandbox_output_flood(stream: &str) -> io::Result<()> {
    let lane = Lane::new(&format!("sandbox-{stream}-flood"))?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let mut spec = lane.spec(vec![OsString::from(format!("sandbox-flood-{stream}"))]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let process = OwnedProcess::start_sandboxed(
        &spec,
        &SandboxPolicy {
            readonly_paths: vec![spec.executable.clone()],
            writable_directory: scratch,
        },
    )?;
    let capture_failed = || match process.try_wait() {
        Err(error) if error.to_string().contains("retained-log bound") => Ok(true),
        Err(error) => Err(error),
        Ok(None) => Ok(false),
        Ok(Some(_)) => Err(io::Error::other(
            "flood worker exited without a log-limit error",
        )),
    };
    until(capture_failed)?;
    // Log overflow is an operation failure, while cleanup has its own verdict.
    // The original native Job and exact root must still be confirmed stopped.
    process.stop(TIMEOUT)?;
    assert_eq!(process.active_processes()?, 0);
    for (name, path) in [("stdout", &spec.stdout), ("stderr", &spec.stderr)] {
        let length = fs::metadata(path)?.len();
        if name == stream {
            assert_eq!(length, 16 * 1024 * 1024);
        } else {
            assert!(length <= 16 * 1024 * 1024);
        }
    }
    drop(process);
    fs::remove_dir_all(lane.path)
}

#[cfg(feature = "containment")]
fn assert_other_worker_temp_denied(first_scratch: &std::path::Path) -> io::Result<()> {
    // Windows can replace the requested TEMP/TMP with an OS-managed private
    // AppContainer temp. Prove isolation with a live second worker, rather than
    // assuming that the native environment preserves a particular pathname.
    let first = fs::read_to_string(first_scratch.join("kernel-temp-path.txt"))?;
    assert_eq!(fs::read_to_string(&first)?, "private kernel temp");
    let lane = Lane::new("sandbox-other-temp")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let mut spec = lane.spec(vec![
        OsString::from("sandbox-temp-denied"),
        OsString::from(&first),
    ]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let process = OwnedProcess::start_sandboxed(
        &spec,
        &SandboxPolicy {
            readonly_paths: vec![spec.executable.clone()],
            writable_directory: scratch.clone(),
        },
    )
    .map_err(|error| io::Error::new(error.kind(), format!("second temp worker launch: {error}")))?;
    assert_eq!(
        process.wait_for_exit(TIMEOUT)?,
        Some(0),
        "{}",
        fs::read_to_string(&spec.stderr)?
    );
    assert_ne!(
        fs::read_to_string(scratch.join("kernel-temp-path.txt"))?,
        first
    );
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires native Windows PSEC v1.1 and an IPv4 non-loopback host route."]
fn native_sandbox_host_loopback_capability_includes_host_interface_services() -> io::Result<()> {
    native_sandbox_available()?;
    // UDP connect selects a local route without sending any packet. The TCP
    // target belongs to this host; no Internet service or external peer is used.
    let route = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
    route.connect((Ipv4Addr::new(192, 0, 2, 1), 9))?;
    let address = route.local_addr()?.ip();
    if address.is_loopback() || address.is_unspecified() {
        return Err(io::Error::other(
            "no non-loopback IPv4 route for sandbox qualification",
        ));
    }
    let listener = TcpListener::bind((address, 0))?;
    let endpoint = listener.local_addr()?;
    // The native host-loopback capability also includes a host's other local
    // addresses. This scope is deliberately recorded rather than described as
    // broker-only communication or an external network grant.
    let control = TcpStream::connect_timeout(&endpoint, Duration::from_secs(2))?;
    drop(control);
    let lane = Lane::new("sandbox-network")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let mut spec = lane.spec(vec![
        OsString::from("sandbox-connect-allowed"),
        OsString::from(endpoint.to_string()),
    ]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let policy = SandboxPolicy {
        readonly_paths: vec![spec.executable.clone()],
        writable_directory: scratch.clone(),
    };
    let process = OwnedProcess::start_sandboxed(&spec, &policy)?;
    assert_eq!(
        process.wait_for_exit(TIMEOUT)?,
        Some(0),
        "{}",
        fs::read_to_string(&spec.stderr)?
    );
    assert_eq!(
        fs::read_to_string(scratch.join("network-allowed.txt"))?,
        "connected"
    );
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires native Windows PSEC v1.1 with networking policy support."]
fn native_sandbox_denies_external_tcp_connect_by_policy() -> io::Result<()> {
    native_sandbox_available()?;
    let lane = Lane::new("sandbox-external-network")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    // RFC 5737 TEST-NET-1 is a documentation-only address. There is no required
    // external service: only immediate Windows WSAEACCES counts as denial.
    // Timeout, missing route and connection refusal all fail this qualification.
    let mut spec = lane.spec(vec![
        OsString::from("sandbox-connect-denied"),
        OsString::from("192.0.2.1:9"),
    ]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let process = OwnedProcess::start_sandboxed(
        &spec,
        &SandboxPolicy {
            readonly_paths: vec![spec.executable.clone()],
            writable_directory: scratch.clone(),
        },
    )?;
    assert_eq!(
        process.wait_for_exit(TIMEOUT)?,
        Some(0),
        "{}",
        fs::read_to_string(&spec.stderr)?
    );
    assert_eq!(
        fs::read_to_string(scratch.join("network-denied.txt"))?,
        "WSAEACCES"
    );
    Ok(())
}

#[test]
fn argv_environment_and_native_cwd() -> io::Result<()> {
    let lane = Lane::new("arguments")?;
    let arguments = vec![
        OsString::from("report"),
        OsString::new(),
        OsString::from("with space"),
        OsString::from("a\\\"quoted\"\\"),
        OsString::from_wide(&[0xd800, 0x5c, 0x22]),
    ];
    let process = OwnedProcess::start(&lane.spec(arguments.clone()))?;
    assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
    let text = fs::read_to_string(lane.path.join("report.txt"))?;
    assert!(text.contains(&format!("cwd={}\n", lane.path.display())));
    for argument in arguments.iter().skip(1) {
        assert!(text.contains(&format!(
            "arg={:?}\n",
            argument.encode_wide().collect::<Vec<_>>()
        )));
    }
    for name in [
        "SystemDrive",
        "SystemRoot",
        "windir",
        "ProgramData",
        "LOCALAPPDATA",
        "APPDATA",
    ] {
        assert!(text.contains(&format!("{name}={:?}\n", std::env::var_os(name))));
    }
    assert!(available_memory_bytes()? > 0);
    // Evidence is create-new, so a repeated launch cannot overwrite it.
    assert!(OwnedProcess::start(&lane.spec(vec![OsString::from("report")])).is_err());
    Ok(())
}

#[test]
fn background_tool_job_lowers_root_and_nested_descendant_priority() -> io::Result<()> {
    use windows_sys::Win32::System::Threading::BELOW_NORMAL_PRIORITY_CLASS;

    let (inherited, results) = std::thread::scope(|scope| -> io::Result<_> {
        let ctrl_h = scope.spawn(|| -> io::Result<String> {
            let control = Lane::new("priority-control")?;
            let status = Command::new(probe_executable())
                .arg("priority-report")
                .current_dir(&control.path)
                .creation_flags(0x0800_0000)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()?;
            assert!(status.success());
            fs::read_to_string(control.path.join("priority.txt"))
        });
        let arm_handles: Vec<_> = [false, true]
            .into_iter()
            .map(|background| {
                scope.spawn(move || -> io::Result<(bool, String, String)> {
                    let lane = Lane::new("priority-tree")?;
                    let spec = lane.spec(vec![OsString::from("priority-tree")]);
                    let tool = ToolProcessSpec {
                        executable: spec.executable,
                        arguments: spec.arguments,
                        working_directory: spec.working_directory,
                        stdout: spec.stdout,
                        stderr: spec.stderr,
                        environment: std::collections::BTreeMap::new(),
                    };
                    let process = if background {
                        OwnedProcess::start_background_tool(&tool)?
                    } else {
                        OwnedProcess::start_tool(&tool)?
                    };
                    until(|| {
                        if process.try_wait()?.is_some() {
                            return Err(io::Error::other("priority probe exited before reporting"));
                        }
                        Ok(lane.path.join("child-priority.txt").is_file())
                    })?;
                    let root_p = fs::read_to_string(lane.path.join("root-priority.txt"))?;
                    let child_p = fs::read_to_string(lane.path.join("child-priority.txt"))?;
                    assert_eq!(process.active_processes()?, 2);
                    process.stop(TIMEOUT)?;
                    assert!(process.try_wait()?.is_some());
                    assert_eq!(process.active_processes()?, 0);
                    Ok((background, root_p, child_p))
                })
            })
            .collect();
        let inherited = ctrl_h
            .join()
            .map_err(|_| io::Error::other("priority control panicked"))??;
        let mut results = Vec::with_capacity(2);
        for handle in arm_handles {
            results.push(
                handle
                    .join()
                    .map_err(|_| io::Error::other("priority arm panicked"))??,
            );
        }
        Ok((inherited, results))
    })?;
    for (background, root_p, child_p) in results {
        let expected = if background {
            BELOW_NORMAL_PRIORITY_CLASS.to_string()
        } else {
            inherited.clone()
        };
        assert_eq!(root_p, expected);
        assert_eq!(child_p, expected);
    }
    Ok(())
}

#[test]
fn stop_and_drop_kill_descendants_and_observe_memory() -> io::Result<()> {
    let arms = [(true, unused_port()?), (false, unused_port()?)];
    std::thread::scope(|scope| -> io::Result<()> {
        let handles: Vec<_> = arms
            .into_iter()
            .map(|(explicitly_stop, port)| {
                scope.spawn(move || -> io::Result<()> {
                    let lane = Lane::new("tree")?;
                    let process = OwnedProcess::start(&lane.spec(vec![
                        OsString::from("tree"),
                        OsString::from(port.to_string()),
                    ]))?;
                    until(|| listener_process_id(port).map(|owner| owner.is_some()))?;
                    let descendant = listener_process_id(port)?
                        .ok_or_else(|| io::Error::other("listener disappeared"))?;
                    assert_ne!(descendant, process.id());
                    // Hosted run 41333126433 identified two extra conhost.exe children, one
                    // per console-subsystem probe. The fixture now links without a console.
                    // Require exactly the two intended nodes within the existing bound;
                    // the root and same listening descendant must stay live throughout.
                    // Extra processes still fail and remain in bounded failure diagnostics.
                    let mut observed = 0;
                    until(|| {
                        if process.try_wait()?.is_some()
                            || listener_process_id(port)? != Some(descendant)
                        {
                            return Err(io::Error::other(
                                "owned root or descendant exited before tree accounting settled",
                            ));
                        }
                        observed = process.active_processes()?;
                        Ok(observed == 2)
                    })
                    .map_err(|error| {
                        io::Error::new(
                            error.kind(),
                            format!(
                                "two-process tree accounting did not settle (root={}, descendant={descendant}, last={observed}): {error}; diagnostic={:?}",
                                process.id(), process.test_job_accounting()
                            ),
                        )
                    })?;
                    assert_eq!(
                        process.active_processes()?,
                        2,
                        "{:?}",
                        process.test_job_accounting()
                    );
                    assert!(process.memory_usage()?.private_bytes > 0);
                    assert!(process.memory_usage()?.working_set_bytes > 0);
                    assert!(process.peak_job_memory_bytes()? > 0);
                    if explicitly_stop {
                        assert!(process.stop(Duration::ZERO).is_err());
                        process.stop(TIMEOUT)?;
                        assert!(process.try_wait()?.is_some());
                        assert_eq!(process.active_processes()?, 0);
                    }
                    drop(process);
                    until(|| listener_process_id(port).map(|owner| owner.is_none()))?;
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle
                .join()
                .map_err(|_| io::Error::other("tree arm panicked"))??;
        }
        Ok(())
    })
}

#[test]
fn active_process_limit_blocks_descendants() -> io::Result<()> {
    let lane = Lane::new("process-limit")?;
    let mut spec = lane.spec(vec![OsString::from("spawn-denied")]);
    spec.active_process_limit = 1;
    let process = OwnedProcess::start(&spec)?;
    assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
    assert_eq!(fs::read_to_string(lane.path.join("spawn.txt"))?, "denied");
    Ok(())
}

#[test]
fn headless_mitigations_apply_at_creation_without_changing_tool_policy() -> io::Result<()> {
    let (baseline, observed, tool_observed, nested_observed) =
        std::thread::scope(|scope| -> io::Result<_> {
            let ctrl_h = scope.spawn(|| -> io::Result<String> {
                let control = Lane::new("mitigation-control")?;
                let status = Command::new(probe_executable())
                    .arg("mitigations")
                    .current_dir(&control.path)
                    .creation_flags(0x0800_0000)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()?;
                assert!(status.success());
                fs::read_to_string(control.path.join("mitigations.txt"))
            });
            let worker_h = scope.spawn(|| -> io::Result<String> {
                let worker = Lane::new("mitigation-worker")?;
                let process =
                    OwnedProcess::start(&worker.spec(vec![OsString::from("mitigations")]))?;
                assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
                fs::read_to_string(worker.path.join("mitigations.txt"))
            });
            let tool_h = scope.spawn(|| -> io::Result<String> {
                let tool = Lane::new("mitigation-tool")?;
                let spec = tool.spec(vec![OsString::from("mitigations")]);
                let process = OwnedProcess::start_tool(&ToolProcessSpec {
                    executable: spec.executable,
                    arguments: spec.arguments,
                    working_directory: spec.working_directory,
                    stdout: spec.stdout,
                    stderr: spec.stderr,
                    environment: std::collections::BTreeMap::new(),
                })?;
                assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
                fs::read_to_string(tool.path.join("mitigations.txt"))
            });
            let nested_h = scope.spawn(|| -> io::Result<String> {
                // A second owner inside the restricted worker must retain Job nesting.
                // JobObjectBasicUIRestrictions would prevent this valid hierarchy.
                let nested = Lane::new("mitigation-nested")?;
                let process =
                    OwnedProcess::start(&nested.spec(vec![OsString::from("nested-mitigations")]))?;
                assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
                fs::read_to_string(nested.path.join("mitigations.txt"))
            });
            Ok((
                ctrl_h
                    .join()
                    .map_err(|_| io::Error::other("mitigation control panicked"))??,
                worker_h
                    .join()
                    .map_err(|_| io::Error::other("mitigation worker panicked"))??,
                tool_h
                    .join()
                    .map_err(|_| io::Error::other("mitigation tool panicked"))??,
                nested_h
                    .join()
                    .map_err(|_| io::Error::other("mitigation nested panicked"))??,
            ))
        })?;
    for (name, required) in [("win32k", 1), ("extensions", 1), ("images", 3)] {
        let flags = observed
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name}=")))
            .ok_or_else(|| io::Error::other("missing mitigation report"))?
            .parse::<u32>()
            .map_err(io::Error::other)?;
        assert_eq!(flags & required, required, "{name}: {observed}");
    }
    assert_eq!(tool_observed, baseline);
    assert_eq!(nested_observed, observed);
    Ok(())
}

#[test]
fn inspection_keeps_non_win32k_mitigations_and_rejects_worker_arguments() -> io::Result<()> {
    let invalid = Lane::new("inspection-invalid")?;
    for arguments in [
        Vec::new(),
        vec![OsString::from("mitigations")],
        vec![OsString::from("--version"), OsString::from("--model")],
    ] {
        let mut spec = invalid.spec(arguments);
        spec.active_process_limit = 1;
        let error = OwnedProcess::start_inspection(&spec)
            .err()
            .ok_or_else(|| io::Error::other("worker arguments accepted for inspection"))?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(!spec.stdout.exists());
        assert!(!spec.stderr.exists());
    }
    let invalid_count = invalid.spec(vec![OsString::from("--version")]);
    assert!(OwnedProcess::start_inspection(&invalid_count).is_err());
    assert!(!invalid_count.stdout.exists());
    assert!(!invalid_count.stderr.exists());

    std::thread::scope(|scope| -> io::Result<()> {
        let ctrl_h = scope.spawn(|| -> io::Result<()> {
            let control = Lane::new("inspection-control")?;
            let status = Command::new(probe_executable())
                .arg("--version")
                .current_dir(&control.path)
                .creation_flags(0x0800_0000)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()?;
            assert!(status.success());
            assert_eq!(
                fs::read_to_string(control.path.join("inspection.txt"))?,
                "spawn=allowed\n"
            );
            Ok(())
        });
        let probe_handles: Vec<_> = ["--version", "--help"]
            .into_iter()
            .map(|argument| {
                scope.spawn(move || -> io::Result<()> {
                    let lane = Lane::new("inspection-mitigations")?;
                    let mut spec = lane.spec(vec![OsString::from(argument)]);
                    spec.active_process_limit = 1;
                    let process = OwnedProcess::start_inspection(&spec)?;
                    assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
                    let observed = fs::read_to_string(lane.path.join("mitigations.txt"))?;
                    for (name, expected) in [("win32k", 0), ("extensions", 1), ("images", 3)] {
                        assert!(
                            observed
                                .lines()
                                .any(|line| line == format!("{name}={expected}")),
                            "{observed}"
                        );
                    }
                    assert_eq!(
                        fs::read_to_string(lane.path.join("inspection.txt"))?,
                        "spawn=denied\n"
                    );
                    Ok(())
                })
            })
            .collect();
        ctrl_h
            .join()
            .map_err(|_| io::Error::other("inspection control panicked"))??;
        for handle in probe_handles {
            handle
                .join()
                .map_err(|_| io::Error::other("inspection probe panicked"))??;
        }
        Ok(())
    })
}

#[test]
fn graphics_probe_keeps_non_win32k_mitigations_and_rejects_worker_arguments() -> io::Result<()> {
    let invalid = Lane::new("graphics-probe-invalid")?;
    let valid_request = invalid.path.join("request.json").into_os_string();
    for arguments in [
        Vec::new(),
        vec![OsString::from("host-gpu-query")],
        vec![
            OsString::from("mitigations"),
            OsString::from("--request"),
            valid_request.clone(),
        ],
        vec![
            OsString::from("host-gpu-query"),
            OsString::from("--other"),
            valid_request.clone(),
        ],
        vec![
            OsString::from("host-gpu-query"),
            OsString::from("--request"),
            OsString::from("relative.json"),
        ],
    ] {
        let mut spec = invalid.spec(arguments);
        spec.active_process_limit = 1;
        let error = OwnedProcess::start_graphics_probe(&spec)
            .err()
            .ok_or_else(|| io::Error::other("invalid arguments accepted for graphics probe"))?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(!spec.stdout.exists());
        assert!(!spec.stderr.exists());
    }
    let invalid_count = invalid.spec(vec![
        OsString::from("host-gpu-query"),
        OsString::from("--request"),
        valid_request,
    ]);
    assert!(OwnedProcess::start_graphics_probe(&invalid_count).is_err());
    assert!(!invalid_count.stdout.exists());
    assert!(!invalid_count.stderr.exists());

    std::thread::scope(|scope| -> io::Result<()> {
        let handles: Vec<_> = [
            "host-gpu-query",
            "host-opencl-query",
            "opencl-projection-worker",
        ]
        .into_iter()
        .map(|subcommand| {
            scope.spawn(move || -> io::Result<()> {
                let lane = Lane::new("graphics-probe-mitigations")?;
                let request = lane.path.join("request.json");
                fs::write(&request, b"{}")?;
                let mut spec = lane.spec(vec![
                    OsString::from(subcommand),
                    OsString::from("--request"),
                    request.into_os_string(),
                ]);
                spec.active_process_limit = 1;
                let process = OwnedProcess::start_graphics_probe(&spec)?;
                assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
                let observed = fs::read_to_string(lane.path.join("mitigations.txt"))?;
                for (name, expected) in [("win32k", 0), ("extensions", 1), ("images", 3)] {
                    assert!(
                        observed
                            .lines()
                            .any(|line| line == format!("{name}={expected}")),
                        "{observed}"
                    );
                }
                assert_eq!(
                    fs::read_to_string(lane.path.join("inspection.txt"))?,
                    "spawn=denied\n"
                );
                Ok(())
            })
        })
        .collect();
        for handle in handles {
            handle
                .join()
                .map_err(|_| io::Error::other("graphics probe panicked"))??;
        }
        Ok(())
    })
}

#[test]
fn committed_memory_limit_denies_allocations() -> io::Result<()> {
    let lane = Lane::new("memory-limit")?;
    let mut spec = lane.spec(vec![OsString::from("commit")]);
    spec.memory_limit_bytes = 96 * 1024 * 1024;
    let process = OwnedProcess::start(&spec)?;
    assert_eq!(process.wait_for_exit(TIMEOUT)?, Some(0));
    assert_eq!(fs::read_to_string(lane.path.join("commit.txt"))?, "denied");
    let peak = process.peak_job_memory_bytes()?;
    assert!(peak > 32 * 1024 * 1024);
    // Windows may include a denied allocation request in this diagnostic peak.
    // Enforcement is established by the successful small allocations followed
    // by the fallible allocation being denied, not by an upper peak bound.
    Ok(())
}

#[test]
fn abrupt_owner_death_kills_entire_tree() -> io::Result<()> {
    let lane = Lane::new("owner-death")?;
    let port = unused_port()?;
    let mut owner = Command::new(probe_executable())
        .args(["owner", &port.to_string()])
        .current_dir(&lane.path)
        .creation_flags(0x0800_0000)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let result = until(|| listener_process_id(port).map(|pid| pid.is_some()));
    owner.kill()?;
    let _ = owner.wait()?;
    result?;
    until(|| listener_process_id(port).map(|pid| pid.is_none()))?;
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires native Windows PSEC v1.1 with terminate-on-close support."]
fn native_sandbox_abrupt_owner_death_ends_worker_and_listener() -> io::Result<()> {
    native_sandbox_available()?;
    let lane = Lane::new("sandbox-owner-death")?;
    let port = unused_port()?;
    let mut owner = Command::new(probe_executable())
        .args(["sandbox-owner", &port.to_string()])
        .current_dir(&lane.path)
        .creation_flags(0x0800_0000)
        .stdin(Stdio::null())
        .stdout(Stdio::from(fs::File::create_new(
            lane.path.join("owner.stdout"),
        )?))
        .stderr(Stdio::from(fs::File::create_new(
            lane.path.join("owner.stderr"),
        )?))
        .spawn()?;
    let ready_path = lane.path.join("sandbox-owner-ready.txt");
    let observed = (|| {
        until(|| {
            if ready_path.exists() {
                return Ok(true);
            }
            if let Some(status) = owner.try_wait()? {
                return Err(io::Error::other(format!(
                    "sandbox owner exited {status}: {}",
                    fs::read_to_string(lane.path.join("owner.stderr"))?
                )));
            }
            Ok(false)
        })?;
        let identity = fs::read_to_string(ready_path)?;
        let mut fields = identity.lines();
        let id = fields
            .next()
            .ok_or_else(|| io::Error::other("missing worker PID"))?
            .parse::<u32>()
            .map_err(io::Error::other)?;
        let creation = fields
            .next()
            .ok_or_else(|| io::Error::other("missing worker creation time"))?
            .parse::<u64>()
            .map_err(io::Error::other)?;
        let tracked = ParentProcess::open(id, creation)?;
        tracked.ensure_alive()?;
        until(|| listener_process_id(port).map(|pid| pid == Some(id)))?;
        Ok::<_, io::Error>(tracked)
    })();
    // Always end the disposable owner, even when readiness validation fails.
    let killed = owner.kill();
    let waited = owner.wait();
    let tracked = observed?;
    killed?;
    let _status = waited?;
    until(|| match tracked.ensure_alive() {
        Ok(()) => Ok(false),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(true),
        Err(error) => Err(error),
    })?;
    until(|| listener_process_id(port).map(|pid| pid.is_none()))?;
    Ok(())
}

#[test]
fn backend_overrides_are_rejected_without_changing_parent_environment() -> io::Result<()> {
    let lane = Lane::new("environment")?;
    let status = Command::new(probe_executable())
        .args(["owner", &unused_port()?.to_string()])
        .env("LLAMA_TEST_OVERRIDE", "1")
        .current_dir(&lane.path)
        .creation_flags(0x0800_0000)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    assert!(!status.success());
    assert!(!lane.path.join("owned.stdout").exists());
    Ok(())
}

#[test]
fn active_standard_evidence_allows_readers_and_rejects_other_writers() -> io::Result<()> {
    let lane = Lane::new("stream-sharing")?;
    let process = OwnedProcess::start(&lane.spec(vec![OsString::from("sleep")]))?;
    for name in ["stdout.log", "stderr.log"] {
        let path = lane.path.join(name);
        let _reader = fs::OpenOptions::new()
            .read(true)
            .share_mode(3)
            .open(&path)?;
        let writer = fs::OpenOptions::new().write(true).share_mode(3).open(&path);
        assert_eq!(
            writer.err().and_then(|error| error.raw_os_error()),
            Some(32)
        );
    }
    process.stop(TIMEOUT)?;
    Ok(())
}

#[test]
fn headless_error_policy_applies_to_root_and_descendants_only() -> io::Result<()> {
    let lane = Lane::new("error-policy")?;
    let status = Command::new(probe_executable())
        .arg("error-policy-owner")
        .current_dir(&lane.path)
        .creation_flags(0x0800_0000)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    assert!(status.success());
    assert_eq!(
        fs::read_to_string(lane.path.join("crash-exit.txt"))?,
        0xe042_4242_u32.to_string()
    );
    assert_eq!(
        fs::read_to_string(lane.path.join("error-policy-exit.txt"))?,
        "0"
    );
    assert_eq!(
        fs::read_to_string(lane.path.join("child-crash.txt"))?,
        format!(
            "Some({})",
            i32::from_ne_bytes(0xe042_4242_u32.to_ne_bytes())
        )
    );
    assert_eq!(
        fs::read_to_string(lane.path.join("owner-error-mode-before.txt"))?,
        "0"
    );
    assert_eq!(
        fs::read_to_string(lane.path.join("owner-error-mode-after.txt"))?,
        "0"
    );
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires a native Windows PSEC v1.1 host."]
fn native_sandbox_broker_pipe_enforces_offline_inference_authority() -> io::Result<()> {
    native_sandbox_offline_available()?;
    let lane = Lane::new("broker-pipe")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let approved = lane.path.join("approved-model.mxc");
    let unrelated = lane.path.join("unrelated.key");
    fs::write(&approved, "approved audio input")?;
    fs::write(&unrelated, "unrelated credential")?;
    // Trusted host controls prove write/read and loopback accept work outside containment.
    drop(fs::OpenOptions::new().write(true).open(&approved)?);
    let sentinel = fs::File::open(&unrelated)?;
    assert_eq!(fs::read_to_string(&unrelated)?, "unrelated credential");
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let endpoint = listener.local_addr()?;
    drop(TcpStream::connect_timeout(
        &endpoint,
        Duration::from_secs(1),
    )?);
    drop(listener.accept()?);

    let mut spec = lane.spec(vec![
        OsString::from("sandbox-broker-pipe"),
        approved.clone().into_os_string(),
        unrelated.clone().into_os_string(),
        OsString::from(endpoint.port().to_string()),
        OsString::from(inheritable_file_identity(&sentinel)?),
    ]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let (process, mut input, output) = OwnedProcess::start_sandboxed_broker_pipe(
        &spec,
        &SandboxPolicy {
            readonly_paths: vec![spec.executable.clone(), approved.clone()],
            writable_directory: scratch.clone(),
        },
    )?;
    assert_eq!(process.active_processes()?, 1);
    let expected = (0..=255_u8).cycle().take(131_079).collect::<Vec<_>>();
    let payload = expected.clone();
    let exchange = std::thread::spawn(move || -> io::Result<Vec<u8>> {
        input.write_all(&payload)?;
        drop(input);
        let mut response = Vec::new();
        output.take(131_088).read_to_end(&mut response)?;
        Ok(response)
    });
    let exit = process.wait_for_exit(TIMEOUT);
    let stopped = process.stop(TIMEOUT);
    stopped?;
    let response = exchange
        .join()
        .map_err(|_| io::Error::other("broker-pipe fixture panicked"))?;
    assert_eq!(
        exit?,
        Some(0),
        "{} | {}",
        fs::read_to_string(scratch.join("broker-pipe-error.txt")).unwrap_or_default(),
        fs::read_to_string(&spec.stderr).unwrap_or_default()
    );
    let response = response?;
    assert_eq!(process.active_processes()?, 0);
    assert_eq!(response.len(), 8 + expected.len());
    assert_eq!(&response[..8], b"SB2OKAY1");
    assert_eq!(&response[8..], expected.as_slice());
    assert_eq!(fs::read_to_string(&approved)?, "approved audio input");
    assert_eq!(fs::read_to_string(&unrelated)?, "unrelated credential");
    assert!(spec.stdout.exists());
    assert_eq!(fs::metadata(&spec.stdout)?.len(), 0);
    assert!(fs::read_to_string(&spec.stderr)?.contains("broker-pipe-stderr-marker"));

    let first_temp = fs::read_to_string(scratch.join("kernel-temp-path.txt"))?;
    assert_eq!(fs::read_to_string(&first_temp)?, "private kernel temp");
    let second_lane = Lane::new("broker-pipe-other-temp")?;
    let second_scratch = second_lane.path.join("scratch");
    fs::create_dir(&second_scratch)?;
    let mut second_spec = second_lane.spec(vec![
        OsString::from("sandbox-broker-pipe-temp-denied"),
        OsString::from(&first_temp),
    ]);
    second_spec.working_directory.clone_from(&second_scratch);
    second_spec.active_process_limit = 1;
    let (second_process, second_input, second_output) = OwnedProcess::start_sandboxed_broker_pipe(
        &second_spec,
        &SandboxPolicy {
            readonly_paths: vec![second_spec.executable.clone()],
            writable_directory: second_scratch.clone(),
        },
    )?;
    drop((second_input, second_output));
    let second_exit = second_process.wait_for_exit(TIMEOUT);
    second_process.stop(TIMEOUT)?;
    assert_eq!(
        second_exit?,
        Some(0),
        "{}",
        fs::read_to_string(&second_spec.stderr).unwrap_or_default()
    );
    assert_ne!(
        fs::read_to_string(second_scratch.join("kernel-temp-path.txt"))?,
        first_temp
    );
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires a native Windows PSEC v1.1 host."]
fn native_sandbox_broker_pipe_private_storage_budget_caps_scratch_and_temp() -> io::Result<()> {
    native_sandbox_offline_available()?;
    let lane = Lane::new("broker-pipe-storage")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let mut spec = lane.spec(vec![OsString::from("sandbox-broker-pipe-storage")]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let (process, mut input, mut output) = OwnedProcess::start_sandboxed_broker_pipe(
        &spec,
        &SandboxPolicy {
            readonly_paths: vec![spec.executable.clone()],
            writable_directory: scratch,
        },
    )?;

    input.write_all(b"1")?;
    input.flush()?;
    let mut ack = [0_u8; 7];
    output.read_exact(&mut ack)?;
    assert_eq!(&ack, b"phase1\n");

    let (bytes, entries) = process.private_storage_usage()?;
    assert!(bytes >= 12_288, "observed private storage bytes: {bytes}");
    assert!(entries >= 2, "observed private storage entries: {entries}");
    process.check_private_storage_budget(
        MAX_WORKER_PRIVATE_STORAGE_BYTES,
        MAX_WORKER_PRIVATE_STORAGE_ENTRIES,
    )?;
    let byte_error = process
        .check_private_storage_budget(10_000, MAX_WORKER_PRIVATE_STORAGE_ENTRIES)
        .err()
        .ok_or_else(|| io::Error::other("tight private byte budget unexpectedly passed"))?;
    assert_eq!(byte_error.kind(), io::ErrorKind::InvalidInput);

    input.write_all(b"2")?;
    input.flush()?;
    output.read_exact(&mut ack)?;
    assert_eq!(&ack, b"phase2\n");

    let entry_error = process
        .check_private_storage_budget(
            MAX_WORKER_PRIVATE_STORAGE_BYTES,
            MAX_WORKER_PRIVATE_STORAGE_ENTRIES,
        )
        .err()
        .ok_or_else(|| io::Error::other("private storage entry burst did not fail budget"))?;
    assert_eq!(entry_error.kind(), io::ErrorKind::InvalidInput);

    input.write_all(b"3")?;
    input.flush()?;
    drop((input, output));
    let exit = process.wait_for_exit(TIMEOUT);
    process.stop(TIMEOUT)?;
    assert_eq!(
        exit?,
        Some(0),
        "{}",
        fs::read_to_string(&spec.stderr).unwrap_or_default()
    );
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires a native Windows PSEC v1.1 host."]
fn native_sandbox_broker_pipe_abrupt_owner_death_ends_worker() -> io::Result<()> {
    native_sandbox_offline_available()?;
    let lane = Lane::new("broker-pipe-owner-death")?;
    let mut owner = Command::new(probe_executable())
        .arg("sandbox-broker-pipe-owner")
        .current_dir(&lane.path)
        .creation_flags(0x0800_0000)
        .stdin(Stdio::null())
        .stdout(Stdio::from(fs::File::create_new(
            lane.path.join("owner.stdout"),
        )?))
        .stderr(Stdio::from(fs::File::create_new(
            lane.path.join("owner.stderr"),
        )?))
        .spawn()?;
    let ready_path = lane.path.join("sandbox-broker-pipe-owner-ready.txt");
    let observed = (|| {
        until(|| {
            if ready_path.exists() {
                return Ok(true);
            }
            if let Some(status) = owner.try_wait()? {
                return Err(io::Error::other(format!(
                    "broker-pipe sandbox owner exited {status}: {}",
                    fs::read_to_string(lane.path.join("owner.stderr"))?
                )));
            }
            Ok(false)
        })?;
        let identity = fs::read_to_string(&ready_path)?;
        let mut fields = identity.lines();
        let id = fields
            .next()
            .ok_or_else(|| io::Error::other("missing broker-pipe worker PID"))?
            .parse::<u32>()
            .map_err(io::Error::other)?;
        let creation = fields
            .next()
            .ok_or_else(|| io::Error::other("missing broker-pipe worker creation time"))?
            .parse::<u64>()
            .map_err(io::Error::other)?;
        let tracked = ParentProcess::open(id, creation)?;
        tracked.ensure_alive()?;
        let owned_stdout = lane.path.join("owned.stdout");
        if !owned_stdout.exists() || fs::metadata(&owned_stdout)?.len() != 0 {
            return Err(io::Error::other(
                "expected 0-byte owned.stdout sentinel under broker-pipe",
            ));
        }
        Ok::<_, io::Error>(tracked)
    })();
    let killed = owner.kill();
    let waited = owner.wait();
    let tracked = observed?;
    killed?;
    let _status = waited?;
    until(|| match tracked.ensure_alive() {
        Ok(()) => Ok(false),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(true),
        Err(error) => Err(error),
    })?;
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
fn broker_pipe_with_sections_enforces_bounds_and_file_kinds_before_creating_streams()
-> io::Result<()> {
    use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;

    let lane = Lane::new("broker-pipe-section-bounds")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let mut spec = lane.spec(vec![OsString::from("sleep")]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;

    let valid_path = lane.path.join("model.bin");
    fs::write(&valid_path, b"model-section-bytes")?;
    let valid_file = fs::File::open(&valid_path)?;
    let policy = SandboxPolicy {
        readonly_paths: vec![spec.executable.clone()],
        writable_directory: scratch.clone(),
    };

    // More than MAX_INHERITED_SECTIONS (8) must reject with InvalidInput before creating streams.
    let too_many: Vec<&fs::File> = vec![&valid_file; MAX_INHERITED_SECTIONS + 1];
    let error = OwnedProcess::start_sandboxed_broker_pipe_with_sections(&spec, &policy, &too_many)
        .err()
        .ok_or_else(|| io::Error::other("excess inherited sections were accepted"))?;
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(!spec.stdout.exists());
    assert!(!spec.stderr.exists());

    // Empty (0-byte) files must reject with InvalidInput before creating streams.
    let empty_path = lane.path.join("empty.bin");
    fs::write(&empty_path, b"")?;
    let empty_file = fs::File::open(&empty_path)?;
    let error =
        OwnedProcess::start_sandboxed_broker_pipe_with_sections(&spec, &policy, &[&empty_file])
            .err()
            .ok_or_else(|| io::Error::other("empty section file was accepted"))?;
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(!spec.stdout.exists());
    assert!(!spec.stderr.exists());

    // Non-regular directory handles must reject with InvalidInput before creating streams.
    let dir_handle = fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(&lane.path)?;
    let error =
        OwnedProcess::start_sandboxed_broker_pipe_with_sections(&spec, &policy, &[&dir_handle])
            .err()
            .ok_or_else(|| io::Error::other("directory section handle was accepted"))?;
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(!spec.stdout.exists());
    assert!(!spec.stderr.exists());

    // Multi-process limit must reject cleanly with valid sections without creating streams.
    spec.active_process_limit = 2;
    let error =
        OwnedProcess::start_sandboxed_broker_pipe_with_sections(&spec, &policy, &[&valid_file])
            .err()
            .ok_or_else(|| {
                io::Error::other("multi-process broker-pipe section launch succeeded")
            })?;
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(!spec.stdout.exists());
    assert!(!spec.stderr.exists());

    // Exactly MAX_INHERITED_SECTIONS valid files followed by invalid policy drops all section
    // handles cleanly without creating evidence files.
    spec.active_process_limit = 1;
    let invalid_policy = SandboxPolicy {
        readonly_paths: Vec::new(),
        writable_directory: scratch,
    };
    let exact_max: Vec<&fs::File> = vec![&valid_file; MAX_INHERITED_SECTIONS];
    let error =
        OwnedProcess::start_sandboxed_broker_pipe_with_sections(&spec, &invalid_policy, &exact_max)
            .err()
            .ok_or_else(|| io::Error::other("invalid sandbox policy launched with max sections"))?;
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(!spec.stdout.exists());
    assert!(!spec.stderr.exists());

    drop(dir_handle);
    drop(empty_file);
    drop(valid_file);
    let _ = fs::remove_dir_all(&lane.path);
    Ok(())
}

#[cfg(feature = "containment")]
#[test]
#[ignore = "Requires a native Windows PSEC v1.1 host."]
fn native_sandbox_broker_pipe_with_sections_inherits_grants_and_stops_cleanly() -> io::Result<()> {
    native_sandbox_offline_available()?;
    let lane = Lane::new("broker-pipe-sections")?;
    let scratch = lane.path.join("scratch");
    fs::create_dir(&scratch)?;
    let section_a_path = lane.path.join("section-a.bin");
    let section_b_path = lane.path.join("section-b.bin");
    fs::write(&section_a_path, b"alpha-weights-fixture")?;
    fs::write(&section_b_path, b"beta-weights-fixture-extra")?;
    let file_a = fs::File::open(&section_a_path)?;
    let file_b = fs::File::open(&section_b_path)?;

    let mut spec = lane.spec(vec![OsString::from("sandbox-broker-pipe-wait")]);
    spec.working_directory.clone_from(&scratch);
    spec.active_process_limit = 1;
    let policy = SandboxPolicy {
        readonly_paths: vec![spec.executable.clone()],
        writable_directory: scratch,
    };
    let (process, input, mut output, grants) =
        OwnedProcess::start_sandboxed_broker_pipe_with_sections(
            &spec,
            &policy,
            &[&file_a, &file_b],
        )?;
    assert_eq!(grants.len(), 2);
    assert_eq!(grants[0].slot, 0);
    assert_eq!(grants[0].bytes, 21);
    assert_ne!(grants[0].child_handle, 0);
    assert_eq!(grants[1].slot, 1);
    assert_eq!(grants[1].bytes, 26);
    assert_ne!(grants[1].child_handle, 0);
    assert_ne!(grants[0].child_handle, grants[1].child_handle);
    assert_eq!(process.active_processes()?, 1);

    let mut alive = [0_u8; 6];
    output.read_exact(&mut alive)?;
    assert_eq!(&alive, b"alive\n");

    drop((input, output));
    process.stop(TIMEOUT)?;
    assert_eq!(process.active_processes()?, 0);
    assert!(spec.stdout.exists());
    assert_eq!(fs::metadata(&spec.stdout)?.len(), 0);
    assert!(fs::read_to_string(&spec.stderr)?.contains("stderr-alive"));
    drop((file_a, file_b));
    let _ = fs::remove_dir_all(&lane.path);
    Ok(())
}
