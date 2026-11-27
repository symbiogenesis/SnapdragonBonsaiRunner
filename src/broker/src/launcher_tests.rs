//! Synthetic owned-backend lifecycle regressions; no model process is executed.

use super::*;
use crate::{
    pipe_transport::{
        KIND_DATA, KIND_EOF, KIND_OPEN, KIND_RESET, KIND_SERVER_READY, read_test_frame_opt,
        write_test_frame,
    },
    test_util::TempDir,
};
use std::{
    fs,
    future::pending,
    io::{Read, Write},
    net::{TcpListener as BlockingListener, TcpStream as BlockingStream},
};

fn synthetic_launched_service(
    service: LocalChatService,
    process: OwnedProcess,
    credentials: PrivateCredentials,
    backend_port: u16,
    storage_budget: Option<(u64, usize)>,
) -> LaunchedService {
    LaunchedService {
        service,
        process,
        credentials,
        _artifacts: VerifiedArtifacts::synthetic_for_test(),
        _daisyrose: None,
        _mmproj: None,
        _native_executable: None,
        memory_limit_bytes: PROBE_MEMORY,
        working_set_limit_bytes: PROBE_MEMORY,
        backend_port,
        storage_budget,
        worker_root_covered_by_process: false,
    }
}

fn parse_native_launch_config(
    data: &std::path::Path,
    extra: &[&str],
) -> Result<LaunchConfig, LaunchError> {
    let mut args = vec![
        OsString::from("--launch-development"),
        "--allow-unisolated-backend".into(),
        "--native-engine".into(),
    ];
    args.extend(extra.iter().map(OsString::from));
    args.extend([
        "--profile".into(),
        "swift-pq2-assessment".into(),
        "--data-directory".into(),
        data.as_os_str().to_owned(),
        "--evidence-parent".into(),
        data.as_os_str().to_owned(),
        "--frontend-port".into(),
        "0".into(),
        "--backend-port".into(),
        "8081".into(),
    ]);
    LaunchConfig::parse(args)
}

fn native_launch_config(data: &std::path::Path, extra: &[&str]) -> io::Result<LaunchConfig> {
    parse_native_launch_config(data, extra)
        .map_err(|_| io::Error::other("Invalid fixture launch configuration."))
}

const PROBE_NAME: &str = "launcher::lifecycle_tests::synthetic_backend_probe";
const HOLDER_PROBE_NAME: &str = "launcher::lifecycle_tests::synthetic_pipe_holder_probe";
const PROBE_MEMORY: usize = 512 * 1024 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

struct Fixture {
    process: OwnedProcess,
    credentials: PrivateCredentials,
    port: u16,
}

fn holder_probe(directory: &std::path::Path) -> io::Result<OwnedProcess> {
    let specification = ProcessSpec {
        executable: std::env::current_exe()?,
        arguments: ["--exact", HOLDER_PROBE_NAME, "--ignored", "--nocapture"]
            .into_iter()
            .map(OsString::from)
            .collect(),
        working_directory: directory.to_path_buf(),
        stdout: directory.join("probe.stdout"),
        stderr: directory.join("probe.stderr"),
        memory_limit_bytes: PROBE_MEMORY,
        active_process_limit: 1,
    };
    OwnedProcess::start(&specification)
}

fn probe() -> io::Result<Fixture> {
    let parent: PathBuf = std::env::temp_dir().components().collect();
    let credentials = PrivateCredentials::create(&parent)?;
    let reservation = BlockingListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let port = reservation.local_addr()?.port();
    fs::write(credentials.directory().join("probe.port"), port.to_string())?;
    let specification = ProcessSpec {
        executable: std::env::current_exe()?,
        arguments: ["--exact", PROBE_NAME, "--ignored", "--nocapture"]
            .into_iter()
            .map(OsString::from)
            .collect(),
        working_directory: credentials.directory().to_path_buf(),
        stdout: credentials.directory().join("probe.stdout"),
        stderr: credentials.directory().join("probe.stderr"),
        memory_limit_bytes: PROBE_MEMORY,
        active_process_limit: 1,
    };
    drop(reservation);
    let process = OwnedProcess::start(&specification)?;
    Ok(Fixture {
        process,
        credentials,
        port,
    })
}

fn check(test: impl Future<Output = io::Result<()>>) -> io::Result<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            timeout(PROBE_TIMEOUT, test).await.map_err(|_| {
                io::Error::new(io::ErrorKind::TimedOut, "Lifecycle probe timed out.")
            })?
        })
}

#[test]
fn missing_native_runtime_is_distinct_from_missing_model_artifacts() -> io::Result<()> {
    let data = TempDir::new("Bonsai-NativeRuntimePreflight")?;
    let mut config = native_launch_config(data.path(), &[])?;
    assert_eq!(
        config.native_executable_guard().err(),
        Some(LaunchError::NativeRuntime)
    );
    let plan = LaunchPlan::for_swift_bonsai25(
        data.path(),
        config.backend_port,
        &data.path().join("not-created.key"),
        bonsai_cpu_policy::Bonsai25Variant::Pq2,
    )
    .map_err(|_| io::Error::other("Invalid fixture launch policy."))?;
    assert_eq!(
        config
            .verify_artifacts(&plan, crate::artifacts::default_chunk_workers())
            .err(),
        Some(LaunchError::Artifacts)
    );
    config.native_engine = false;
    assert!(
        config
            .native_executable_guard()
            .is_ok_and(|value| value.is_none())
    );
    Ok(())
}

#[test]
fn parallel_launch_artifact_verification_reports_deterministic_error_precedence() -> io::Result<()>
{
    let data = TempDir::new("Bonsai-ParallelVerifyPrecedence")?;
    let mut config = native_launch_config(data.path(), &[])?;
    let plan = LaunchPlan::for_swift_bonsai25(
        data.path(),
        config.backend_port,
        &data.path().join("not-created.key"),
        bonsai_cpu_policy::Bonsai25Variant::Pq2,
    )
    .map_err(|_| io::Error::other("Invalid fixture launch policy."))?;
    for _ in 0..4 {
        assert_eq!(
            config.verify_launch_artifacts(&plan).err(),
            Some(LaunchError::NativeRuntime)
        );
    }
    config.native_engine = false;
    for _ in 0..4 {
        assert_eq!(
            config.verify_launch_artifacts(&plan).err(),
            Some(LaunchError::Artifacts)
        );
    }
    Ok(())
}

#[test]
fn owned_ready_backend_stops_with_service() -> io::Result<()> {
    check(async {
        let Fixture {
            process,
            credentials,
            port,
        } = probe()?;
        assert!(process.try_wait()?.is_none());
        assert_eq!(
            wait_ready(
                &process,
                port,
                credentials.backend_key(),
                PROBE_MEMORY,
                pending()
            )
            .await,
            Ok(())
        );
        assert_eq!(listener_process_id(port)?, Some(process.id()));
        assert!(healthy(port, credentials.backend_key(), process.id()).await);
        assert!(authenticated(port, credentials.backend_key(), process.id()).await);
        let wrong_key = if credentials.backend_key().starts_with('0') {
            "1".repeat(64)
        } else {
            "0".repeat(64)
        };
        assert!(!healthy(port, &wrong_key, process.id()).await);
        assert!(!authenticated(port, &wrong_key, process.id()).await);
        let mut service = LocalChatService::bind(
            0,
            credentials.frontend_key(),
            port,
            credentials.backend_key(),
            0,
        )
        .await
        .map_err(|_| io::Error::other("Synthetic service bind failed."))?;
        service.require_backend_process(process.id());
        let launched = synthetic_launched_service(service, process, credentials, port, None);
        let frontend_port = launched.port();
        let report = launched
            .run(async { Ok(()) })
            .await
            .map_err(|_| io::Error::other("Synthetic service cleanup failed."))?;
        assert_eq!(report.exit, RunExit::Shutdown);
        assert_eq!(report.completed, 0);
        assert_eq!(report.failed, 0);
        assert_eq!(report.peak_in_flight, 0);
        let resources = report
            .backend_resources
            .ok_or_else(|| io::Error::other("Missing owned resource observation."))?;
        assert!(
            resources
                .os_peak_job_commit_bytes
                .is_some_and(|bytes| bytes > 0)
        );
        assert!(
            resources
                .os_peak_root_working_set_bytes
                .is_some_and(|bytes| bytes > 0)
        );
        assert_eq!(listener_process_id(port)?, None);
        assert_eq!(listener_process_id(frontend_port)?, None);
        Ok(())
    })
}

#[test]
fn resource_observations_keep_sampled_and_lifetime_counters_distinct() {
    let observed = BackendResources {
        os_peak_job_commit_bytes: Some(1000),
        os_peak_root_working_set_bytes: Some(900),
        ..BackendResources::default()
    };
    let first = resource_sample(
        observed,
        ProcessMemory {
            private_bytes: 80,
            working_set_bytes: 70,
            peak_working_set_bytes: 900,
        },
        5000,
    );
    let second = resource_sample(
        first,
        ProcessMemory {
            private_bytes: 60,
            working_set_bytes: 85,
            peak_working_set_bytes: 900,
        },
        6000,
    );
    assert_eq!(second.sample_count, 2);
    assert_eq!(second.sampled_peak_root_private_bytes, 80);
    assert_eq!(second.sampled_peak_root_working_set_bytes, 85);
    assert_eq!(second.sampled_min_available_physical_bytes, Some(5000));
    assert_eq!(second.os_peak_job_commit_bytes, Some(1000));
    assert_eq!(second.os_peak_root_working_set_bytes, Some(900));
    let saturated = resource_sample(
        BackendResources {
            sample_count: u64::MAX,
            ..second
        },
        ProcessMemory {
            private_bytes: 0,
            working_set_bytes: 0,
            peak_working_set_bytes: 900,
        },
        4000,
    );
    assert_eq!(saturated.sample_count, u64::MAX);
    assert_eq!(saturated.sampled_min_available_physical_bytes, Some(4000));
    assert_eq!(saturated.sampled_peak_root_private_bytes, 80);
}

#[test]
fn readiness_rejects_another_listener_and_cancellation() -> io::Result<()> {
    check(async {
        let fixture = probe()?;
        let impostor = BlockingListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let other_port = impostor.local_addr()?.port();
        assert_eq!(listener_process_id(other_port)?, Some(std::process::id()));
        assert_eq!(
            wait_ready(
                &fixture.process,
                other_port,
                fixture.credentials.backend_key(),
                PROBE_MEMORY,
                pending()
            )
            .await,
            Err(LaunchError::Backend)
        );
        assert_eq!(
            wait_ready(
                &fixture.process,
                fixture.port,
                fixture.credentials.backend_key(),
                PROBE_MEMORY,
                async { Ok(()) }
            )
            .await,
            Err(LaunchError::Cancelled)
        );
        fixture.process.stop(STOP_TIMEOUT)?;
        assert!(fixture.process.try_wait()?.is_some());
        assert_eq!(listener_process_id(fixture.port)?, None);
        Ok(())
    })
}

// The parent invokes this ignored test in a separate, atomically Job-owned child.
// Port discovery uses the protected native cwd; inherited environment is untouched.
#[test]
#[ignore = "subprocess entry point; invoked only with a protected fixture cwd"]
fn synthetic_backend_probe() -> io::Result<()> {
    let directory = std::env::current_dir()?;
    let port = fs::read_to_string(directory.join("probe.port"))?
        .parse::<u16>()
        .map_err(io::Error::other)?;
    let key = fs::read_to_string(directory.join("backend.key"))?;
    if key.len() != 64 || port < 1024 {
        return Err(io::Error::other("Invalid synthetic backend fixture."));
    }
    let listener = BlockingListener::bind((Ipv4Addr::LOCALHOST, port))?;
    for incoming in listener.incoming() {
        serve_probe(&mut incoming?, &key)?;
    }
    Ok(())
}

fn serve_probe(stream: &mut BlockingStream, key: &str) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let mut request = Vec::with_capacity(1024);
    let mut buffer = [0; 1024];
    while !request.array_windows().any(|bytes| bytes == b"\r\n\r\n") {
        let count = stream.read(&mut buffer)?;
        if count == 0 || request.len() + count > 16 * 1024 {
            return Err(io::Error::other("Invalid synthetic backend request."));
        }
        request.extend_from_slice(&buffer[..count]);
    }
    let request = std::str::from_utf8(&request).map_err(io::Error::other)?;
    let authorized = request
        .split("\r\n")
        .any(|line| line == format!("Authorization: Bearer {key}"));
    let (status, body) = if request.starts_with("GET /health HTTP/1.1\r\n") && authorized {
        ("200 OK", r#"{"status":"ok"}"#)
    } else if request.starts_with("GET /v1/models HTTP/1.1\r\n") && authorized {
        ("200 OK", r#"{"data":[]}"#)
    } else {
        ("401 Unauthorized", r#"{"error":"unauthorized"}"#)
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

#[test]
#[ignore = "subprocess entry point; invoked only as an owned holder child"]
fn synthetic_pipe_holder_probe() -> io::Result<()> {
    std::thread::sleep(Duration::from_secs(30));
    Ok(())
}

fn complete_http_request(buffer: &[u8]) -> Option<usize> {
    let head_end = buffer.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    let head = std::str::from_utf8(&buffer[..head_end]).ok()?;
    let content_length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    (buffer.len() >= head_end + content_length).then_some(head_end)
}

fn run_synthetic_pipe_server(
    mut reader: io::PipeReader,
    mut writer: io::PipeWriter,
    backend_key: &str,
) -> io::Result<(bool, bool)> {
    let json_reply = |status: &str, body: &str| {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    };

    let mut handshake = [0_u8; 8 + 64];
    reader.read_exact(&mut handshake)?;
    if &handshake[..8] != b"SB2PIPE1" || &handshake[8..] != backend_key.as_bytes() {
        return Err(io::Error::other("Invalid synthetic pipe handshake."));
    }
    writer.write_all(b"SB2OKAY1")?;
    writer.flush()?;
    write_test_frame(&mut writer, 0, KIND_SERVER_READY, &[])?;

    let expected_auth = format!("Authorization: Bearer {backend_key}");
    let mut active: Option<(u32, Vec<u8>)> = None;
    let mut chat_count = 0_u32;
    let mut saw_generation_reset = false;
    let mut saw_reset_post = false;

    while let Some((stream_id, kind, payload)) = read_test_frame_opt(&mut reader)? {
        match kind {
            KIND_OPEN => {
                if active.is_some() {
                    return Err(io::Error::other("Unexpected concurrent stream open."));
                }
                active = Some((stream_id, Vec::new()));
            }
            KIND_DATA => {
                let Some((active_id, buffer)) = active.as_mut() else {
                    continue;
                };
                if *active_id != stream_id {
                    return Err(io::Error::other("Data frame stream mismatch."));
                }
                buffer.extend_from_slice(&payload);
                let Some(head_end) = complete_http_request(buffer) else {
                    continue;
                };
                let head = std::str::from_utf8(&buffer[..head_end]).map_err(io::Error::other)?;
                let authorized = head.lines().any(|line| line == expected_auth);
                let response = if head.starts_with("GET /health HTTP/1.1\r\n") && authorized {
                    Some(json_reply("200 OK", r#"{"status":"ok"}"#))
                } else if head.starts_with("GET /v1/models HTTP/1.1\r\n") {
                    if authorized {
                        Some(json_reply("200 OK", r#"{"data":[]}"#))
                    } else {
                        Some(json_reply(
                            "401 Unauthorized",
                            r#"{"error":"unauthorized"}"#,
                        ))
                    }
                } else if head.starts_with("POST /reset HTTP/1.1\r\n") && authorized {
                    if head.to_ascii_lowercase().contains("content-type:") {
                        return Err(io::Error::other(
                            "POST /reset included unexpected Content-Type.",
                        ));
                    }
                    saw_reset_post = true;
                    Some(json_reply("200 OK", r#"{"status":"ok"}"#))
                } else if head.starts_with("POST /v1/chat/completions/input_tokens ") && authorized
                {
                    Some(json_reply(
                        "200 OK",
                        r#"{"object":"response.input_tokens","input_tokens":8}"#,
                    ))
                } else if head.starts_with("POST /v1/chat/completions ") && authorized {
                    chat_count += 1;
                    if chat_count == 1 {
                        let sse_chunk =
                            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n";
                        let partial = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{sse_chunk}\r\n",
                            sse_chunk.len()
                        );
                        write_test_frame(&mut writer, stream_id, KIND_DATA, partial.as_bytes())?;
                        None
                    } else {
                        let done = "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                        Some(format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{done}\r\n0\r\n\r\n",
                            done.len()
                        ))
                    }
                } else {
                    Some(json_reply(
                        "401 Unauthorized",
                        r#"{"error":"unauthorized"}"#,
                    ))
                };
                if let Some(response) = response {
                    write_test_frame(&mut writer, stream_id, KIND_DATA, response.as_bytes())?;
                    write_test_frame(&mut writer, stream_id, KIND_EOF, &[])?;
                    active = None;
                }
            }
            KIND_EOF => {
                if active
                    .as_ref()
                    .is_some_and(|(active_id, _)| *active_id == stream_id)
                {
                    active = None;
                }
            }
            KIND_RESET => {
                if active
                    .as_ref()
                    .is_some_and(|(active_id, _)| *active_id == stream_id)
                {
                    saw_generation_reset = true;
                    active = None;
                }
            }
            _ => return Err(io::Error::other("Unexpected frame kind from parent pipe.")),
        }
    }
    Ok((saw_generation_reset, saw_reset_post))
}

fn pipe_client_chat_request(port: u16, key: &str) -> String {
    let body = r#"{"model":"bonsai","messages":[{"role":"user","content":"hi"}],"stream":true,"max_completion_tokens":32}"#;
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

#[test]
fn broker_pipe_launched_service_serves_cancels_resets_and_avoids_backend_key_file() -> io::Result<()>
{
    check(async {
        let parent: PathBuf = std::env::temp_dir().components().collect();
        let credentials = PrivateCredentials::create_frontend_only(&parent)?;
        assert!(!credentials.has_backend_file());
        assert!(!credentials.backend_path().exists());
        let worker_dir = credentials.directory().join("worker");
        fs::create_dir(&worker_dir)?;

        let process = holder_probe(credentials.directory())?;
        let (p2c_rx, p2c_tx) = io::pipe()?;
        let (c2p_rx, c2p_tx) = io::pipe()?;

        let backend_key = credentials.backend_key().to_owned();
        let server_handle =
            std::thread::spawn(move || run_synthetic_pipe_server(p2c_rx, c2p_tx, &backend_key));

        let pipe_backend =
            crate::pipe_transport::PipeBackend::from_io(c2p_rx, p2c_tx, credentials.backend_key())
                .map_err(|_| io::Error::other("PipeBackend::from_io failed."))?;
        let models = wait_ready_models_pipe(
            ProcessGuard {
                process: &process,
                private_limit: PROBE_MEMORY,
                working_set_limit: PROBE_MEMORY,
                storage_budget: Some((
                    MAX_WORKER_PRIVATE_STORAGE_BYTES,
                    MAX_WORKER_PRIVATE_STORAGE_ENTRIES,
                )),
                worker_root: Some(worker_dir.as_path()),
            },
            &pipe_backend,
            credentials.backend_key(),
            pending(),
        )
        .await
        .map_err(|_| io::Error::other("wait_ready_models_pipe failed."))?;
        assert!(String::from_utf8_lossy(&models).contains("\"data\""));

        let mut service = LocalChatService::bind_pipe(
            0,
            credentials.frontend_key(),
            credentials.backend_key(),
            pipe_backend,
            1,
        )
        .await
        .map_err(|_| io::Error::other("bind_pipe failed."))?;
        service.enable_reset_on_cancel();

        let frontend_key = credentials.frontend_key().to_owned();
        let launched = synthetic_launched_service(
            service,
            process,
            credentials,
            0,
            Some((
                MAX_WORKER_PRIVATE_STORAGE_BYTES,
                MAX_WORKER_PRIVATE_STORAGE_ENTRIES,
            )),
        );
        let frontend_port = launched.port();
        let evidence_dir = launched.evidence_directory().to_path_buf();
        assert!(!evidence_dir.join("backend.key").exists());

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let client_task = tokio::spawn(async move {
            // 1. Start a streaming chat request and abandon mid-stream after the first delta.
            let mut first = TcpStream::connect((Ipv4Addr::LOCALHOST, frontend_port)).await?;
            first
                .write_all(pipe_client_chat_request(frontend_port, &frontend_key).as_bytes())
                .await?;
            first.flush().await?;
            let mut partial = Vec::new();
            let mut buf = [0_u8; 512];
            while !partial.windows(3).any(|window| window == b"Hel") {
                let count = first.read(&mut buf).await?;
                if count == 0 {
                    return Err(io::Error::other("First stream ended before SSE delta."));
                }
                partial.extend_from_slice(&buf[..count]);
            }
            drop(first);

            // 2. Follow-up request waits through KIND_RESET + POST /reset and completes.
            let mut second = TcpStream::connect((Ipv4Addr::LOCALHOST, frontend_port)).await?;
            second
                .write_all(pipe_client_chat_request(frontend_port, &frontend_key).as_bytes())
                .await?;
            second.flush().await?;
            let mut completed = Vec::new();
            second.read_to_end(&mut completed).await?;
            let response = String::from_utf8_lossy(&completed);
            if !response.contains("data: [DONE]") {
                return Err(io::Error::other("Second stream did not finish cleanly."));
            }
            let _ = shutdown_tx.send(());
            Ok::<(), io::Error>(())
        });

        let report = launched
            .run(async {
                let _ = shutdown_rx.await;
                Ok(())
            })
            .await
            .map_err(|_| io::Error::other("Broker-pipe LaunchedService failed."))?;
        client_task.await.map_err(io::Error::other)??;
        let (saw_generation_reset, saw_reset_post) = server_handle
            .join()
            .map_err(|_| io::Error::other("Synthetic pipe server panicked."))??;
        assert!(saw_generation_reset);
        assert!(saw_reset_post);
        assert_eq!(report.exit, RunExit::Shutdown);
        assert_eq!(report.completed, 1);
        assert!(!evidence_dir.join("backend.key").exists());
        let _ = fs::remove_dir_all(&evidence_dir);
        Ok(())
    })
}

#[test]
fn launched_service_stops_with_sandbox_error_when_storage_budget_exceeded() -> io::Result<()> {
    check(async {
        let parent: PathBuf = std::env::temp_dir().components().collect();
        let credentials = PrivateCredentials::create_frontend_only(&parent)?;
        let worker_dir = credentials.directory().join("worker");
        fs::create_dir(&worker_dir)?;
        let process = holder_probe(credentials.directory())?;

        let (p2c_rx, p2c_tx) = io::pipe()?;
        let (c2p_rx, c2p_tx) = io::pipe()?;
        let pipe_backend =
            crate::pipe_transport::PipeBackend::from_io(c2p_rx, p2c_tx, credentials.backend_key())
                .map_err(|_| io::Error::other("PipeBackend::from_io failed."))?;
        let service = LocalChatService::bind_pipe(
            0,
            credentials.frontend_key(),
            credentials.backend_key(),
            pipe_backend,
            0,
        )
        .await
        .map_err(|_| io::Error::other("bind_pipe failed."))?;

        fs::write(worker_dir.join("overflow.bin"), [0x5A_u8; 256])?;

        let launched =
            synthetic_launched_service(service, process, credentials, 0, Some((128, 16)));
        let evidence_dir = launched.evidence_directory().to_path_buf();
        let result = launched.run(pending()).await;
        drop((p2c_rx, c2p_tx));
        assert_eq!(result.err(), Some(LaunchError::Sandbox));
        let _ = fs::remove_dir_all(&evidence_dir);
        Ok(())
    })
}

#[test]
fn launch_usage_and_vulkan_composite_wy_f32_gates_match_pinned_contract() -> Result<(), LaunchError>
{
    for snippet in [
        "--prompt-reuse disabled|latest|prompt-end-and-latest",
        "--prompt-reuse-checkpoints 1..=8",
        "--prompt-reuse-checkpoint-bytes",
        "--cpu-prefill-gdn sequential|wy-f32",
        "--spec-type none|suffix",
        "--draft-max 1..=8",
        "--cpu-family x1|x2",
        "--placement auto|cpu|vulkan|composite",
    ] {
        assert!(
            LAUNCH_USAGE.contains(snippet),
            "LAUNCH_USAGE missing `{snippet}`"
        );
    }

    let data = std::env::temp_dir().join("Bonsai-vulkan-wyf32-gates");
    for (flag, expected) in [
        ("vulkan", crate::inference::Placement::Vulkan),
        ("composite", crate::inference::Placement::Composite),
    ] {
        let parsed = parse_native_launch_config(&data, &["--placement", flag])?;
        assert_eq!(parsed.placement, expected);
    }
    for wy_admitted in [
        &["--cpu-prefill-tokens", "16", "--cpu-prefill-gdn", "wy-f32"][..],
        &[
            "--cpu-prefill-tokens",
            "16",
            "--kv-cache",
            "rsq4",
            "--cpu-prefill-gdn",
            "wy-f32",
        ][..],
    ] {
        let parsed = parse_native_launch_config(&data, wy_admitted)?;
        assert_eq!(
            parsed.cpu_prefill_gdn,
            Some(crate::inference::CpuPrefillGdn::WyF32)
        );
    }
    assert_eq!(
        parse_native_launch_config(
            &data,
            &[
                "--cpu-prefill-tokens",
                "16",
                "--kv-cache",
                "rsq4k3v",
                "--cpu-prefill-gdn",
                "wy-f32",
            ]
        )
        .err(),
        Some(LaunchError::NativeRuntime)
    );

    let mut base = parse_native_launch_config(
        &data,
        &[
            "--cpu-prefill-tokens",
            "16",
            "--kv-cache",
            "f16",
            "--cpu-prefill-gdn",
            "wy-f32",
        ],
    )?;
    let key = data.join("backend.key");
    let plan =
        LaunchPlan::for_swift_bonsai25(&data, 8081, &key, bonsai_cpu_policy::Bonsai25Variant::Pq2)
            .map_err(|_| LaunchError::Policy)?;
    for (placement, expected) in [
        (crate::inference::Placement::Vulkan, "vulkan"),
        (crate::inference::Placement::Composite, "composite"),
    ] {
        base.placement = placement;
        let child = base.native_arguments(&plan, Some(&key));
        assert!(
            child
                .array_windows::<2>()
                .any(|[flag, value]| flag == "--placement" && value == expected),
            "missing [--placement, {expected}] in {child:?}"
        );
    }
    Ok(())
}

#[test]
fn daisyrose_stacks_with_nariq_no_orca_suffix_speculation_and_placements() -> Result<(), LaunchError>
{
    let data = std::env::temp_dir().join("Bonsai-daisyrose-stacking");
    let daisy_path = data.join("adapters").join("custom-daisy.gguf");
    let daisy_text = daisy_path.to_str().ok_or(LaunchError::Path)?;
    let key = data.join("backend.key");

    // NariQ LaunchPlan budgets DaisyRose reserve bytes cleanly.
    let mut nariq_plan =
        LaunchPlan::for_swift_nariq(&data, 8081, &key, bonsai_cpu_policy::NariqVariant::Bpw2_5)
            .map_err(|_| LaunchError::Policy)?;
    let base_ws = nariq_plan.working_set_limit_bytes;
    let daisy_plan = crate::launch_policy::DaisyRoseLoraPlan {
        path: daisy_path.clone(),
        bytes: 32 * 1024 * 1024,
        sha256: "a".repeat(64),
        max_rank: 16,
    };
    let reserve_usize = daisy_plan
        .reserve_bytes()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(LaunchError::Memory)?;
    nariq_plan.working_set_limit_bytes = base_ws + reserve_usize;
    nariq_plan
        .enable_daisyrose_lora(daisy_plan)
        .map_err(|_| LaunchError::Policy)?;
    assert_eq!(nariq_plan.native_child_memory_limit_bytes(), base_ws);

    // Suffix speculation + Vulkan/Composite placements parse alongside --daisyrose-lora.
    for placement in ["vulkan", "composite"] {
        let parsed = parse_native_launch_config(
            &data,
            &[
                "--cpu-prefill-tokens",
                "16",
                "--spec-type",
                "suffix",
                "--draft-max",
                "6",
                "--placement",
                placement,
                "--daisyrose-lora",
                daisy_text,
            ],
        )?;
        assert_eq!(
            parsed.spec_type,
            Some(crate::inference::SpeculationType::Suffix)
        );
        assert_eq!(parsed.draft_max, Some(6));
        assert_eq!(parsed.daisyrose_lora.as_ref(), Some(&daisy_path));
    }
    Ok(())
}

#[test]
fn bonsai25_supports_daisyrose_with_exact_cow_bytes() -> Result<(), LaunchError> {
    let data = std::env::temp_dir().join("Bonsai-bonsai25-launch-plan");
    let daisy_path = data.join("adapters").join("custom-daisy.gguf");
    let key = data.join("backend.key");
    let daisy_plan = crate::launch_policy::DaisyRoseLoraPlan {
        path: daisy_path,
        bytes: 32 * 1024 * 1024,
        sha256: "a".repeat(64),
        max_rank: 16,
    };
    let reserve_usize = daisy_plan
        .reserve_bytes()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(LaunchError::Memory)?;

    for variant in [
        bonsai_cpu_policy::Bonsai25Variant::Pq2,
        bonsai_cpu_policy::Bonsai25Variant::Ptq1,
    ] {
        let mut plan = LaunchPlan::for_swift_bonsai25(&data, 8081, &key, variant)
            .map_err(|_| LaunchError::Policy)?;
        let base_ws = plan.working_set_limit_bytes;
        let expected_fixed = usize::try_from(
            bonsai_cpu_policy::context::SwiftWeightPacking::from_bonsai25(variant)
                .calibrated_fixed_runtime_bytes(),
        )
        .map_err(|_| LaunchError::Memory)?;
        assert_eq!(base_ws, expected_fixed);
        plan.working_set_limit_bytes = base_ws + reserve_usize;
        plan.enable_daisyrose_lora(daisy_plan.clone())
            .map_err(|_| LaunchError::Policy)?;
        assert_eq!(plan.native_child_memory_limit_bytes(), base_ws);
    }
    Ok(())
}
