//! Production-image regression for the offline parser's native launch boundary.

use bonsai_windows_credentials::PrivateDirectory;
use bonsai_windows_process::{OwnedProcess, ProcessSpec, SandboxPolicy, ToolProcessSpec};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Read as _, Write as _},
    path::PathBuf,
    sync::{Arc, Condvar, Mutex, mpsc},
    thread,
    time::Duration,
};

fn complete(results: impl IntoIterator<Item = (&'static str, io::Result<()>)>) -> io::Result<()> {
    let errors: Vec<_> = results
        .into_iter()
        .filter_map(|(stage, result)| result.err().map(|error| format!("{stage}: {error}")))
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(io::Error::other(errors.join("; ")))
    }
}

#[test]
fn markdown_worker_refuses_direct_invocation_without_stdout() -> io::Result<()> {
    let scratch = PrivateDirectory::create(&std::env::temp_dir())?;
    let root = scratch.path().to_owned();
    let stdout = root.join("stdout");
    let stderr = root.join("stderr");
    let result = (|| {
        let process = OwnedProcess::start_tool(&ToolProcessSpec {
            executable: PathBuf::from(env!("CARGO_BIN_EXE_bonsai-broker")),
            arguments: vec!["--markdown-worker".into()],
            working_directory: root.clone(),
            stdout: stdout.clone(),
            stderr: stderr.clone(),
            environment: BTreeMap::new(),
        })?;
        let exited = process.wait_for_exit(Duration::from_secs(5));
        let stopped = process.stop(Duration::from_secs(5));
        let assertions = (|| {
            let exited = exited?;
            if exited != Some(1) {
                return Err(io::Error::other(format!(
                    "Direct Markdown worker returned {exited:?}, expected refusal code 1"
                )));
            }
            if !fs::read(&stdout)?.is_empty() || !fs::read(&stderr)?.is_empty() {
                return Err(io::Error::other("Direct Markdown worker emitted output"));
            }
            Ok(())
        })();
        complete([
            ("direct invocation", assertions),
            ("worker cleanup", stopped),
        ])
    })();
    let remove = |path| match fs::remove_file(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        result => result,
    };
    let stdout_cleanup = remove(stdout);
    let stderr_cleanup = remove(stderr);
    drop(scratch);
    complete([
        ("direct invocation", result),
        ("stdout cleanup", stdout_cleanup),
        ("stderr cleanup", stderr_cleanup),
        ("scratch cleanup", fs::remove_dir(root)),
    ])
}

fn read_response_payload(output: &mut File, ticket: u64) -> io::Result<Vec<u8>> {
    let mut header = [0; 20];
    output.read_exact(&mut header)?;
    assert_eq!(&header[..8], b"BSMR0001");
    assert_eq!(&header[8..16], &ticket.to_le_bytes());
    let length = usize::try_from(u32::from_le_bytes(
        header[16..].try_into().map_err(io::Error::other)?,
    ))
    .map_err(io::Error::other)?;
    let mut bytes = vec![0; length];
    output.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn response(output: &mut File, ticket: u64, expected: &[u8]) -> io::Result<()> {
    let bytes = read_response_payload(output, ticket)?;
    assert_eq!(bytes, expected);
    Ok(())
}

fn send_markdown(input: &mut File, ticket: u64, source: &str) -> io::Result<()> {
    input.write_all(b"BSMD0001")?;
    input.write_all(&ticket.to_le_bytes())?;
    input.write_all(
        &u32::try_from(source.len())
            .map_err(io::Error::other)?
            .to_le_bytes(),
    )?;
    input.write_all(source.as_bytes())?;
    input.flush()
}

fn send_raster(input: &mut File, ticket: u64, max_w: u32, max_h: u32, svg: &str) -> io::Result<()> {
    let body_len = 8_usize
        .checked_add(svg.len())
        .and_then(|len| u32::try_from(len).ok())
        .ok_or_else(|| io::Error::other("SVG raster frame overflow"))?;
    input.write_all(b"BSMG0001")?;
    input.write_all(&ticket.to_le_bytes())?;
    input.write_all(&body_len.to_le_bytes())?;
    input.write_all(&max_w.to_le_bytes())?;
    input.write_all(&max_h.to_le_bytes())?;
    input.write_all(svg.as_bytes())?;
    input.flush()
}

fn exchange(mut input: File, mut output: File) -> io::Result<()> {
    response(&mut output, 0, &[])?;
    // Independently assemble the render packet for one strong Unicode span.
    // This checks real parsing, source mapping and persistent worker reuse.
    let source = "**界**";
    let mut expected = b"BMD1".to_vec();
    expected.extend_from_slice(&3_u32.to_le_bytes());
    expected.extend_from_slice("界".as_bytes());
    for value in [1_u32, 0, 3] {
        expected.extend_from_slice(&value.to_le_bytes());
    }
    expected.extend_from_slice(&[4, 16]);
    for value in [1_u32, 2, 5, 0, 3] {
        expected.extend_from_slice(&value.to_le_bytes());
    }
    expected.push(1);
    for ticket in [1_u64, 2] {
        send_markdown(&mut input, ticket, source)?;
        response(&mut output, ticket, &expected)?;
    }

    // Verify offline segmented SVG extraction (`BMD2`) and `BMG1` CPU rasterization
    // inside the network-denied MXC AppContainer, plus closed rejection of scripted
    // and external-URL SVGs without tearing down the persistent worker.
    let safe_svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 32\"><rect width=\"64\" height=\"32\" fill=\"#2a9d8f\"/></svg>";
    send_markdown(
        &mut input,
        3,
        &format!("Diagram:\n\n```svg\n{safe_svg}\n```"),
    )?;
    let segmented = read_response_payload(&mut output, 3)?;
    assert!(segmented.starts_with(b"BMD2"));

    send_raster(&mut input, 4, 64, 32, safe_svg)?;
    let raster = read_response_payload(&mut output, 4)?;
    assert!(raster.starts_with(b"BMG1"));
    assert_eq!(raster.get(4).copied(), Some(1));
    assert_eq!(&raster[5..9], &64_u32.to_le_bytes());
    assert_eq!(&raster[9..13], &32_u32.to_le_bytes());
    let pixels = &raster[13..];
    assert_eq!(pixels.len(), 64 * 32 * 4);
    assert!(
        pixels
            .chunks_exact(4)
            .all(|px| px[0] <= px[3] && px[1] <= px[3] && px[2] <= px[3])
    );
    assert!(pixels.chunks_exact(4).any(|px| px != [0, 0, 0, 0]));

    let hostile_svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 32\"><script>alert(1)</script><image href=\"https://example.invalid/p.png\" width=\"64\" height=\"32\"/></svg>";
    send_markdown(&mut input, 5, &format!("```svg\n{hostile_svg}\n```"))?;
    let fallback_code = read_response_payload(&mut output, 5)?;
    assert!(fallback_code.starts_with(b"BMD1"));

    send_raster(&mut input, 6, 64, 32, hostile_svg)?;
    let mut rejected_header = b"BMG1".to_vec();
    rejected_header.extend_from_slice(&[0; 9]);
    response(&mut output, 6, &rejected_header)?;

    // Confirm persistent worker remains healthy after rejecting untrusted SVGs.
    send_markdown(&mut input, 7, source)?;
    response(&mut output, 7, &expected)?;
    Ok(())
}

#[test]
#[ignore = "Requires owner-selected native Windows ARM64 MXC/PSEC qualification and its aggregate power/budget observations."]
fn markdown_production_image_formats_under_native_containment() -> io::Result<()> {
    let scratch = PrivateDirectory::create(&std::env::temp_dir())?;
    let root = scratch.path().to_owned();
    let worker = root.join("worker");
    let executable = PathBuf::from(env!("CARGO_BIN_EXE_bonsai-broker"));
    let spec = ProcessSpec {
        executable: executable.clone(),
        arguments: vec!["--markdown-worker".into()],
        working_directory: worker.clone(),
        stdout: root.join("stdout"),
        stderr: root.join("stderr"),
        memory_limit_bytes: 256 * 1024 * 1024,
        active_process_limit: 1,
    };
    let policy = SandboxPolicy {
        readonly_paths: vec![executable],
        writable_directory: worker.clone(),
    };
    let result = (|| {
        fs::create_dir(&worker)?;
        let (process, input, output) = OwnedProcess::start_sandboxed_piped_minimal(&spec, &policy)?;
        let (sender, receiver) = mpsc::sync_channel(1);
        let finished = Arc::new((Mutex::new(false), Condvar::new()));
        let finished_signal = Arc::clone(&finished);
        let reader = thread::Builder::new()
            .name("markdown-production-protocol".into())
            .spawn(move || {
                struct MarkDone(Arc<(Mutex<bool>, Condvar)>);
                impl Drop for MarkDone {
                    fn drop(&mut self) {
                        if let Ok(mut done) = self.0.0.lock() {
                            *done = true;
                            self.0.1.notify_all();
                        }
                    }
                }
                let _mark = MarkDone(finished_signal);
                let _sent = sender.send(exchange(input, output));
            });
        let reader = match reader {
            Ok(reader) => reader,
            Err(error) => {
                return complete([
                    ("pipe thread", Err(error)),
                    ("worker cleanup", process.stop(Duration::from_secs(5))),
                ]);
            }
        };
        let result = receiver
            .recv_timeout(Duration::from_secs(14))
            .map_err(io::Error::other)
            .and_then(|result| result);
        // Retain the actual child state before stop can replace its exit code.
        // GUI-import initialization failures previously appeared only as EOF.
        let state = match process.try_wait() {
            Ok(Some(code)) => format!("exited 0x{code:08X}"),
            Ok(None) => "still running".to_owned(),
            Err(error) => format!("exit status unavailable: {error}"),
        };
        let result = result
            .map_err(|error| io::Error::other(format!("{error}; worker before cleanup: {state}")));
        // Attempt stop and reader cleanup even when startup/framing/parsing failed.
        let stopped = process.stop(Duration::from_secs(5));
        if stopped.is_ok()
            && let Ok(done) = finished.0.lock()
        {
            let _ = finished
                .1
                .wait_timeout_while(done, Duration::from_secs(5), |done| !*done);
        }
        let joined = if reader.is_finished() {
            reader
                .join()
                .map_err(|_| io::Error::other("Markdown protocol assertion failed"))
        } else {
            Err(io::Error::other("Markdown pipe cleanup was not confirmed"))
        };
        complete([
            ("Markdown exchange", result),
            ("worker cleanup", stopped),
            ("pipe cleanup", joined),
        ])
    })();
    let worker_cleanup = fs::remove_dir(worker);
    drop(scratch);
    let root_cleanup = fs::remove_dir(root);
    complete([
        ("contained worker", result),
        ("worker directory cleanup", worker_cleanup),
        ("scratch cleanup", root_cleanup),
    ])
}
