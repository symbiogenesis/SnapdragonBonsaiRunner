//! Production-image regression for the offline parser's native launch boundary.
#![cfg(windows)]
#![forbid(unsafe_code)]

use bonsai_windows_credentials::PrivateDirectory;
use bonsai_windows_process::{OwnedProcess, ProcessSpec, SandboxPolicy, ToolProcessSpec};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Read as _, Write as _},
    path::PathBuf,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
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

fn response(output: &mut File, ticket: u64, expected: &[u8]) -> io::Result<()> {
    let mut header = [0; 20];
    output.read_exact(&mut header)?;
    assert_eq!(&header[..8], b"BSMR0001");
    assert_eq!(&header[8..16], &ticket.to_le_bytes());
    assert_eq!(
        &header[16..],
        &u32::try_from(expected.len())
            .map_err(io::Error::other)?
            .to_le_bytes()
    );
    let mut bytes = vec![0; expected.len()];
    output.read_exact(&mut bytes)?;
    assert_eq!(bytes, expected);
    Ok(())
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
        input.write_all(b"BSMD0001")?;
        input.write_all(&ticket.to_le_bytes())?;
        input.write_all(&7_u32.to_le_bytes())?;
        input.write_all(source.as_bytes())?;
        input.flush()?;
        response(&mut output, ticket, &expected)?;
    }
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
        let reader = thread::Builder::new()
            .name("markdown-production-protocol".into())
            .spawn(move || {
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
        let deadline = Instant::now() + Duration::from_secs(5);
        let stopped = process.stop(Duration::from_secs(5));
        while stopped.is_ok() && !reader.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(25));
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
