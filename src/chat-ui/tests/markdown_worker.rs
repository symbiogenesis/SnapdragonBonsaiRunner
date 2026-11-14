//! Production-image regression for the offline parser's native launch boundary.
#![cfg(windows)]
#![forbid(unsafe_code)]

use bonsai_windows_credentials::PrivateDirectory;
use bonsai_windows_process::{OwnedProcess, ProcessSpec, SandboxPolicy};
use std::{
    fs::{self, File},
    io::{self, Read as _, Write as _},
    path::PathBuf,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

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
    fs::create_dir(&worker)?;
    let executable = PathBuf::from(env!("CARGO_BIN_EXE_bonsai-chat"));
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
        let (process, input, output) = OwnedProcess::start_sandboxed_piped_minimal(&spec, &policy)?;
        let (sender, receiver) = mpsc::sync_channel(1);
        let reader = thread::spawn(move || {
            let _sent = sender.send(exchange(input, output));
        });
        let result = receiver
            .recv_timeout(Duration::from_secs(14))
            .map_err(io::Error::other)
            .and_then(|result| result);
        // Stop before joining even when startup, framing or parsing failed.
        let deadline = Instant::now() + Duration::from_secs(5);
        process.stop(Duration::from_secs(5))?;
        while !reader.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(25));
        }
        if !reader.is_finished() {
            return Err(io::Error::other("Markdown pipe cleanup was not confirmed"));
        }
        reader
            .join()
            .map_err(|_| io::Error::other("Markdown protocol assertion failed"))?;
        result
    })();
    let worker_cleanup = fs::remove_dir(worker);
    drop(scratch);
    let root_cleanup = fs::remove_dir(root);
    result.and(worker_cleanup).and(root_cleanup)
}
