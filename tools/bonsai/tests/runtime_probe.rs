//! Native model-free probe regressions under the production headless policy.
#![cfg(all(windows, feature = "test-probe"))]
#![forbid(unsafe_code)]

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use bonsai_tools::startup_cancel;
use bonsai_windows_process::ProcessSpec;

// Exercise the private implementation without exposing a production test API.
#[path = "../src/runtime_probe.rs"]
mod runtime_probe;

type Result<T> = std::result::Result<T, String>;

fn make_spec(
    working_directory: &std::path::Path,
    arguments: Vec<std::ffi::OsString>,
    stem: &str,
) -> ProcessSpec {
    ProcessSpec {
        executable: env!("CARGO_BIN_EXE_bonsai-completion-test-probe").into(),
        arguments,
        working_directory: working_directory.into(),
        stdout: working_directory.join(format!("{stem}.stdout")),
        stderr: working_directory.join(format!("{stem}.stderr")),
        memory_limit_bytes: 512 * 1024 * 1024,
        active_process_limit: 1,
    }
}

#[test]
fn probes_keep_streams_separate_and_reject_failure_limits_and_cancellation() -> Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let scratch = std::env::temp_dir().join(format!("Bonsai-Probe-{}-{nonce}", std::process::id()));
    fs::create_dir(&scratch).map_err(|error| error.to_string())?;
    let spec = make_spec(&scratch, vec!["streams".into()], "streams");
    let limit_spec = make_spec(&scratch, vec!["streams".into()], "limit");
    let cancel_spec = make_spec(&scratch, vec!["streams".into()], "cancel");
    let failed_spec = make_spec(&scratch, vec!["partial-failure".into()], "failed");
    std::thread::scope(|scope| -> Result<()> {
        let h_ok = scope.spawn(|| -> Result<()> {
            let output = runtime_probe::run(&spec, 1024, "Fixture", || Ok(()))?;
            assert_eq!(output.stdout, "output");
            assert_eq!(output.stderr, "diagnostic");
            Ok(())
        });
        let h_limit = scope.spawn(|| -> Result<()> {
            assert_eq!(
                runtime_probe::run(&limit_spec, 1, "Fixture", || Ok(()))
                    .err()
                    .as_deref(),
                Some("Fixture probe exceeded its output bound.")
            );
            Ok(())
        });
        let h_cancel = scope.spawn(|| -> Result<()> {
            assert_eq!(
                runtime_probe::run(&cancel_spec, 1024, "Fixture", || Err("cancelled".into()))
                    .err()
                    .as_deref(),
                Some("cancelled")
            );
            Ok(())
        });
        let h_failed = scope.spawn(|| -> Result<()> {
            assert_eq!(
                runtime_probe::run(&failed_spec, 1024, "Fixture", || Ok(()))
                    .err()
                    .as_deref(),
                Some("Fixture probe failed with exit code 7 (0x00000007).")
            );
            Ok(())
        });
        for handle in [h_ok, h_limit, h_cancel, h_failed] {
            handle
                .join()
                .map_err(|_| "Runtime probe thread panicked.")??;
        }
        Ok(())
    })?;
    // All process and redirected-output handles must be released on errors.
    fs::remove_dir_all(scratch).map_err(|error| error.to_string())
}

#[test]
fn inspection_keeps_fixed_commands_output_limits_and_cancellation() -> Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let scratch =
        std::env::temp_dir().join(format!("Bonsai-Inspection-{}-{nonce}", std::process::id()));
    fs::create_dir(&scratch).map_err(|error| error.to_string())?;
    let limit_spec = make_spec(&scratch, vec!["--version".into()], "limit");
    let cancel_spec = make_spec(&scratch, vec!["--version".into()], "cancel");
    let invalid_spec = make_spec(&scratch, vec!["streams".into()], "invalid");
    std::thread::scope(|scope| -> Result<()> {
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, Result<()>>> = Vec::new();
        for kind in ["version", "help"] {
            let spec = make_spec(&scratch, vec![format!("--{kind}").into()], kind);
            handles.push(scope.spawn(move || {
                let output = runtime_probe::inspect(&spec, 1024, "Fixture", || Ok(()))?;
                assert_eq!(output.stdout, "output");
                assert_eq!(output.stderr, "diagnostic");
                Ok(())
            }));
        }
        handles.push(scope.spawn(|| {
            assert_eq!(
                runtime_probe::inspect(&limit_spec, 1, "Fixture", || Ok(()))
                    .err()
                    .as_deref(),
                Some("Fixture probe exceeded its output bound.")
            );
            Ok(())
        }));
        handles.push(scope.spawn(|| {
            assert_eq!(
                runtime_probe::inspect(&cancel_spec, 1024, "Fixture", || Err("cancelled".into()))
                    .err()
                    .as_deref(),
                Some("cancelled")
            );
            Ok(())
        }));
        handles.push(scope.spawn(|| {
            assert!(runtime_probe::inspect(&invalid_spec, 1024, "Fixture", || Ok(())).is_err());
            assert!(!invalid_spec.stdout.exists());
            assert!(!invalid_spec.stderr.exists());
            Ok(())
        }));
        for handle in handles {
            handle
                .join()
                .map_err(|_| "Inspection probe thread panicked.")??;
        }
        Ok(())
    })?;
    fs::remove_dir_all(scratch).map_err(|error| error.to_string())
}

#[test]
fn graphics_probe_keeps_allowlisted_commands_output_limits_and_cancellation() -> Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let scratch = std::env::temp_dir().join(format!(
        "Bonsai-GraphicsProbe-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&scratch).map_err(|error| error.to_string())?;
    let request_path = scratch.join("request.json");
    let gpu_args = vec![
        "host-gpu-query".into(),
        "--request".into(),
        request_path.as_os_str().to_os_string(),
    ];
    let limit_spec = make_spec(&scratch, gpu_args.clone(), "limit");
    let cancel_spec = make_spec(&scratch, gpu_args, "cancel");
    let invalid_spec = make_spec(&scratch, vec!["streams".into()], "invalid");
    std::thread::scope(|scope| -> Result<()> {
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, Result<()>>> = Vec::new();
        for command in [
            "host-gpu-query",
            "host-opencl-query",
            "opencl-projection-worker",
        ] {
            let spec = make_spec(
                &scratch,
                vec![
                    command.into(),
                    "--request".into(),
                    request_path.as_os_str().to_os_string(),
                ],
                command,
            );
            handles.push(scope.spawn(move || {
                let output = runtime_probe::graphics_probe(&spec, 1024, "Fixture", || Ok(()))?;
                assert_eq!(output.stdout, "output");
                assert_eq!(output.stderr, "diagnostic");
                Ok(())
            }));
        }
        handles.push(scope.spawn(|| {
            assert_eq!(
                runtime_probe::graphics_probe(&limit_spec, 1, "Fixture", || Ok(()))
                    .err()
                    .as_deref(),
                Some("Fixture probe exceeded its output bound.")
            );
            Ok(())
        }));
        handles.push(scope.spawn(|| {
            assert_eq!(
                runtime_probe::graphics_probe(&cancel_spec, 1024, "Fixture", || {
                    Err("cancelled".into())
                })
                .err()
                .as_deref(),
                Some("cancelled")
            );
            Ok(())
        }));
        handles.push(scope.spawn(|| {
            assert!(
                runtime_probe::graphics_probe(&invalid_spec, 1024, "Fixture", || Ok(())).is_err()
            );
            assert!(!invalid_spec.stdout.exists());
            assert!(!invalid_spec.stderr.exists());
            Ok(())
        }));
        for handle in handles {
            handle
                .join()
                .map_err(|_| "Graphics probe thread panicked.")??;
        }
        Ok(())
    })?;
    fs::remove_dir_all(scratch).map_err(|error| error.to_string())
}
