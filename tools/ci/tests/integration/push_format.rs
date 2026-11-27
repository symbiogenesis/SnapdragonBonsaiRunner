//! Cargo-to-native formatter handoff, including local dependencies and modules.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bonsai_windows_process::{OwnedProcess, ToolProcessSpec};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn executable(name: &str) -> Result<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH").ok_or("Missing PATH.")?)
        .map(|directory| directory.join(name))
        .find(|path| path.is_absolute() && path.is_file())
        .ok_or_else(|| format!("Missing fixture executable {name}.").into())
}

fn invoke(
    root: &Path,
    program: &Path,
    arguments: Vec<OsString>,
    environment: BTreeMap<String, String>,
) -> Result<(u32, String)> {
    let serial = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let stdout = root.join(format!("{serial}.stdout"));
    let stderr = root.join(format!("{serial}.stderr"));
    let process = OwnedProcess::start_tool(&ToolProcessSpec {
        executable: program.into(),
        arguments,
        working_directory: root.into(),
        stdout: stdout.clone(),
        stderr: stderr.clone(),
        environment,
    })?;
    let start = Instant::now();
    let result = loop {
        if let Some(code) = process.try_wait()? {
            break Ok(code);
        }
        if start.elapsed() >= Duration::from_secs(15)
            || fs::metadata(&stdout)?
                .len()
                .saturating_add(fs::metadata(&stderr)?.len())
                > 1024 * 1024
        {
            break Err("Formatter fixture exceeded its work/output allowance.");
        }
        let step = if start.elapsed() < Duration::from_millis(25) {
            Duration::from_millis(1)
        } else if start.elapsed() < Duration::from_millis(250) {
            Duration::from_millis(5)
        } else {
            Duration::from_millis(10)
        };
        std::thread::sleep(step);
    };
    process.stop(Duration::from_secs(5))?;
    let code = result?;
    let output = format!(
        "{}{}",
        fs::read_to_string(&stdout)?,
        fs::read_to_string(&stderr)?
    );
    fs::remove_file(stdout)?;
    fs::remove_file(stderr)?;
    Ok((code, output))
}

fn write(root: &Path, relative: &str, text: &str) -> Result<()> {
    let target = root.join(relative);
    fs::create_dir_all(target.parent().ok_or("Fixture file has no parent.")?)?;
    fs::write(target, text)?;
    Ok(())
}

fn write_fixture_workspace(dir: &Path, pin: &str) -> Result<()> {
    fs::create_dir_all(dir)?;
    write(dir, "rust-toolchain.toml", pin)?;
    write(
        dir,
        "rustfmt.toml",
        "edition = \"2024\"\nstyle_edition = \"2024\"\n",
    )?;
    write(
        dir,
        "Cargo.toml",
        "[workspace]\nmembers = [\"app\", \"peer\"]\nexclude = [\"external\"]\nresolver = \"3\"\n",
    )?;
    for package in ["app", "peer", "external"] {
        let dependency = if package == "app" {
            "[dependencies]\nexternal = { path = \"../external\" }\n"
        } else {
            ""
        };
        write(
            dir,
            &format!("{package}/Cargo.toml"),
            &format!(
                "[package]\nname = \"{package}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n{dependency}"
            ),
        )?;
        write(
            dir,
            &format!("{package}/src/lib.rs"),
            &format!("pub fn {package}() {{}}\n"),
        )?;
    }
    write(dir, "app/src/lib.rs", "mod child;\n")?;
    write(dir, "app/src/child.rs", "pub fn child() {}\n")?;
    Ok(())
}

#[test]
fn push_format_preserves_cargo_path_dependencies_configs_and_child_rejection() -> Result<()> {
    let temporary = fs::canonicalize(std::env::temp_dir())?;
    // The process boundary deliberately accepts ordinary native paths only.
    let temporary = PathBuf::from(temporary.to_string_lossy().trim_start_matches("\\\\?\\"));
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = temporary.join(format!(
        "bonsai-push-format-test-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&root)?;
    let result: Result<()> = (|| {
        let pin = include_str!("../../../../rust-toolchain.toml");
        let parsed: toml::Value = toml::from_str(pin)?;
        let channel = parsed["toolchain"]["channel"]
            .as_str()
            .ok_or("Missing compiler pin.")?;
        let toolchain = format!("{channel}-aarch64-pc-windows-msvc");
        let clean_dir = root.join("clean");
        write_fixture_workspace(&clean_dir, pin)?;
        let (code, image) = invoke(
            &clean_dir,
            &executable("rustup.exe")?,
            vec![
                "which".into(),
                "--toolchain".into(),
                toolchain.clone().into(),
                "rustfmt".into(),
            ],
            BTreeMap::new(),
        )?;
        assert_eq!(code, 0, "{image}");
        let adapter = PathBuf::from(env!("CARGO_BIN_EXE_bonsai-ci"));
        let environment = BTreeMap::from([
            (
                "RUSTFMT".into(),
                adapter.to_str().ok_or("Non-UTF8 adapter path.")?.into(),
            ),
            ("BONSAI_PUSH_RUSTFMT_IMAGE".into(), image.trim().into()),
            ("BONSAI_PUSH_RUSTFMT_MILLISECONDS".into(), "14000".into()),
        ]);
        let cargo = executable("cargo.exe")?;
        let arguments = vec![
            format!("+{toolchain}").into(),
            "fmt".into(),
            "--all".into(),
            "--".into(),
            "--check".into(),
        ];

        std::thread::scope(|scope| -> Result<()> {
            let h_clean = scope.spawn(|| -> std::result::Result<(), String> {
                let (code, output) =
                    invoke(&clean_dir, &cargo, arguments.clone(), environment.clone())
                        .map_err(|e| e.to_string())?;
                assert_eq!(code, 0, "{output}");
                Ok(())
            });
            let h_peer = scope.spawn(|| -> std::result::Result<(), String> {
                // Configurations still follow the real formatter's per-root search.
                let dir = root.join("peer-config");
                write_fixture_workspace(&dir, pin).map_err(|e| e.to_string())?;
                write(&dir, "peer/rustfmt.toml", "hard_tabs = true\n")
                    .map_err(|e| e.to_string())?;
                write(
                    &dir,
                    "peer/src/lib.rs",
                    "pub fn peer() {\n\tlet _value = 1;\n}\n",
                )
                .map_err(|e| e.to_string())?;
                let (code, output) = invoke(&dir, &cargo, arguments.clone(), environment.clone())
                    .map_err(|e| e.to_string())?;
                assert_eq!(code, 0, "{output}");
                Ok(())
            });
            let h_child = scope.spawn(|| -> std::result::Result<(), String> {
                let dir = root.join("broken-child");
                write_fixture_workspace(&dir, pin).map_err(|e| e.to_string())?;
                let broken = "pub fn child( ){ }\n";
                write(&dir, "app/src/child.rs", broken).map_err(|e| e.to_string())?;
                let (code, output) = invoke(&dir, &cargo, arguments.clone(), environment.clone())
                    .map_err(|e| e.to_string())?;
                assert_ne!(code, 0, "{output}");
                assert!(output.contains("child.rs"), "{output}");
                assert_eq!(
                    fs::read_to_string(dir.join("app/src/child.rs")).map_err(|e| e.to_string())?,
                    broken
                );
                Ok(())
            });
            let h_external = scope.spawn(|| -> std::result::Result<(), String> {
                // --all must still inspect a path dependency excluded from membership.
                let dir = root.join("broken-external");
                write_fixture_workspace(&dir, pin).map_err(|e| e.to_string())?;
                let broken = "pub fn external( ){ }\n";
                write(&dir, "external/src/lib.rs", broken).map_err(|e| e.to_string())?;
                let (code, output) = invoke(&dir, &cargo, arguments.clone(), environment.clone())
                    .map_err(|e| e.to_string())?;
                assert_ne!(code, 0, "{output}");
                assert!(output.contains("external"), "{output}");
                assert_eq!(
                    fs::read_to_string(dir.join("external/src/lib.rs"))
                        .map_err(|e| e.to_string())?,
                    broken
                );
                Ok(())
            });
            let h_emit = scope.spawn(|| -> std::result::Result<(), String> {
                let dir = root.join("emit-files");
                write_fixture_workspace(&dir, pin).map_err(|e| e.to_string())?;
                let broken = "pub fn external( ){ }\n";
                write(&dir, "external/src/lib.rs", broken).map_err(|e| e.to_string())?;
                let (code, output) = invoke(
                    &dir,
                    &adapter,
                    vec![
                        dir.join("external/src/lib.rs").into_os_string(),
                        "--edition".into(),
                        "2024".into(),
                        "--emit=files".into(),
                    ],
                    environment.clone(),
                )
                .map_err(|e| e.to_string())?;
                assert_ne!(code, 0, "{output}");
                assert_eq!(
                    fs::read_to_string(dir.join("external/src/lib.rs"))
                        .map_err(|e| e.to_string())?,
                    broken
                );
                Ok(())
            });
            for handle in [h_clean, h_peer, h_child, h_external, h_emit] {
                handle
                    .join()
                    .map_err(|_| "Push format worker panicked.")??;
            }
            Ok(())
        })?;
        Ok(())
    })();
    if root.parent() != Some(temporary.as_path()) {
        return Err("Formatter fixture cleanup escaped temporary storage.".into());
    }
    fs::remove_dir_all(&root)?;
    result
}
