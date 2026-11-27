//! First-run Swift artifacts bound to the same immutable policy as the broker.

use super::{Request, external};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    fs,
    io::Write as _,
    path::{Path, PathBuf},
    sync::OnceLock,
};

const SWIFT: &str = include_str!("../../../../config/swift-assessment.lock.json");

static SWIFT_LOCK: OnceLock<crate::Result<Value>> = OnceLock::new();

fn cached_json(
    cache: &'static OnceLock<crate::Result<Value>>,
    raw: &str,
) -> crate::Result<&'static Value> {
    cache
        .get_or_init(|| serde_json::from_str(raw).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(Clone::clone)
}

pub(super) struct Model {
    pub(super) path: PathBuf,
    pub(super) bytes: u64,
    pub(super) sha256: String,
    pub(super) url: Option<String>,
}

fn text<'a>(value: &'a Value, key: &str) -> crate::Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("Missing compiled Swift setup identity: {key}."))
}

pub(super) fn model(request: &Request) -> crate::Result<Model> {
    if let Some(variant) = request.nariq_variant {
        let selected = bonsai_cpu_policy::nariq_model_artifact(&request.data_directory, variant)?;
        return Ok(Model {
            path: selected.path,
            bytes: selected.bytes,
            sha256: selected.sha256,
            url: Some(selected.url),
        });
    }
    if let Some(variant) = request.effective_bonsai25_variant() {
        let selected =
            bonsai_cpu_policy::bonsai25_model_artifact(&request.data_directory, variant)?;
        return Ok(Model {
            path: selected.path,
            bytes: selected.bytes,
            sha256: selected.sha256,
            url: Some(selected.url),
        });
    }
    let lock = cached_json(&SWIFT_LOCK, SWIFT)?;
    let pin = bonsai_cpu_policy::select_artifact(
        lock,
        bonsai_cpu_policy::Profile::SwiftPq2Assessment,
        false,
        None,
    )?;
    let source = &lock["model"];
    let filename = text(&pin, "Filename")?;
    Ok(Model {
        path: request
            .data_directory
            .join("models")
            .join(text(source, "revision")?)
            .join(filename),
        bytes: pin["Bytes"]
            .as_u64()
            .ok_or("Missing compiled Swift model length.")?,
        sha256: text(&pin, "Sha256")?.into(),
        url: Some(format!(
            "https://huggingface.co/{}/resolve/{}/{filename}",
            source["repository"].as_str().unwrap_or_default(),
            source["revision"].as_str().unwrap_or_default(),
        )),
    })
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct StatusReport<'a> {
    schema_version: u8,
    stage: &'a str,
    message: &'a str,
    kind: &'a str,
    activity: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_path: Option<&'a Path>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_bytes: Option<u64>,
}

pub(super) fn status(
    working: &Path,
    stage: &str,
    message: &str,
    kind: &str,
    activity: &str,
    artifact: Option<(&Path, u64)>,
) -> crate::Result<()> {
    external(working)?;
    let path = working.join("chat-setup-progress.json");
    if let Ok(metadata) = fs::symlink_metadata(&path)
        && (metadata.file_type().is_symlink()
            || bonsai_artifacts::is_redirect(&metadata)
            || metadata.is_dir())
    {
        return Err("Setup progress path must be a regular file.".into());
    }
    let temporary = working.join(format!(
        ".chat-setup-progress-{}.json",
        crate::startup_process::unique()
    ));
    let (artifact_path, expected_bytes) = match artifact {
        Some((artifact_path, expected_bytes)) => (Some(artifact_path), Some(expected_bytes)),
        None => (None, None),
    };
    let report = StatusReport {
        schema_version: 2,
        stage,
        message,
        kind,
        activity,
        artifact_path,
        expected_bytes,
    };
    let encoded = serde_json::to_vec(&report).map_err(|error| error.to_string())?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("Cannot write setup progress: {error}"))?;
    output
        .write_all(&encoded)
        .map_err(|error| format!("Cannot write setup progress: {error}"))?;
    drop(output);
    fs::rename(&temporary, &path).map_err(|error| format!("Cannot publish setup progress: {error}"))
}

fn preflight(data: &Path, model: &Model, model_exists: bool) -> crate::Result<()> {
    if model_exists || model.url.is_none() {
        return Ok(());
    }
    #[cfg(windows)]
    {
        if model.path.file_name().is_none() {
            return Err("Missing model filename.".into());
        }
        let partial = model.path.with_added_extension("partial");
        external(&partial)?;
        let downloaded = fs::symlink_metadata(&partial)
            .ok()
            .filter(|metadata| metadata.is_file() && !bonsai_artifacts::is_redirect(metadata))
            .map_or(0, |metadata| metadata.len().min(model.bytes));
        let needed = model
            .bytes
            .saturating_sub(downloaded)
            .saturating_add(256 * 1024 * 1024);
        let available = bonsai_windows_process::available_disk_bytes(data).map_err(|error| {
            format!(
                "Swift could not check free storage in {}: {error}",
                data.display()
            )
        })?;
        if available < needed {
            return Err(format!(
                "Swift needs at least {} MiB of free storage in {} for model setup; {} MiB is available. Free space, then select Retry setup.",
                needed.div_ceil(1024 * 1024),
                data.display(),
                available / (1024 * 1024),
            ));
        }
    }
    #[cfg(not(windows))]
    let _ = data;
    Ok(())
}

// Availability describes this selected model, not a persistent first-launch receipt.
pub(super) fn kind(request: &Request) -> crate::Result<&'static str> {
    let selected = model(request)?;
    Ok(if selected.path.exists() {
        "existing"
    } else {
        "first-run"
    })
}

fn publisher(request: &Request, working: &Path, kind: &str) -> crate::Result<()> {
    runtime_status(
        working,
        kind,
        crate::startup_download::RepairActivity::Publishing,
        None,
    )?;
    // Portable packages do not carry a checkout. Materialize only the compiled
    // lock for the existing strict provisioner, then remove this private scratch.
    let path = working.join(format!(
        "chat-swift-lock-{}.json",
        crate::startup_process::unique()
    ));
    external(&path)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| error.to_string())?;
    file.write_all(SWIFT.as_bytes())
        .map_err(|error| error.to_string())?;
    drop(file);
    let result = crate::provision::prepare_chat_runtime_observed(
        json!({
            "operation":"engine", "dataDirectory":request.data_directory,
            "workingDirectory":working, "lockPath":path,
        }),
        &mut |phase, artifact| runtime_status(working, kind, phase, artifact),
    );
    let cleanup = fs::remove_file(&path);
    result.map_err(|error| format!("The publisher runtime could not be prepared in {}: {error}. Select Retry setup after fixing the reported problem.", request.data_directory.display()))?;
    cleanup.map_err(|error| error.to_string())?;
    Ok(())
}

fn prepare_cached_or_receive(
    url: &str,
    path: &Path,
    bytes: u64,
    hash: &str,
    working: &Path,
    mut receive: impl FnMut(&str, &Path, u64, &str, &Path) -> crate::Result<()>,
) -> crate::Result<()> {
    // Cached files with exact locked byte length are cryptographically hashed
    // and bound via `KIND_VERIFIED_SECTION` immediately by `bonsai-broker`;
    // skipping Pass-1 single-threaded SHA-256 removes ~7.9 GiB of cold-start I/O,
    // while missing or size-mismatched files fall through to verified repair.
    if crate::startup_download::verify_length(path, bytes).is_ok() {
        return Ok(());
    }
    receive(url, path, bytes, hash, working)
}

pub(super) fn prepare(request: &Request, working: &Path) -> crate::Result<()> {
    crate::startup_cancel::checkpoint()?;
    let selected = model(request)?;
    let model_exists = selected.path.exists();
    let kind = if model_exists {
        "existing"
    } else {
        "first-run"
    };
    if !request.native_engine {
        publisher(request, working, kind)?;
    }
    let mmproj = (!request.no_mmproj && request.nariq_variant.is_none())
        .then(|| bonsai_cpu_policy::swift_mmproj_artifact(&request.data_directory))
        .transpose()?;
    let daisyrose =
        (request.native_engine && !request.no_daisyrose_lora && request.daisyrose_lora.is_none())
            .then(|| {
                bonsai_cpu_policy::DaisyRoseLoraArtifact::pinned_for_base_sha256(
                    &request.data_directory,
                    &selected.sha256,
                )
            })
            .transpose()?
            .flatten()
            .filter(|artifact| artifact.default_enabled && artifact.has_remote_source());
    prepare_weights_with(
        request,
        working,
        &selected,
        model_exists,
        kind,
        mmproj.as_ref(),
        daisyrose.as_ref(),
        |url, path, bytes, hash, working| {
            let stage = if path == selected.path {
                "model"
            } else if mmproj
                .as_ref()
                .is_some_and(|artifact| artifact.path == path)
            {
                "mmproj"
            } else if daisyrose
                .as_ref()
                .is_some_and(|artifact| artifact.path == path)
            {
                "daisyrose-lora"
            } else {
                "adapter"
            };
            prepare_cached_or_receive(url, path, bytes, hash, working, |u, p, b, h, w| {
                crate::startup_download::receive_repair_observed(u, p, b, h, w, &mut |activity| {
                    artifact_status(working, stage, kind, p, b, activity)
                })
            })
        },
        |path, bytes, _hash| crate::startup_download::verify_length(path, bytes),
    )
}

#[cfg(test)]
fn prepare_weights(
    request: &Request,
    working: &Path,
    receive: impl FnMut(&str, &Path, u64, &str, &Path) -> crate::Result<()>,
    verify: impl FnMut(&Path, u64, &str) -> crate::Result<()>,
) -> crate::Result<()> {
    let selected = model(request)?;
    let model_exists = selected.path.exists();
    let kind = if model_exists {
        "existing"
    } else {
        "first-run"
    };
    let mmproj = (!request.no_mmproj && request.nariq_variant.is_none())
        .then(|| bonsai_cpu_policy::swift_mmproj_artifact(&request.data_directory))
        .transpose()?;
    let daisyrose =
        (request.native_engine && !request.no_daisyrose_lora && request.daisyrose_lora.is_none())
            .then(|| {
                bonsai_cpu_policy::DaisyRoseLoraArtifact::pinned_for_base_sha256(
                    &request.data_directory,
                    &selected.sha256,
                )
            })
            .transpose()?
            .flatten()
            .filter(|artifact| artifact.default_enabled && artifact.has_remote_source());
    prepare_weights_with(
        request,
        working,
        &selected,
        model_exists,
        kind,
        mmproj.as_ref(),
        daisyrose.as_ref(),
        receive,
        verify,
    )
}

fn emit_checking_status_unless_cached(
    working: &Path,
    stage: &str,
    message: &str,
    kind: &str,
    path: &Path,
    bytes: u64,
) -> crate::Result<()> {
    if kind == "existing" && crate::startup_download::verify_length(path, bytes).is_ok() {
        return Ok(());
    }
    status(working, stage, message, kind, "checking", Some((path, bytes)))
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares pre-resolved model, existence, kind, mmproj, and daisyrose state from prepare without reparsing."
)]
fn prepare_weights_with(
    request: &Request,
    working: &Path,
    selected: &Model,
    model_exists: bool,
    kind: &str,
    mmproj: Option<&bonsai_cpu_policy::SwiftMmprojArtifact>,
    daisyrose: Option<&bonsai_cpu_policy::DaisyRoseLoraArtifact>,
    mut receive: impl FnMut(&str, &Path, u64, &str, &Path) -> crate::Result<()>,
    mut verify: impl FnMut(&Path, u64, &str) -> crate::Result<()>,
) -> crate::Result<()> {
    preflight(&request.data_directory, selected, model_exists)?;
    if let Some(url) = selected.url.as_deref() {
        emit_checking_status_unless_cached(
            working,
            "model",
            "Checking Swift model files…",
            kind,
            &selected.path,
            selected.bytes,
        )?;
        receive(url, &selected.path, selected.bytes, &selected.sha256, working)
            .map_err(|error| format!("Swift model setup failed at {}: {error}. Check the internet connection, free storage and folder permissions, then select Retry setup. Partial downloads are kept for resume.", selected.path.display()))?;
    } else {
        status(
            working,
            "model",
            "Checking Swift model files…",
            kind,
            "checking",
            Some((&selected.path, selected.bytes)),
        )?;
        verify(&selected.path, selected.bytes, &selected.sha256).map_err(|error| {
            format!(
                "The selected model export at {} could not be verified: {error}.",
                selected.path.display()
            )
        })?;
    }
    if let Some(mmproj) = mmproj {
        emit_checking_status_unless_cached(
            working,
            "mmproj",
            "Checking vision projector files…",
            kind,
            &mmproj.path,
            mmproj.bytes,
        )?;
        receive(&mmproj.url, &mmproj.path, mmproj.bytes, &mmproj.sha256, working)
            .map_err(|error| format!("Vision projector setup failed at {}: {error}. Check the internet connection, free storage and folder permissions, then select Retry setup.", mmproj.path.display()))?;
    }
    if let Some(daisyrose) = daisyrose.filter(|artifact| {
        artifact.default_enabled
            && artifact.has_remote_source()
            && artifact.base_model_sha256 == selected.sha256
            && request.native_engine
            && !request.no_daisyrose_lora
            && request.daisyrose_lora.is_none()
    }) {
        emit_checking_status_unless_cached(
            working,
            "daisyrose-lora",
            "Checking DaisyRose adapter files…",
            kind,
            &daisyrose.path,
            daisyrose.bytes,
        )?;
        receive(
            &daisyrose.url,
            &daisyrose.path,
            daisyrose.bytes,
            &daisyrose.sha256,
            working,
        )
        .map_err(|error| {
            format!(
                "DaisyRose adapter setup failed at {}: {error}. Check the internet connection, free storage and folder permissions, then select Retry setup.",
                daisyrose.path.display()
            )
        })?;
    }
    status(
        working,
        "ready",
        "Starting your local model\u{2026}",
        kind,
        "ready",
        None,
    )
}

pub(super) fn runtime_status(
    working: &Path,
    kind: &str,
    phase: crate::startup_download::RepairActivity,
    artifact: Option<(&Path, u64)>,
) -> crate::Result<()> {
    use crate::startup_download::RepairActivity;
    let (activity, message) = match phase {
        RepairActivity::VerifyingExisting => ("verifying", "Checking local runtime files\u{2026}"),
        RepairActivity::Downloading { offset: 0 } => {
            ("downloading", "Downloading the runtime\u{2026}")
        }
        RepairActivity::Downloading { .. } => ("downloading", "Resuming runtime download\u{2026}"),
        RepairActivity::VerifyingDownloaded => {
            ("verifying", "Verifying the downloaded runtime\u{2026}")
        }
        RepairActivity::Publishing => ("preparing", "Preparing the runtime\u{2026}"),
    };
    status(working, "runtime", message, kind, activity, artifact)
}

fn artifact_status(
    working: &Path,
    stage: &str,
    kind: &str,
    path: &Path,
    bytes: u64,
    phase: crate::startup_download::RepairActivity,
) -> crate::Result<()> {
    use crate::startup_download::RepairActivity;
    let name = match stage {
        "model" => "Swift model",
        "mmproj" => "vision projector",
        "daisyrose-lora" => "DaisyRose adapter",
        _ => "model adapter",
    };
    let (activity, message) = match phase {
        RepairActivity::VerifyingExisting if path.exists() => {
            ("verifying", format!("Verifying local {name}…"))
        }
        RepairActivity::VerifyingExisting => ("checking", format!("Checking {name} files…")),
        RepairActivity::Downloading { offset: 0 } => {
            ("downloading", format!("Downloading {name}…"))
        }
        RepairActivity::Downloading { .. } => ("downloading", format!("Resuming {name} download…")),
        RepairActivity::VerifyingDownloaded => {
            ("verifying", format!("Verifying downloaded {name}…"))
        }
        RepairActivity::Publishing => ("preparing", format!("Preparing verified {name}…")),
    };
    status(
        working,
        stage,
        &message,
        kind,
        activity,
        Some((path, bytes)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_model_identity_matches_swift_policy_and_never_downloads_ptq1() -> crate::Result<()> {
        let data = std::env::temp_dir().join("Bonsai-setup-model-pin-only");
        let mut request: Request = serde_json::from_value(json!({
            "DataDirectory":data,"RepositoryRoot":data.join("source"),
        }))
        .map_err(|error| error.to_string())?;
        let default_b25 = model(&request)?;
        assert_eq!(
            default_b25.bytes,
            bonsai_cpu_policy::Bonsai25Variant::Pq2.weight_bytes()
        );
        assert_eq!(
            default_b25.sha256,
            bonsai_cpu_policy::Bonsai25Variant::Pq2.sha256()
        );
        assert!(
            default_b25
                .path
                .ends_with("Ternary-Bonsai-2.5-27B-PQ2_0.gguf")
        );

        request.native_engine = false;
        let pq2 = model(&request)?;
        assert_eq!(pq2.bytes, 7_206_168_928);
        assert_eq!(
            pq2.sha256,
            "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5"
        );
        assert!(pq2.path.ends_with(
            "models/a3bdac086bbb6b04d87d0108d18943b5f63868f7/Swift-Bonsai-2-PQ2_0.gguf"
        ));
        assert_eq!(
            pq2.url.as_deref(),
            Some(
                "https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF/resolve/a3bdac086bbb6b04d87d0108d18943b5f63868f7/Swift-Bonsai-2-PQ2_0.gguf"
            )
        );
        Ok(())
    }

    #[test]
    fn automatic_weight_setup_downloads_only_selected_pins_and_stops_at_first_failure()
    -> crate::Result<()> {
        let working = std::env::temp_dir().join(format!(
            "Bonsai-SetupDownloads-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&working).map_err(|error| error.to_string())?;
        let data = working.join("data");
        fs::create_dir(&data).map_err(|error| error.to_string())?;
        let mut request: Request = serde_json::from_value(json!({
            "DataDirectory":data,"RepositoryRoot":working.join("source"),"NativeEngine":true,
        }))
        .map_err(|error| error.to_string())?;
        // A tiny placeholder keeps this pure callback control independent of
        // the owner's free storage; no model bytes or network are consumed.
        let selected = model(&request)?;
        fs::create_dir_all(selected.path.parent().ok_or("Missing parent.")?)
            .map_err(|error| error.to_string())?;
        fs::write(&selected.path, b"intentionally invalid model-free fixture")
            .map_err(|error| error.to_string())?;
        for no_mmproj in [false, true] {
            request.no_mmproj = no_mmproj;
            let mut received = Vec::new();
            prepare_weights(
                &request,
                &working,
                |url, path, bytes, hash, _| {
                    received.push((url.to_owned(), path.to_owned(), bytes, hash.to_owned()));
                    Ok(())
                },
                |_, _, _| Err("Native setup must use verified download preparation.".into()),
            )?;
            let expected_count = 1 + usize::from(!no_mmproj);
            assert_eq!(received.len(), expected_count);
            assert_eq!(received[0].1, selected.path);
            assert_eq!(received[0].2, selected.bytes);
            assert_eq!(received[0].3, selected.sha256);
            if !no_mmproj {
                let expected = bonsai_cpu_policy::swift_mmproj_artifact(&data)?;
                assert_eq!(received[1].1, expected.path);
                assert_eq!(received[1].2, expected.bytes);
                assert_eq!(received[1].3, expected.sha256);
                assert_eq!(received[1].0, expected.url);
            }
        }
        request.no_mmproj = false;
        let mut count = 0;
        let error = prepare_weights(
            &request,
            &working,
            |_, _, _, _, _| {
                count += 1;
                Err("Synthetic interrupted download".into())
            },
            |_, _, _| Ok(()),
        )
        .err()
        .ok_or("Download failure was ignored.")?;
        assert!(error.contains("Synthetic interrupted download"));
        assert_eq!(count, 1);
        fs::remove_dir_all(working).map_err(|error| error.to_string())
    }

    #[test]
    fn automatic_weight_setup_handles_promoted_daisyrose_opt_in_and_opt_out() -> crate::Result<()> {
        let working = std::env::temp_dir().join(format!(
            "Bonsai-SetupDaisyRose-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&working).map_err(|error| error.to_string())?;
        let data = working.join("data");
        fs::create_dir(&data).map_err(|error| error.to_string())?;
        let mut request: Request = serde_json::from_value(json!({
            "DataDirectory": data,
            "RepositoryRoot": working.join("source"),
            "NativeEngine": true,
            "NoMmproj": true,
        }))
        .map_err(|error| error.to_string())?;
        let selected = model(&request)?;
        fs::create_dir_all(selected.path.parent().ok_or("Missing parent.")?)
            .map_err(|error| error.to_string())?;
        fs::write(&selected.path, b"model-free placeholder").map_err(|error| error.to_string())?;

        // Unpinned default downloads zero DaisyRose files.
        let mut default_received = Vec::new();
        prepare_weights(
            &request,
            &working,
            |_url, path, _bytes, _hash, _| {
                default_received.push(path.to_owned());
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        assert_eq!(default_received, vec![selected.path.clone()]);

        let lock = json!({
            "schemaVersion": 1,
            "envelope": "npu-friendly",
            "dtype": "f16",
            "maxRank": 16,
            "targetCount": 128,
            "bytes": 67_108_864_u64,
            "sha256": "a".repeat(64),
            "baseModelSha256": bonsai_cpu_policy::Bonsai25Variant::Pq2.sha256(),
            "source": {
                "host": "huggingface",
                "repository": "symbiogenesis/DaisyRose-Swift-LoRA",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "path": "gguf/daisyrose.gguf",
                "filename": "daisyrose.gguf",
                "url": "https://huggingface.co/symbiogenesis/DaisyRose-Swift-LoRA/resolve/0123456789abcdef0123456789abcdef01234567/gguf/daisyrose.gguf",
                "license": "Apache-2.0"
            },
            "defaultEnabled": true,
            "defaultScaleTenths": 10
        });
        let promoted = bonsai_cpu_policy::DaisyRoseLoraArtifact::for_bonsai25_lock(
            &data,
            bonsai_cpu_policy::Bonsai25Variant::Pq2,
            &lock.to_string(),
        )?;

        // Promoted default downloads DaisyRose after model when enabled.
        let mut received = Vec::new();
        prepare_weights_with(
            &request,
            &working,
            &selected,
            true,
            "existing",
            None,
            Some(&promoted),
            |url, path, bytes, hash, _| {
                received.push((url.to_owned(), path.to_owned(), bytes, hash.to_owned()));
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        assert_eq!(received.len(), 2);
        assert_eq!(received[1].0, promoted.url);
        assert_eq!(received[1].1, promoted.path);
        assert_eq!(received[1].2, promoted.bytes);
        assert_eq!(received[1].3, promoted.sha256);

        // Local-only DaisyRose (no remote URL) and foreign-base locks skip download cleanly.
        let mut local_only = promoted.clone();
        local_only.url.clear();
        received.clear();
        prepare_weights_with(
            &request,
            &working,
            &selected,
            true,
            "existing",
            None,
            Some(&local_only),
            |url, path, bytes, hash, _| {
                received.push((url.to_owned(), path.to_owned(), bytes, hash.to_owned()));
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        assert_eq!(received.len(), 1);

        let mut foreign_base = promoted.clone();
        foreign_base.base_model_sha256 =
            "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5".into();
        received.clear();
        prepare_weights_with(
            &request,
            &working,
            &selected,
            true,
            "existing",
            None,
            Some(&foreign_base),
            |url, path, bytes, hash, _| {
                received.push((url.to_owned(), path.to_owned(), bytes, hash.to_owned()));
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        assert_eq!(received.len(), 1);

        // NoDaisyRoseLora skips automatic DaisyRose download.
        request.no_daisyrose_lora = true;
        received.clear();
        prepare_weights_with(
            &request,
            &working,
            &selected,
            true,
            "existing",
            None,
            Some(&promoted),
            |url, path, bytes, hash, _| {
                received.push((url.to_owned(), path.to_owned(), bytes, hash.to_owned()));
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        assert_eq!(received.len(), 1);

        // Explicit --daisyrose-lora override skips automatic pinned download too.
        request.no_daisyrose_lora = false;
        request.daisyrose_lora = Some(data.join("explicit-daisy.gguf"));
        received.clear();
        prepare_weights_with(
            &request,
            &working,
            &selected,
            true,
            "existing",
            None,
            Some(&promoted),
            |url, path, bytes, hash, _| {
                received.push((url.to_owned(), path.to_owned(), bytes, hash.to_owned()));
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        assert_eq!(received.len(), 1);

        // Download failure on DaisyRose stops setup cleanly.
        request.daisyrose_lora = None;
        let err = prepare_weights_with(
            &request,
            &working,
            &selected,
            true,
            "existing",
            None,
            Some(&promoted),
            |_, path, _, _, _| {
                if path == promoted.path {
                    Err("Synthetic DaisyRose failure".into())
                } else {
                    Ok(())
                }
            },
            |_, _, _| Ok(()),
        )
        .err()
        .ok_or("Expected DaisyRose setup failure.")?;
        assert!(err.contains("DaisyRose adapter setup failed"));

        fs::remove_dir_all(working).map_err(|error| error.to_string())
    }

    #[test]
    fn nariq_setup_downloads_model_skips_abliterate_and_mmproj_and_keeps_daisyrose()
    -> crate::Result<()> {
        let working = std::env::temp_dir().join(format!(
            "Bonsai-SetupNariQ-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&working).map_err(|error| error.to_string())?;
        let data = working.join("data");
        fs::create_dir(&data).map_err(|error| error.to_string())?;
        let request: Request = serde_json::from_value(json!({
            "DataDirectory": data,
            "RepositoryRoot": working.join("source"),
            "NativeEngine": true,
            "NariqVariant": "2.5bpw",
        }))
        .map_err(|error| error.to_string())?;
        let selected = model(&request)?;
        let pinned_nariq = bonsai_cpu_policy::nariq_model_artifact(
            &data,
            bonsai_cpu_policy::NariqVariant::Bpw2_5,
        )?;
        assert_eq!(selected.path, pinned_nariq.path);
        assert_eq!(selected.bytes, pinned_nariq.bytes);
        assert_eq!(selected.sha256, pinned_nariq.sha256);
        assert_eq!(selected.url.as_deref(), Some(pinned_nariq.url.as_str()));

        let daisy_lock = json!({
            "schemaVersion": 1,
            "envelope": "npu-friendly",
            "dtype": "f16",
            "maxRank": 16,
            "targetCount": 128,
            "bytes": 67_108_864_u64,
            "sha256": "a".repeat(64),
            "baseModelSha256": bonsai_cpu_policy::NariqVariant::Bpw2_5.sha256(),
            "source": {
                "host": "huggingface",
                "repository": "symbiogenesis/DaisyRose-Swift-LoRA",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "path": "gguf/daisyrose.gguf",
                "filename": "daisyrose.gguf",
                "url": "https://huggingface.co/symbiogenesis/DaisyRose-Swift-LoRA/resolve/0123456789abcdef0123456789abcdef01234567/gguf/daisyrose.gguf",
                "license": "Apache-2.0"
            },
            "defaultEnabled": true,
            "defaultScaleTenths": 10
        });
        let promoted = bonsai_cpu_policy::DaisyRoseLoraArtifact::for_nariq_lock(
            &data,
            bonsai_cpu_policy::NariqVariant::Bpw2_5,
            &daisy_lock.to_string(),
        )?;
        let mut received = Vec::new();
        prepare_weights_with(
            &request,
            &working,
            &selected,
            false,
            "first-run",
            None,
            Some(&promoted),
            |url, path, bytes, hash, _| {
                received.push((url.to_owned(), path.to_owned(), bytes, hash.to_owned()));
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        assert_eq!(received.len(), 2);
        assert_eq!(received[0].0, pinned_nariq.url);
        assert_eq!(received[0].1, pinned_nariq.path);
        assert_eq!(received[0].2, pinned_nariq.bytes);
        assert_eq!(received[0].3, pinned_nariq.sha256);
        assert_eq!(received[1].0, promoted.url);
        assert_eq!(received[1].1, promoted.path);

        fs::remove_dir_all(working).map_err(|error| error.to_string())
    }

    #[test]
    fn bonsai25_setup_downloads_model_and_mmproj_and_skips_pq2_daisyrose() -> crate::Result<()> {
        let working = std::env::temp_dir().join(format!(
            "Bonsai-SetupBonsai25-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&working).map_err(|error| error.to_string())?;
        let data = working.join("data");
        fs::create_dir(&data).map_err(|error| error.to_string())?;
        let request: Request = serde_json::from_value(json!({
            "DataDirectory": data,
            "RepositoryRoot": working.join("source"),
            "NativeEngine": true,
            "Bonsai25Variant": "ptq1",
        }))
        .map_err(|error| error.to_string())?;
        let selected = model(&request)?;
        let pinned_b25 = bonsai_cpu_policy::bonsai25_model_artifact(
            &data,
            bonsai_cpu_policy::Bonsai25Variant::Ptq1,
        )?;
        assert_eq!(selected.path, pinned_b25.path);
        assert_eq!(selected.bytes, pinned_b25.bytes);
        assert_eq!(selected.sha256, pinned_b25.sha256);
        assert_eq!(selected.url.as_deref(), Some(pinned_b25.url.as_str()));

        let daisy_lock = json!({
            "schemaVersion": 1,
            "envelope": "npu-friendly",
            "dtype": "f16",
            "maxRank": 16,
            "targetCount": 128,
            "bytes": 67_108_864_u64,
            "sha256": "a".repeat(64),
            "baseModelSha256": "5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5",
            "source": {
                "host": "huggingface",
                "repository": "symbiogenesis/DaisyRose-Swift-LoRA",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "path": "gguf/daisyrose.gguf",
                "filename": "daisyrose.gguf",
                "url": "https://huggingface.co/symbiogenesis/DaisyRose-Swift-LoRA/resolve/0123456789abcdef0123456789abcdef01234567/gguf/daisyrose.gguf",
                "license": "Apache-2.0"
            },
            "defaultEnabled": true,
            "defaultScaleTenths": 10
        });
        let promoted =
            bonsai_cpu_policy::daisyrose_lora_artifact_from_lock(&data, &daisy_lock.to_string())?;
        let mmproj = bonsai_cpu_policy::swift_mmproj_artifact(&data)?;
        let mut received = Vec::new();
        prepare_weights_with(
            &request,
            &working,
            &selected,
            false,
            "first-run",
            Some(&mmproj),
            Some(&promoted),
            |url, path, bytes, hash, _| {
                received.push((url.to_owned(), path.to_owned(), bytes, hash.to_owned()));
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        // Bonsai 2.5 downloads model + mmproj (foreign DaisyRose skipped).
        assert_eq!(received.len(), 2);
        assert_eq!(received[0].1, pinned_b25.path);
        assert_eq!(received[1].1, mmproj.path);

        fs::remove_dir_all(working).map_err(|error| error.to_string())
    }

    #[test]
    fn prepare_cached_or_receive_skips_exact_length_and_falls_back_on_missing_or_mismatch()
    -> crate::Result<()> {
        let working = std::env::temp_dir().join(format!(
            "Bonsai-SetupCachedLength-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&working).map_err(|error| error.to_string())?;
        let artifact = working.join("cached.gguf");

        let mut repairs = 0_usize;
        prepare_cached_or_receive(
            "https://huggingface.co/example/resolve/main/cached.gguf",
            &artifact,
            5,
            &"a".repeat(64),
            &working,
            |_, _, _, _, _| {
                repairs += 1;
                Ok(())
            },
        )?;
        assert_eq!(
            repairs, 1,
            "missing file must enter verified receive/repair"
        );

        fs::write(&artifact, b"hello").map_err(|error| error.to_string())?;
        prepare_cached_or_receive(
            "https://huggingface.co/example/resolve/main/cached.gguf",
            &artifact,
            5,
            &"a".repeat(64),
            &working,
            |_, _, _, _, _| {
                repairs += 1;
                Ok(())
            },
        )?;
        assert_eq!(
            repairs, 1,
            "exact-length cached file must skip redundant Pass-1 SHA-256/repair"
        );

        fs::write(&artifact, b"hell").map_err(|error| error.to_string())?;
        prepare_cached_or_receive(
            "https://huggingface.co/example/resolve/main/cached.gguf",
            &artifact,
            5,
            &"a".repeat(64),
            &working,
            |_, _, _, _, _| {
                repairs += 1;
                Ok(())
            },
        )?;
        assert_eq!(
            repairs, 2,
            "1-byte length mismatch must enter verified receive/repair"
        );

        fs::remove_dir_all(working).map_err(|error| error.to_string())
    }

    #[test]
    fn status_writes_expected_json_and_skips_checking_on_cached_existing_lengths()
    -> crate::Result<()> {
        let working = std::env::temp_dir().join(format!(
            "Bonsai-SetupStatusSkip-{}",
            crate::startup_process::unique()
        ));
        fs::create_dir(&working).map_err(|error| error.to_string())?;
        let data = working.join("data");
        fs::create_dir(&data).map_err(|error| error.to_string())?;
        let request: Request = serde_json::from_value(json!({
            "DataDirectory": data,
            "RepositoryRoot": working.join("source"),
            "NativeEngine": true,
            "NoMmproj": true,
        }))
        .map_err(|error| error.to_string())?;

        let cached_file = working.join("cached-model.gguf");
        fs::write(&cached_file, b"123456").map_err(|error| error.to_string())?;
        let selected = Model {
            path: cached_file.clone(),
            bytes: 6,
            sha256: "a".repeat(64),
            url: Some("https://huggingface.co/example/resolve/main/cached-model.gguf".into()),
        };

        let mut received = 0_usize;
        prepare_weights_with(
            &request,
            &working,
            &selected,
            true,
            "existing",
            None,
            None,
            |_, _, _, _, _| {
                received += 1;
                Ok(())
            },
            |_, _, _| Ok(()),
        )?;
        assert_eq!(received, 1);
        let progress: Value = serde_json::from_slice(
            &fs::read(working.join("chat-setup-progress.json"))
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(progress["SchemaVersion"], 2);
        assert_eq!(progress["Stage"], "ready");
        assert_eq!(progress["Kind"], "existing");
        assert_eq!(progress["Activity"], "ready");
        assert!(progress.get("ArtifactPath").is_none());

        status(
            &working,
            "model",
            "Checking Swift model files…",
            "first-run",
            "checking",
            Some((&cached_file, 6)),
        )?;
        let with_artifact: Value = serde_json::from_slice(
            &fs::read(working.join("chat-setup-progress.json"))
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(with_artifact["SchemaVersion"], 2);
        assert_eq!(with_artifact["ArtifactPath"], json!(cached_file));
        assert_eq!(with_artifact["ExpectedBytes"], 6);

        fs::remove_dir_all(working).map_err(|error| error.to_string())
    }
}
