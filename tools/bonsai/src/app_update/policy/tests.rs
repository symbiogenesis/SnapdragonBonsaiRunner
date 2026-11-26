use super::*;
use serde_json::json;

fn release(version: &str) -> Value {
    let asset = |name: &str, bytes| {
        json!({"name": name, "size": bytes,
        "digest": format!("sha256:{}", "a".repeat(64)), "state": "uploaded",
        "browser_download_url": canonical_asset(version, name)})
    };
    json!({"tag_name": format!("v{version}"), "draft": false, "prerelease": true,
        "published_at": "2026-11-12T00:00:00Z",
        "html_url": format!("https://github.com/{REPOSITORY}/releases/tag/v{version}"),
        "assets": [asset("release-manifest.json", 512), asset("SHA256SUMS.txt", 256),
            asset(&format!("SwiftBonsai2Runner-{version}-windows-arm64.msi"), 4096)]})
}

fn selected(releases: &Value) -> crate::Result<(bool, Option<Candidate>)> {
    select(
        &serde_json::to_vec(releases).map_err(|error| error.to_string())?,
        "0.1.0",
    )
}

fn fixture() -> crate::Result<Candidate> {
    selected(&json!([release("0.1.1")]))?
        .1
        .ok_or_else(|| "Expected newer fixture release.".into())
}

fn manifest(candidate: &Candidate) -> Value {
    json!({"schemaVersion": 1, "version": candidate.version,
        "tag": format!("v{}", candidate.version), "target": TARGET,
        "sourceRevision": "b".repeat(40), "installerSmoke": "passed-install-inspect-uninstall-without-model",
        "assets": [{"name": candidate.msi.name, "bytes": candidate.msi.bytes, "sha256": candidate.msi.sha256}]})
}

fn named_portable_release(version: &str, name: &str) -> Value {
    let mut value = release(version);
    value["assets"][2]["name"] = name.into();
    value["assets"][2]["browser_download_url"] = canonical_asset(version, name).into();
    value
}

fn portable_release(version: &str) -> Value {
    named_portable_release(
        version,
        &format!("SnapdragonBonsaiRunner-{version}-windows-arm64.exe"),
    )
}

fn legacy_portable_release(version: &str) -> Value {
    named_portable_release(
        version,
        &format!("SwiftBonsai2Runner-{version}-windows-arm64.exe"),
    )
}

fn selected_portable(releases: &Value) -> crate::Result<(bool, Option<Candidate>)> {
    select_portable(
        &serde_json::to_vec(releases).map_err(|error| error.to_string())?,
        "0.1.0",
    )
}

fn portable_manifest(candidate: &Candidate) -> Value {
    let mut value = manifest(candidate);
    value["distribution"] = "portable-executable".into();
    value["executableSmoke"] = "passed-inspect-extract-verify-without-model".into();
    value["manifestSha256"] = "c".repeat(64).into();
    value
}

#[test]
fn portable_discovery_selects_newest_published_numeric_version_without_fallback()
-> crate::Result<()> {
    let mut draft = portable_release("0.2.0");
    draft["draft"] = true.into();
    let mut stable = portable_release("0.1.2");
    stable["prerelease"] = false.into();
    let mut historical = release("0.1.1");
    historical["assets"] = json!([]);
    let newest = portable_release("0.1.10");
    for values in [
        json!([historical, draft, stable, newest]),
        json!([newest, stable, draft, historical]),
    ] {
        let candidate = selected_portable(&values)?
            .1
            .ok_or("Missing portable release.")?;
        assert_eq!(candidate.version, "0.1.10");
        assert_eq!(
            candidate.msi.name,
            "SnapdragonBonsaiRunner-0.1.10-windows-arm64.exe"
        );
    }
    let mut invalid_newest = portable_release("0.1.11");
    invalid_newest["assets"][2]["digest"] = Value::Null;
    assert!(selected_portable(&json!([newest, invalid_newest])).is_err());
    assert!(selected_portable(&json!([newest, newest])).is_err());
    let (published, candidate) = selected_portable(&json!([portable_release("0.1.0")]))?;
    assert!(published);
    assert!(candidate.is_none());
    let (published, candidate) = selected_portable(&json!([]))?;
    assert!(!published);
    assert!(candidate.is_none());
    Ok(())
}

#[test]
fn portable_discovery_accepts_both_asset_names_but_rejects_legacy_repo_and_dual_assets()
-> crate::Result<()> {
    assert_eq!(REPOSITORY, "symbiogenesis/SnapdragonBonsaiRunner");
    for (release_entry, expected_name) in [
        (
            portable_release("0.3.7"),
            "SnapdragonBonsaiRunner-0.3.7-windows-arm64.exe",
        ),
        (
            legacy_portable_release("0.3.7"),
            "SwiftBonsai2Runner-0.3.7-windows-arm64.exe",
        ),
    ] {
        let candidate = selected_portable(&json!([release_entry]))?
            .1
            .ok_or("Missing portable release.")?;
        assert_eq!(candidate.msi.name, expected_name);
        let manifest_json = portable_manifest(&candidate);
        let checksum = format!("{}  {}\n", candidate.msi.sha256, candidate.msi.name);
        let verified = portable(
            &candidate,
            &serde_json::to_vec(&manifest_json).map_err(|error| error.to_string())?,
            checksum.as_bytes(),
        )?;
        verified.validate(&candidate.version)?;

        let mut wrong_manifest = manifest_json;
        let other_name = if expected_name.starts_with("Snapdragon") {
            "SwiftBonsai2Runner-0.3.7-windows-arm64.exe"
        } else {
            "SnapdragonBonsaiRunner-0.3.7-windows-arm64.exe"
        };
        wrong_manifest["assets"][0]["name"] = other_name.into();
        assert!(
            portable(
                &candidate,
                &serde_json::to_vec(&wrong_manifest).map_err(|error| error.to_string())?,
                checksum.as_bytes(),
            )
            .is_err()
        );
        let wrong_checksum = format!("{}  {other_name}\n", candidate.msi.sha256);
        assert!(
            portable(
                &candidate,
                &serde_json::to_vec(&portable_manifest(&candidate))
                    .map_err(|error| error.to_string())?,
                wrong_checksum.as_bytes(),
            )
            .is_err()
        );
    }

    let mut dual = portable_release("0.3.7");
    let legacy_asset = legacy_portable_release("0.3.7")["assets"][2].clone();
    dual["assets"]
        .as_array_mut()
        .ok_or("Missing fixture assets.")?
        .push(legacy_asset);
    assert!(selected_portable(&json!([dual])).is_err());

    for mutation in 0..2 {
        let mut legacy_url = portable_release("0.1.1");
        let old_base = "https://github.com/symbiogenesis/SwiftBonsai2Runner/releases";
        if mutation == 0 {
            legacy_url["html_url"] = format!("{old_base}/tag/v0.1.1").into();
        } else {
            let name = legacy_url["assets"][2]["name"]
                .as_str()
                .ok_or("Missing fixture asset name.")?
                .to_owned();
            legacy_url["assets"][2]["browser_download_url"] =
                format!("{old_base}/download/v0.1.1/{name}").into();
        }
        assert!(selected_portable(&json!([legacy_url])).is_err());
    }
    Ok(())
}

#[test]
fn portable_discovery_rejects_wrong_arch_foreign_urls_missing_digests_and_duplicates()
-> crate::Result<()> {
    for mutation in 0..6 {
        let mut value = portable_release("0.1.1");
        match mutation {
            0 => value["html_url"] = "https://github.com/other/repo/releases/tag/v0.1.1".into(),
            1 => value["assets"][2]["name"] = "SwiftBonsai2Runner-0.1.1-windows-x64.exe".into(),
            2 => value["assets"][2]["digest"] = Value::Null,
            3 => {
                value["assets"][2]["browser_download_url"] =
                    "https://example.com/update.exe".into();
            }
            4 => value["assets"][2]["size"] = (MAX_INSTALLER + 1).into(),
            _ => {
                let repeated = value["assets"][2].clone();
                value["assets"]
                    .as_array_mut()
                    .ok_or("Missing fixture assets.")?
                    .push(repeated);
            }
        }
        assert!(selected_portable(&json!([value])).is_err());
    }
    assert!(selected_portable(&json!([release("0.1.1")])).is_err());
    assert!(selected(&json!([portable_release("0.1.1")])).is_err());
    Ok(())
}

#[test]
fn portable_manifest_binds_source_target_smoke_and_three_way_executable_identity()
-> crate::Result<()> {
    let candidate = selected_portable(&json!([portable_release("0.1.1")]))?
        .1
        .ok_or("Missing portable release.")?;
    let valid = portable_manifest(&candidate);
    let encoded = serde_json::to_vec(&valid).map_err(|error| error.to_string())?;
    let checksum = format!("{}  {}\n", candidate.msi.sha256, candidate.msi.name);
    let verified = portable(&candidate, &encoded, checksum.as_bytes())?;
    assert_eq!(verified.distribution(), Distribution::Portable);
    verified.validate(&candidate.version)?;
    verified.verify_source(
        &serde_json::to_vec(&json!({"sha": "b".repeat(40)})).map_err(|error| error.to_string())?,
    )?;
    assert!(
        verified
            .verify_source(
                &serde_json::to_vec(&json!({"sha": "c".repeat(40)}))
                    .map_err(|error| error.to_string())?
            )
            .is_err()
    );
    for (key, value) in [
        ("schemaVersion", json!(2)),
        ("tag", json!("v0.1.2")),
        ("version", json!("0.1.2")),
        ("target", json!("x86_64-pc-windows-msvc")),
        ("sourceRevision", json!("main")),
        ("distribution", json!("msi")),
        ("executableSmoke", json!("pending")),
        ("manifestSha256", json!("invalid")),
    ] {
        let mut invalid = valid.clone();
        invalid[key] = value;
        assert!(
            portable(
                &candidate,
                &serde_json::to_vec(&invalid).map_err(|error| error.to_string())?,
                checksum.as_bytes()
            )
            .is_err()
        );
    }
    for mutation in 0..4 {
        let mut invalid = valid.clone();
        match mutation {
            0 => invalid["assets"][0]["bytes"] = 4097.into(),
            1 => invalid["assets"][0]["sha256"] = "b".repeat(64).into(),
            2 => invalid["assets"][0]["name"] = "wrong.exe".into(),
            _ => {
                let repeated = invalid["assets"][0].clone();
                invalid["assets"]
                    .as_array_mut()
                    .ok_or("Missing fixture assets.")?
                    .push(repeated);
            }
        }
        assert!(
            portable(
                &candidate,
                &serde_json::to_vec(&invalid).map_err(|error| error.to_string())?,
                checksum.as_bytes()
            )
            .is_err()
        );
    }
    let wrong_checksum = format!("{}  {}\n", "c".repeat(64), candidate.msi.name);
    assert!(portable(&candidate, &encoded, wrong_checksum.as_bytes()).is_err());
    assert!(
        portable(
            &candidate,
            &encoded,
            format!("{checksum}{checksum}").as_bytes()
        )
        .is_err()
    );
    assert!(installer(&candidate, &encoded, checksum.as_bytes()).is_err());
    Ok(())
}

#[test]
fn confirmation_identity_binds_exact_source_and_executable_without_version_only_reuse()
-> crate::Result<()> {
    let candidate = selected_portable(&json!([portable_release("0.1.1")]))?
        .1
        .ok_or("Missing portable fixture.")?;
    let published = portable(
        &candidate,
        &serde_json::to_vec(&portable_manifest(&candidate)).map_err(|error| error.to_string())?,
        format!("{}  {}\n", candidate.msi.sha256, candidate.msi.name).as_bytes(),
    )?;
    let identity = published.release_identity()?;
    published.verify_release_identity(&identity)?;
    let encoded = serde_json::to_value(&published).map_err(|error| error.to_string())?;
    for pointer in [
        "/sourceRevision",
        "/asset/sha256",
        "/asset/bytes",
        "/version",
    ] {
        let mut changed = encoded.clone();
        let replacement = match pointer {
            "/sourceRevision" => json!("d".repeat(40)),
            "/asset/sha256" => json!("e".repeat(64)),
            "/asset/bytes" => json!(4097),
            _ => json!("0.1.2"),
        };
        *changed
            .pointer_mut(pointer)
            .ok_or("Missing fixture identity field.")? = replacement;
        let changed: Installer =
            serde_json::from_value(changed).map_err(|error| error.to_string())?;
        assert!(changed.verify_release_identity(&identity).is_err());
    }
    for invalid in [String::new(), "0".repeat(64), "A".repeat(64)] {
        assert!(published.verify_release_identity(&invalid).is_err());
    }
    Ok(())
}

#[test]
fn portable_replacement_identity_admits_legacy_repo_handoff_only_for_legacy_portable_asset()
-> crate::Result<()> {
    let candidate = selected_portable(&json!([legacy_portable_release("0.3.7")]))?
        .1
        .ok_or("Missing legacy portable fixture.")?;
    let published = portable(
        &candidate,
        &serde_json::to_vec(&portable_manifest(&candidate)).map_err(|error| error.to_string())?,
        format!("{}  {}\n", candidate.msi.sha256, candidate.msi.name).as_bytes(),
    )?;
    let legacy = published
        .legacy_portable_handoff()
        .ok_or("Expected legacy portable handoff.")?;
    assert_eq!(
        legacy.asset.url,
        "https://github.com/symbiogenesis/SwiftBonsai2Runner/releases/download/v0.3.7/SwiftBonsai2Runner-0.3.7-windows-arm64.exe"
    );
    assert_eq!(legacy.version, published.version);
    assert_eq!(legacy.source_revision, published.source_revision);
    assert_eq!(legacy.target, published.target);
    assert_eq!(legacy.asset.name, published.asset.name);
    assert_eq!(legacy.asset.bytes, published.asset.bytes);
    assert_eq!(legacy.asset.sha256, published.asset.sha256);
    assert!(legacy.validate("0.3.7").is_err());

    let canonical_id = published.release_identity()?;
    let legacy_id = legacy.release_identity()?;
    assert_ne!(canonical_id, legacy_id);
    assert!(published.resolve_portable_replacement(&canonical_id)? == published);
    assert!(published.resolve_portable_replacement(&legacy_id)? == legacy);
    assert!(published.verify_release_identity(&legacy_id).is_err());

    let encoded = serde_json::to_value(&legacy).map_err(|error| error.to_string())?;
    for (pointer, replacement) in [
        ("/sourceRevision", json!("d".repeat(40))),
        ("/asset/sha256", json!("e".repeat(64))),
        ("/asset/bytes", json!(4097)),
        ("/version", json!("0.3.8")),
        (
            "/asset/url",
            json!(
                "https://github.com/symbiogenesis/SwiftBonsai2/releases/download/v0.3.7/SwiftBonsai2Runner-0.3.7-windows-arm64.exe"
            ),
        ),
        (
            "/asset/url",
            json!(
                "https://github.com/other/repo/releases/download/v0.3.7/SwiftBonsai2Runner-0.3.7-windows-arm64.exe"
            ),
        ),
    ] {
        let mut changed = encoded.clone();
        *changed
            .pointer_mut(pointer)
            .ok_or("Missing fixture identity field.")? = replacement;
        let changed: Installer =
            serde_json::from_value(changed).map_err(|error| error.to_string())?;
        let mutated_id = changed.release_identity()?;
        assert!(published.resolve_portable_replacement(&mutated_id).is_err());
    }
    for invalid in [String::new(), "0".repeat(64), "A".repeat(64)] {
        assert!(published.resolve_portable_replacement(&invalid).is_err());
    }

    let snapdragon_candidate = selected_portable(&json!([portable_release("0.3.7")]))?
        .1
        .ok_or("Missing Snapdragon portable fixture.")?;
    let snapdragon_published = portable(
        &snapdragon_candidate,
        &serde_json::to_vec(&portable_manifest(&snapdragon_candidate))
            .map_err(|error| error.to_string())?,
        format!(
            "{}  {}\n",
            snapdragon_candidate.msi.sha256, snapdragon_candidate.msi.name
        )
        .as_bytes(),
    )?;
    assert!(snapdragon_published.legacy_portable_handoff().is_none());
    let mut synthetic_snapdragon = snapdragon_published.clone();
    synthetic_snapdragon.asset.url =
        canonical_asset_for(LEGACY_REPOSITORY, "0.3.7", &snapdragon_published.asset.name);
    let synthetic_snapdragon_id = synthetic_snapdragon.release_identity()?;
    assert!(
        snapdragon_published
            .resolve_portable_replacement(&synthetic_snapdragon_id)
            .is_err()
    );

    let msi_candidate = selected(&json!([release("0.3.7")]))?
        .1
        .ok_or("Missing MSI fixture.")?;
    let msi_published = installer(
        &msi_candidate,
        &serde_json::to_vec(&manifest(&msi_candidate)).map_err(|error| error.to_string())?,
        format!("{}  {}\n", msi_candidate.msi.sha256, msi_candidate.msi.name).as_bytes(),
    )?;
    assert!(msi_published.legacy_portable_handoff().is_none());
    let mut synthetic_msi = msi_published.clone();
    synthetic_msi.asset.url =
        canonical_asset_for(LEGACY_REPOSITORY, "0.3.7", &msi_published.asset.name);
    let synthetic_msi_id = synthetic_msi.release_identity()?;
    assert!(
        msi_published
            .resolve_portable_replacement(&synthetic_msi_id)
            .is_err()
    );
    Ok(())
}

#[test]
fn versions_compare_numerically_and_reject_noncanonical_or_out_of_msi_bounds() -> crate::Result<()>
{
    assert!(Version::parse("0.10.0")? > Version::parse("0.9.99")?);
    assert_eq!(Version::parse("255.255.65535")?, Version(255, 255, 65535));
    for invalid in [
        "",
        "v0.1.1",
        "0.01.1",
        "00.1.1",
        "0.1",
        "0.1.1.0",
        "0.1.2-beta",
        "256.0.0",
        "0.256.0",
        "0.0.65536",
        "0.1.-1",
        "../0.1.1",
    ] {
        assert!(Version::parse(invalid).is_err(), "accepted {invalid}");
    }
    assert!(
        Version::parse(env!("CARGO_PKG_VERSION"))?
            .newer_than_current()
            .is_err()
    );
    Ok(())
}

#[test]
fn discovery_handles_empty_current_draft_and_numeric_prerelease_ordering() -> crate::Result<()> {
    let (published, candidate) = selected(&json!([]))?;
    assert!(!published);
    assert!(candidate.is_none());
    let mut draft = release("0.1.99");
    draft["draft"] = true.into();
    let (_, candidate) = selected(&json!([
        release("0.1.2"),
        draft,
        release("0.1.10"),
        release("0.1.0")
    ]))?;
    assert_eq!(
        candidate.ok_or("Missing update fixture.")?.version,
        "0.1.10"
    );
    let (published, candidate) = selected(&json!([release("0.1.0")]))?;
    assert!(published);
    assert!(candidate.is_none());
    assert!(select(b"not-json", "0.1.0").is_err());
    assert!(selected(&json!({"releases": []})).is_err());
    Ok(())
}

#[test]
fn discovery_rejects_foreign_urls_wrong_arch_missing_digests_and_duplicate_assets()
-> crate::Result<()> {
    for mutation in 0..5 {
        let mut value = release("0.1.1");
        match mutation {
            0 => value["html_url"] = "https://github.com/other/repo/releases/tag/v0.1.1".into(),
            1 => value["assets"][2]["name"] = "SwiftBonsai2Runner-0.1.1-windows-x64.msi".into(),
            2 => value["assets"][2]["digest"] = Value::Null,
            3 => {
                value["assets"][2]["browser_download_url"] =
                    "https://example.com/update.msi".into();
            }
            _ => {
                let repeated = value["assets"][2].clone();
                value["assets"]
                    .as_array_mut()
                    .ok_or("Missing fixture assets.")?
                    .push(repeated);
            }
        }
        assert!(selected(&json!([value])).is_err());
    }
    Ok(())
}

#[test]
fn discovery_selects_before_asset_validation_and_never_falls_back() -> crate::Result<()> {
    let mut incomplete = release("0.1.1");
    incomplete["assets"] = json!([]);
    for values in [
        json!([incomplete, release("0.1.2")]),
        json!([release("0.1.2"), incomplete]),
    ] {
        assert_eq!(
            selected(&values)?
                .1
                .ok_or("Missing newest fixture.")?
                .version,
            "0.1.2"
        );
    }
    let mut invalid_newest = release("0.1.3");
    invalid_newest["assets"][2]["digest"] = Value::Null;
    for values in [
        json!([invalid_newest, release("0.1.2")]),
        json!([release("0.1.2"), invalid_newest]),
    ] {
        assert!(selected(&values).is_err());
    }
    Ok(())
}

#[test]
fn discovery_rejects_ambiguous_repeated_published_versions() {
    let mut changed = release("0.1.1");
    changed["assets"][2]["digest"] = format!("sha256:{}", "b".repeat(64)).into();
    assert!(selected(&json!([release("0.1.1"), changed])).is_err());
    assert!(selected(&json!([changed, release("0.1.1")])).is_err());
    assert!(selected(&json!([release("0.1.1"), release("0.1.1")])).is_err());
}

#[test]
fn manifest_requires_same_version_source_target_size_and_three_way_checksum() -> crate::Result<()> {
    let candidate = fixture()?;
    let checksum = format!("{}  {}\n", candidate.msi.sha256, candidate.msi.name);
    let valid = manifest(&candidate);
    let encoded = serde_json::to_vec(&valid).map_err(|error| error.to_string())?;
    let verified = installer(&candidate, &encoded, checksum.as_bytes())?;
    verified.verify_source(
        &serde_json::to_vec(&json!({"sha": "b".repeat(40)})).map_err(|error| error.to_string())?,
    )?;
    assert!(
        verified
            .verify_source(
                &serde_json::to_vec(&json!({"sha": "c".repeat(40)}))
                    .map_err(|error| error.to_string())?
            )
            .is_err()
    );
    for (key, value) in [
        ("target", json!("x86_64-pc-windows-msvc")),
        ("version", json!("0.1.2")),
        ("sourceRevision", json!("main")),
        ("installerSmoke", json!("pending")),
    ] {
        let mut invalid = valid.clone();
        invalid[key] = value;
        assert!(
            installer(
                &candidate,
                &serde_json::to_vec(&invalid).map_err(|error| error.to_string())?,
                checksum.as_bytes()
            )
            .is_err()
        );
    }
    let mut invalid = valid;
    invalid["assets"][0]["bytes"] = 4097.into();
    assert!(
        installer(
            &candidate,
            &serde_json::to_vec(&invalid).map_err(|error| error.to_string())?,
            checksum.as_bytes()
        )
        .is_err()
    );
    let wrong_checksum = format!("{}  {}\n", "c".repeat(64), candidate.msi.name);
    assert!(installer(&candidate, &encoded, wrong_checksum.as_bytes()).is_err());
    assert!(
        installer(
            &candidate,
            &encoded,
            format!("{checksum}{checksum}").as_bytes()
        )
        .is_err()
    );
    assert!(
        installer(
            &candidate,
            &encoded,
            format!("{}  ../update.msi\n", "a".repeat(64)).as_bytes()
        )
        .is_err()
    );
    Ok(())
}
