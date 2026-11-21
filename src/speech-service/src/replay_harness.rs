//! Shared source-bound independent replay intake, verification, and reporting harness.
#![forbid(unsafe_code)]

use std::{
    ffi::{OsStr, OsString},
    io::{self, Read as _, Write as _},
    path::{Path, PathBuf},
    process::ExitCode,
};

use bonsai_artifacts::{ArtifactError, VerifiedFile, open_locked_file};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};

/// Source-bound provider identity parsed from a replay request JSON payload.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Identity {
    /// Lowercase hexadecimal 20-byte Git commit SHA.
    pub revision: String,
    /// Compiler version/toolchain label.
    pub compiler: String,
    /// Target triple string (candidate must match `aarch64-pc-windows-msvc`).
    pub target: String,
    /// Lowercase hexadecimal 32-byte SHA-256 digest of the provider executable.
    pub executable_sha256: String,
    /// Arithmetic/backend provenance identifier.
    pub arithmetic: String,
}

/// Validated zero-copy view over a parsed [`Identity`].
#[derive(Clone, Copy, Debug)]
pub struct ParsedIdentity<'a> {
    /// Decoded 20-byte Git commit revision.
    pub revision: [u8; 20],
    /// Non-empty compiler label.
    pub compiler: &'a str,
    /// Non-empty target triple.
    pub target: &'a str,
    /// Decoded 32-byte executable SHA-256 digest.
    pub executable: [u8; 32],
    /// Non-empty arithmetic provenance identifier.
    pub arithmetic: &'a str,
}

impl Identity {
    /// Validates identity text bounds and decodes its hexadecimal digests.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidData`] if any metadata string is empty,
    /// exceeds 4,096 bytes, contains control characters, or carries invalid or
    /// all-zero hexadecimal digests.
    pub fn parse(&self) -> io::Result<ParsedIdentity<'_>> {
        if [&self.compiler, &self.target, &self.arithmetic]
            .iter()
            .any(|text| {
                text.trim().is_empty() || text.len() > 4096 || text.chars().any(char::is_control)
            })
        {
            return Err(invalid());
        }
        Ok(ParsedIdentity {
            revision: hexadecimal(&self.revision)?,
            compiler: &self.compiler,
            target: &self.target,
            executable: hexadecimal(&self.executable_sha256)?,
            arithmetic: &self.arithmetic,
        })
    }
}

/// Content-addressed local artifact reference declared inside a replay request.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Artifact {
    /// File system path to the artifact.
    pub path: PathBuf,
    /// Exact expected byte length.
    pub bytes: u64,
    /// Lowercase hexadecimal 32-byte SHA-256 digest.
    pub sha256: String,
}

/// Comparison tolerance rule declared per stage in a replay request.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum Rule {
    /// Require 100% bit-exact IEEE-754 identity (`f32::to_bits`).
    Bits,
    /// Allow finite absolute/relative deviation bounds.
    Finite {
        /// Maximum non-negative absolute difference.
        absolute: f32,
        /// Maximum non-negative relative difference.
        relative: f32,
    },
}

impl Rule {
    /// Validates tolerance bounds and maps this rule into a crate-local tolerance type.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidData`] if a [`Rule::Finite`] bound is
    /// non-finite or negative.
    pub fn map<T>(self, bits: T, finite: impl FnOnce(f32, f32) -> T) -> io::Result<T> {
        match self {
            Self::Bits => Ok(bits),
            Self::Finite { absolute, relative }
                if absolute.is_finite()
                    && relative.is_finite()
                    && absolute >= 0.0
                    && relative >= 0.0 =>
            {
                Ok(finite(absolute, relative))
            }
            Self::Finite { .. } => Err(invalid()),
        }
    }
}

/// Borrowed clock budget fields shared by source-bound ASR and TTS replay requests.
#[derive(Clone, Copy, Debug)]
pub struct ClockSpec<'a> {
    /// Schema version number (must equal `1`).
    pub schema_version: u32,
    /// Owner monotonic counter start tick.
    pub owner_started_tick: u64,
    /// Owner monotonic counter hard deadline tick.
    pub work_deadline_tick: u64,
    /// Expected monotonic counter frequency in Hz.
    pub clock_frequency: u64,
    /// Description of the fixed replay workload.
    pub fixed_work: &'a str,
    /// Conservative estimated runtime in whole seconds (`1..=240`).
    pub estimated_seconds: u64,
}

impl ClockSpec<'_> {
    /// Validates clock bounds and remaining deadline slack against an observed timestamp.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidData`] if schema version, tick bounds,
    /// frequency, allowance ceiling (`270` seconds), or `fixed_work` validation fails.
    pub fn admit(&self, now: u64, frequency: u64) -> io::Result<()> {
        let allowance = self
            .work_deadline_tick
            .checked_sub(self.owner_started_tick)
            .ok_or_else(invalid)?;
        let maximum = frequency.checked_mul(270).ok_or_else(invalid)?;
        let estimate = frequency
            .checked_mul(self.estimated_seconds)
            .ok_or_else(invalid)?;
        let remaining = self
            .work_deadline_tick
            .checked_sub(now)
            .ok_or_else(invalid)?;
        if self.schema_version != 1
            || frequency == 0
            || frequency != self.clock_frequency
            || self.owner_started_tick == 0
            || self.owner_started_tick > now
            || now >= self.work_deadline_tick
            || allowance > maximum
            || estimate > remaining
            || !(1..=240).contains(&self.estimated_seconds)
            || self.fixed_work.trim().is_empty()
            || self.fixed_work.len() > 1024
            || self.fixed_work.chars().any(char::is_control)
        {
            return Err(invalid());
        }
        Ok(())
    }

    /// Evaluates whether an observed `(tick, frequency)` pair is past deadline or invalid.
    #[must_use]
    pub const fn is_expired_at(&self, tick: u64, frequency: u64) -> bool {
        frequency != self.clock_frequency
            || tick < self.owner_started_tick
            || tick >= self.work_deadline_tick
    }

    /// Queries the host monotonic counter and reports whether this budget has expired.
    #[must_use]
    pub fn expired(&self) -> bool {
        #[cfg(windows)]
        {
            bonsai_windows_process::monotonic_ticks().map_or(true, |(tick, frequency)| {
                self.is_expired_at(tick, frequency)
            })
        }
        #[cfg(not(windows))]
        {
            true
        }
    }

    /// Queries the host clock, admits this spec, and records clock fields into `report`.
    ///
    /// # Errors
    /// Returns the clock query error, [`io::ErrorKind::Unsupported`] on non-Windows
    /// targets, or [`io::ErrorKind::InvalidData`] from [`ClockSpec::admit`].
    pub fn admit_and_record(&self, report: &mut Value) -> io::Result<()> {
        #[cfg(windows)]
        {
            let (tick, frequency) = bonsai_windows_process::monotonic_ticks()?;
            self.admit(tick, frequency)?;
            report["fixedWork"] = self.fixed_work.into();
            report["estimatedSeconds"] = self.estimated_seconds.into();
            report["ownerStartedTick"] = self.owner_started_tick.into();
            report["workDeadlineTick"] = self.work_deadline_tick.into();
            report["clockFrequency"] = frequency.into();
            Ok(())
        }
        #[cfg(not(windows))]
        {
            let _keep = (self, report);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Native Windows ARM64 is required.",
            ))
        }
    }
}

/// Callbacks and report metadata supplied by each model-specific replay binary.
pub struct ReplayRunConfig<'a> {
    /// JSON `provider` identifier (for example `"bonsai-asr-replay"`).
    pub provider: &'a str,
    /// Human-readable `powerScope` notice included in the JSON report.
    pub power_scope: &'a str,
    /// Human-readable `cleanup` notice included in the JSON report.
    pub cleanup: &'a str,
    /// Computes a raw 32-byte SHA-256 digest over request bytes.
    pub sha256: fn(&[u8]) -> io::Result<[u8; 32]>,
    /// Queries an endpoint power snapshot JSON value.
    pub power: fn() -> Value,
    /// Compares `powerBefore` and `powerAfter` endpoint JSON values.
    pub compare_power: fn(&Value, &Value) -> Value,
}

/// Returns the canonical `InvalidData` error for malformed replay requests.
#[must_use]
pub fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid source-bound replay request.",
    )
}

/// Decodes a lowercase non-zero fixed-width hexadecimal digest string.
///
/// # Errors
/// Returns [`io::ErrorKind::InvalidData`] if `text` has the wrong length,
/// contains uppercase or non-hex bytes, or decodes to all zero bytes.
pub fn hexadecimal<const N: usize>(text: &str) -> io::Result<[u8; N]> {
    if text.len() != N * 2
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid());
    }
    let mut bytes = [0; N];
    for (output, pair) in bytes.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
        let nibble = |byte: u8| {
            if byte.is_ascii_digit() {
                byte - b'0'
            } else {
                byte - b'a' + 10
            }
        };
        *output = nibble(pair[0]) * 16 + nibble(pair[1]);
    }
    if bytes.iter().all(|byte| *byte == 0) {
        return Err(invalid());
    }
    Ok(bytes)
}

/// Parses exact `--request <abs-path> --sha256 <64-hex>` replay CLI arguments.
///
/// # Errors
/// Returns [`io::ErrorKind::InvalidData`] if flags, ordering, path absoluteness,
/// or SHA-256 hex format fail validation.
pub fn arguments(arguments: impl IntoIterator<Item = OsString>) -> io::Result<(PathBuf, String)> {
    let mut arguments = arguments.into_iter();
    if arguments.next().as_deref() != Some(OsStr::new("--request")) {
        return Err(invalid());
    }
    let path = PathBuf::from(arguments.next().ok_or_else(invalid)?);
    if arguments.next().as_deref() != Some(OsStr::new("--sha256")) {
        return Err(invalid());
    }
    let digest = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or_else(invalid)?;
    let _bytes: [u8; 32] = hexadecimal(&digest)?;
    if arguments.next().is_some() || !path.is_absolute() {
        return Err(invalid());
    }
    Ok((path, digest))
}

/// Reads an artifact via [`open_locked_file`] enforcing an exact upper byte limit.
///
/// # Errors
/// Returns an I/O error if opening fails, metadata/read size exceeds `maximum`,
/// or reading fails.
pub fn bounded_bytes(path: &Path, maximum: u64) -> io::Result<Vec<u8>> {
    let file = open_locked_file(path).map_err(io::Error::other)?;
    if file.metadata()?.len() > maximum {
        return Err(invalid());
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).map_err(|_| invalid())? > maximum {
        return Err(invalid());
    }
    Ok(bytes)
}

/// Validates distinct non-self reference vs candidate provider identities for native replay.
///
/// # Errors
/// Returns [`io::ErrorKind::InvalidData`] if arithmetic provenance matches
/// self-comparison, candidate provenance mismatches `candidate_arithmetic`,
/// executables match, or candidate target is not `aarch64-pc-windows-msvc`.
pub fn validate_provider_pair(
    reference: &ParsedIdentity<'_>,
    candidate: &ParsedIdentity<'_>,
    candidate_arithmetic: &str,
) -> io::Result<()> {
    if candidate.arithmetic != candidate_arithmetic
        || reference.arithmetic == candidate_arithmetic
        || reference.executable == candidate.executable
        || candidate.target != "aarch64-pc-windows-msvc"
    {
        return Err(invalid());
    }
    Ok(())
}

/// Verifies `artifact` byte length and digest while polling cancellation via `expired`.
///
/// # Errors
/// Returns [`io::ErrorKind::Interrupted`] if `expired()` returns `true` before
/// or during verification, or [`io::Error`] on [`ArtifactError`].
pub fn verify_artifact(
    artifact: &Artifact,
    expired: &(impl Fn() -> bool + ?Sized),
) -> io::Result<VerifiedFile> {
    if expired() {
        return Err(io::ErrorKind::Interrupted.into());
    }
    VerifiedFile::verify_with(&artifact.path, artifact.bytes, &artifact.sha256, || {
        if expired() {
            Err(ArtifactError::Cancelled)
        } else {
            Ok(())
        }
    })
    .map_err(io::Error::other)
}

/// Verifies the currently executing candidate replay binary against `candidate_sha256`.
///
/// # Errors
/// Returns any error from [`std::env::current_exe`], [`Path::metadata`], or [`verify_artifact`].
pub fn verify_candidate_image(
    candidate_sha256: &str,
    expired: &(impl Fn() -> bool + ?Sized),
    report: &mut Value,
) -> io::Result<VerifiedFile> {
    report["phase"] = "verify-candidate-image".into();
    let image = std::env::current_exe()?;
    verify_artifact(
        &Artifact {
            bytes: image.metadata()?.len(),
            path: image,
            sha256: candidate_sha256.to_owned(),
        },
        expired,
    )
}

/// Queries host endpoint power status via caller-supplied snapshot constructors.
#[must_use]
pub fn power_snapshot(
    observed: impl FnOnce(Option<bool>, bool, Option<u8>) -> Value,
    unknown: impl FnOnce() -> Value,
) -> Value {
    #[cfg(windows)]
    {
        bonsai_windows_process::power_status().map_or_else(
            |_| unknown(),
            |status| {
                observed(
                    status.ac_online,
                    status.battery_saver,
                    status.battery_percent,
                )
            },
        )
    }
    #[cfg(not(windows))]
    {
        let _keep = observed;
        unknown()
    }
}

/// Executes CLI request intake, SHA-256 verification, unwinding guard, and JSON reporting.
///
/// # Errors
/// Returns an [`io::Error`] if writing the final JSON report to `stdout` fails.
pub fn run_replay<R: DeserializeOwned>(
    config: &ReplayRunConfig<'_>,
    execute: impl FnOnce(&R, &mut Value) -> io::Result<()>,
) -> io::Result<bool> {
    let mut report = json!({
        "schemaVersion": 1,
        "provider": config.provider,
        "passed": false,
        "completeReplayPassed": false,
        "productionQualified": false,
        "nativeRetirementQualified": false,
        "performanceComparisonEligible": false,
        "phase": "request-intake",
        "powerBefore": (config.power)(),
        "powerScope": config.power_scope,
        "cleanup": config.cleanup,
    });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> io::Result<()> {
        let (path, digest) = arguments(std::env::args_os().skip(1))?;
        let bytes = bounded_bytes(&path, 64 * 1024)?;
        if (config.sha256)(&bytes)? != hexadecimal::<32>(&digest)? {
            return Err(invalid());
        }
        report["requestSha256"] = digest.into();
        let request: R = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        execute(&request, &mut report)
    }));
    let passed = matches!(result, Ok(Ok(())));
    report["failureKind"] = match &result {
        Ok(Ok(())) => Value::Null,
        Ok(Err(error)) => format!("{:?}", error.kind()).into(),
        Err(_) => "unwind".into(),
    };
    report["passed"] = passed.into();
    report["status"] = if passed {
        "complete-replay-passed"
    } else {
        "failed-or-incomplete"
    }
    .into();
    report["powerAfter"] = (config.power)();
    report["powerConditions"] =
        (config.compare_power)(&report["powerBefore"], &report["powerAfter"]);
    serde_json::to_writer(io::stdout().lock(), &report)?;
    writeln!(io::stdout().lock())?;
    Ok(passed)
}

/// Shared `main()` entry wrapper mapping replay execution results to process exit codes.
pub fn run_cli_main(
    run_consumer: impl FnOnce() -> io::Result<bool>,
    failure_message: &str,
) -> ExitCode {
    match run_consumer() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(_) => {
            let _written = writeln!(io::stderr().lock(), "{failure_message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ClockSpec, Identity, Rule, arguments, hexadecimal, validate_provider_pair};
    use serde_json::json;
    use std::io;

    #[test]
    fn harness_validates_hex_arguments_and_rules() -> io::Result<()> {
        assert_eq!(hexadecimal::<2>("01fe")?, [0x01, 0xfe]);
        for bad in ["", "0000", "01FE", "zz"] {
            assert!(hexadecimal::<2>(bad).is_err());
        }
        let digest = "01".repeat(32);
        let ok_path = if cfg!(windows) {
            "C:\\x.json"
        } else {
            "/tmp/x.json"
        };
        let ok = arguments(["--request", ok_path, "--sha256", &digest].map(Into::into))?;
        assert_eq!(ok.1, digest);
        assert!(
            arguments(["--request", "relative.json", "--sha256", &digest].map(Into::into)).is_err()
        );
        assert_eq!(Rule::Bits.map(0_u32, |a, _| a.to_bits())?, 0_u32);
        assert!(
            Rule::Finite {
                absolute: -0.1,
                relative: 0.0
            }
            .map((), |_, _| ())
            .is_err()
        );
        assert!(
            Rule::Finite {
                absolute: f32::NAN,
                relative: 0.0
            }
            .map((), |_, _| ())
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn harness_validates_clock_budget_and_provider_pairs() -> io::Result<()> {
        let clock = ClockSpec {
            schema_version: 1,
            owner_started_tick: 1000,
            work_deadline_tick: 271_000,
            clock_frequency: 1000,
            fixed_work: "one complete replay",
            estimated_seconds: 240,
        };
        assert!(clock.admit(1000, 1000).is_ok());
        assert!(clock.admit(31_000, 1000).is_ok());
        assert!(clock.admit(31_001, 1000).is_err());
        assert!(clock.is_expired_at(271_000, 1000));
        assert!(!clock.is_expired_at(270_999, 1000));

        let reference: Identity = serde_json::from_value(json!({
            "revision": "01".repeat(20),
            "compiler": "clang",
            "target": "aarch64-pc-windows-msvc",
            "executableSha256": "02".repeat(32),
            "arithmetic": "native-ref"
        }))
        .map_err(io::Error::other)?;
        let candidate: Identity = serde_json::from_value(json!({
            "revision": "01".repeat(20),
            "compiler": "rustc",
            "target": "aarch64-pc-windows-msvc",
            "executableSha256": "03".repeat(32),
            "arithmetic": "rust-cand"
        }))
        .map_err(io::Error::other)?;
        validate_provider_pair(&reference.parse()?, &candidate.parse()?, "rust-cand")?;
        assert!(
            validate_provider_pair(&candidate.parse()?, &candidate.parse()?, "rust-cand").is_err()
        );
        Ok(())
    }
}
