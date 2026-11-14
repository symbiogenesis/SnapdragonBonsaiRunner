//! Source-bound pre-bootstrap owner budgets and credential-free X2 supervision.
#![cfg_attr(
    not(any(windows, test)),
    expect(
        dead_code,
        reason = "Pure owner-clock controls are consumed by Windows execution and platform-independent fixtures."
    )
)]

use crate::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;

#[cfg(windows)]
mod windows;

const BUDGET: u64 = 300;
const RETENTION_RESERVE: u64 = 70;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Start {
    schema_version: u64,
    kind: String,
    run_id: String,
    runner_revision: String,
    helper_sha256: String,
    clock_frequency: u64,
    started_tick: u64,
    deadline_tick: u64,
    started_unix_milliseconds: u64,
    deadline_unix_milliseconds: u64,
    budget_seconds: u64,
    conservative_estimate_seconds: u64,
    fixed_work: String,
    power_before: Value,
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
#[cfg_attr(
    not(windows),
    expect(
        dead_code,
        reason = "Owner execution consumes these request fields only on native Windows."
    )
)]
enum Request {
    #[serde(rename = "ci-owner-start", rename_all = "camelCase")]
    Create {
        output_path: PathBuf,
        run_id: String,
        runner_revision: String,
    },
    #[serde(rename = "ci-owner-admit", rename_all = "camelCase")]
    Admit { start_receipt: PathBuf },
    #[serde(rename = "ci-capture", rename_all = "camelCase")]
    Capture {
        start_receipt: PathBuf,
        data_directory: PathBuf,
        cim_processors: Value,
        observe_output_progress: Option<bool>,
        attribution_input: Option<String>,
    },
    #[serde(rename = "ci-owner-finalize", rename_all = "camelCase")]
    Finalize { start_receipt: PathBuf },
}
fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn decimal(value: &str) -> bool {
    !value.is_empty() && !value.starts_with('0') && value.bytes().all(|byte| byte.is_ascii_digit())
}
fn attribution(observed: Option<bool>, input: Option<&str>) -> Result<bool> {
    match (observed, input) {
        (Some(value), None) => Ok(value),
        (None, Some("true")) => Ok(true),
        (None, Some("false") | None) => Ok(false),
        _ => Err("Attribution requires one exact native boolean or true/false CI input.".into()),
    }
}
fn validate(
    start: &Start,
    run: &str,
    revision: &str,
    now: u64,
    frequency: u64,
    unix: u64,
) -> Result<()> {
    require(
        start.schema_version == 1
            && start.kind == "runner-x2-cpu-owner-start"
            && decimal(&start.run_id)
            && start.run_id == run
            && crate::identity::is_lower_hex(&start.runner_revision, 40)
            && start.runner_revision == revision
            && crate::identity::is_lower_hex(&start.helper_sha256, 64)
            && start.budget_seconds == BUDGET
            && start.conservative_estimate_seconds == BUDGET
            && !start.fixed_work.trim().is_empty(),
        "Original X2 allowance/source/run identity missing or foreign.",
    )?;
    require(
        frequency > 0
            && frequency <= 1_000_000_000_000
            && start.clock_frequency == frequency
            && start.started_tick < now
            && u128::from(start.deadline_tick)
                == u128::from(start.started_tick) + u128::from(BUDGET) * u128::from(frequency)
            && u128::from(start.deadline_unix_milliseconds)
                == u128::from(start.started_unix_milliseconds) + u128::from(BUDGET) * 1000,
        "Original X2 monotonic/UTC anchors malformed.",
    )?;
    let elapsed = u128::from(now - start.started_tick) * 1000 / u128::from(frequency);
    require(
        (i128::from(unix)
            - i128::from(start.started_unix_milliseconds)
            - i128::try_from(elapsed).map_err(|error| error.to_string())?)
        .abs()
            <= 2000,
        "Original X2 monotonic and UTC clocks disagree.",
    )
}
fn work(start: &Start, now: u64) -> Result<(u64, u64)> {
    let close = start
        .deadline_tick
        .checked_sub(RETENTION_RESERVE * start.clock_frequency)
        .ok_or("Invalid X2 retention reserve.")?;
    let seconds = (close.saturating_sub(now) / start.clock_frequency).min(240);
    require(
        seconds >= super::capture::MINIMUM_COMPLETE_WORK_SECONDS,
        "The complete fixed X2 screen cannot fit the original allowance and bounded upload; no model work may start.",
    )?;
    Ok((seconds, now + seconds * start.clock_frequency))
}

pub(super) fn run(value: Value) -> Result<Value> {
    let request: Request = serde_json::from_value(value).map_err(|error| error.to_string())?;
    #[cfg(windows)]
    {
        windows::run(request)
    }
    #[cfg(not(windows))]
    {
        let _ = request;
        Err("X2 owner capture requires native Windows ARM64.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn start() -> Start {
        Start {
            schema_version: 1,
            kind: "runner-x2-cpu-owner-start".into(),
            run_id: "123".into(),
            runner_revision: "a".repeat(40),
            helper_sha256: "b".repeat(64),
            clock_frequency: 1000,
            started_tick: 100_000,
            deadline_tick: 400_000,
            started_unix_milliseconds: 2_000_000_000_000,
            deadline_unix_milliseconds: 2_000_000_300_000,
            budget_seconds: 300,
            conservative_estimate_seconds: 300,
            fixed_work: "One fixed four-sample CPU screen, setup, cleanup and upload; no retry."
                .into(),
            power_before: json!({"status":"unknown","source":"GetSystemPowerStatus","ac_online":null,"battery_saver":null,"battery_percent":null}),
        }
    }
    #[test]
    fn original_owner_clocks_include_setup_and_reserve_full_bounded_retention() -> Result<()> {
        let mut receipt = start();
        validate(
            &receipt,
            "123",
            &"a".repeat(40),
            101_000,
            1000,
            2_000_000_001_000,
        )?;
        require(
            work(&receipt, 101_000)? == (229, 330_000),
            "Setup was not charged to the same original close.",
        )?;
        require(
            work(&receipt, 110_000).is_err(),
            "A fixed screen that cannot fit was admitted.",
        )?;
        receipt.started_tick += 1;
        require(
            validate(
                &receipt,
                "123",
                &"a".repeat(40),
                101_000,
                1000,
                2_000_000_001_000,
            )
            .is_err(),
            "Restarted owner clock admitted.",
        )
    }
    #[test]
    fn foreign_identity_clock_changes_and_unknown_power_never_qualify() -> Result<()> {
        let receipt = start();
        for (run, revision, frequency, unix) in [
            ("124", "a".repeat(40), 1000, 2_000_000_001_000),
            ("123", "c".repeat(40), 1000, 2_000_000_001_000),
            ("123", "a".repeat(40), 1001, 2_000_000_001_000),
            ("123", "a".repeat(40), 1000, 2_000_000_004_001),
        ] {
            require(
                validate(&receipt, run, &revision, 101_000, frequency, unix).is_err(),
                "Foreign/changed original anchor admitted.",
            )?;
        }
        require(
            attribution(None, Some("true"))?
                && !attribution(Some(false), None)?
                && attribution(None, Some("True")).is_err()
                && attribution(Some(false), Some("true")).is_err(),
            "Ambiguous attribution input admitted.",
        )?;
        require(
            super::super::bundle::power(&receipt.power_before).is_err(),
            "Unknown setup power interpreted as observed.",
        )
    }
}
