//! Fail-closed aggregation of the selected GitHub Actions jobs.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde_json::Value;

/// Return each failed selected job, rejecting incomplete or malformed results.
///
/// Only a successful job or a deliberately unselected skipped job can pass.
#[must_use]
pub fn failures(needs: &Value, jobs: &Value) -> Vec<String> {
    if needs["changes"]["result"] != "success" {
        let actual = needs["changes"]["result"].as_str().unwrap_or("missing");
        return vec![format!(
            "CI impact selection did not succeed (changes.result={actual})"
        )];
    }
    checked_failures(needs, jobs)
        .unwrap_or_else(|| vec!["Missing or malformed CI selection/job result".into()])
}

/// Format a Markdown step-summary table describing planner and job outcomes.
#[must_use]
pub fn summary(workflow: &str, needs: &Value, jobs: &Value) -> String {
    let mut out = format!("### CI Gate (`{workflow}`)\n\n");
    out.push_str("| Job | Group | Selected | Result | Verdict |\n");
    out.push_str("| --- | --- | --- | --- | --- |\n");
    let planner_result = needs["changes"]["result"].as_str().unwrap_or("missing");
    let planner_verdict = if planner_result == "success" {
        "PASS"
    } else {
        "FAIL"
    };
    let _ = writeln!(
        out,
        "| `changes` | `(planner)` | `true` | `{planner_result}` | `{planner_verdict}` |"
    );
    let selected = needs
        .get("changes")
        .and_then(|changes| changes.get("outputs"))
        .and_then(|outputs| outputs.get("plan"))
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|plan| plan.get("groups").cloned());
    if let Some(jobs_map) = jobs.as_object() {
        for (job, group_val) in jobs_map {
            let group = group_val.as_str().unwrap_or("invalid");
            let result = needs
                .get(job)
                .and_then(|entry| entry.get("result"))
                .and_then(Value::as_str)
                .unwrap_or("missing");
            let is_selected = selected
                .as_ref()
                .and_then(|groups| groups.get(group))
                .and_then(Value::as_bool);
            let selected_label = match is_selected {
                Some(true) => "true",
                Some(false) => "false",
                None => "unknown",
            };
            let verdict = match (is_selected, result) {
                (Some(_), "success") => "PASS",
                (Some(false), "skipped") => "SKIP",
                _ => "FAIL",
            };
            let _ = writeln!(
                out,
                "| `{job}` | `{group}` | `{selected_label}` | `{result}` | `{verdict}` |"
            );
        }
    }
    out
}

fn checked_failures(needs: &Value, jobs: &Value) -> Option<Vec<String>> {
    let needs = needs.as_object()?;
    let jobs = jobs.as_object()?;
    let actual: BTreeSet<_> = needs.keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = jobs.keys().map(String::as_str).chain(["changes"]).collect();
    if actual != expected {
        return None;
    }
    let plan: Value =
        serde_json::from_str(needs.get("changes")?["outputs"]["plan"].as_str()?).ok()?;
    let selected = plan.get("groups")?.as_object()?;
    let mut errors = Vec::new();
    for (job, group) in jobs {
        let group_name = group.as_str()?;
        let result = needs.get(job)?.get("result")?.as_str()?;
        let expected = selected.get(group_name)?.as_bool()?;
        if result != "success" && (expected || result != "skipped") {
            let expected = if expected { "True" } else { "False" };
            errors.push(format!(
                "{job} (group={group_name}): selected={expected}, result={result}"
            ));
        }
    }
    Some(errors)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{failures, summary};

    #[test]
    fn gate_summary_renders_selected_skipped_and_failed_verdicts() {
        let needs = json!({
            "changes": {
                "result": "success",
                "outputs": {
                    "plan": json!({"groups": {"core": true, "tts": false}}).to_string()
                }
            },
            "broker": {"result": "failure"},
            "speech-tts": {"result": "skipped"}
        });
        let jobs = json!({"broker": "core", "speech-tts": "tts"});
        let rendered = summary("foundation", &needs, &jobs);
        assert!(rendered.contains("| `changes` | `(planner)` | `true` | `success` | `PASS` |"));
        assert!(rendered.contains("| `broker` | `core` | `true` | `failure` | `FAIL` |"));
        assert!(rendered.contains("| `speech-tts` | `tts` | `false` | `skipped` | `SKIP` |"));
        let errs = failures(&needs, &jobs);
        assert_eq!(errs, ["broker (group=core): selected=True, result=failure"]);
    }
}
