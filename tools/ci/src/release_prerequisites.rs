//! Exact-source prerequisites for production-only release builds.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::ffi::OsString;

use crate::Result;

const REPOSITORY: &str = "symbiogenesis/SnapdragonBonsaiRunner";
const USAGE: &str = "Usage: bonsai-ci release-prerequisites --revision SHA";

#[derive(Deserialize)]
struct Repository {
    full_name: String,
}

#[derive(Deserialize)]
struct Run {
    id: u64,
    run_attempt: u64,
    path: String,
    head_sha: String,
    head_branch: String,
    event: String,
    status: String,
    conclusion: Option<String>,
    repository: Repository,
    head_repository: Repository,
}

#[derive(Deserialize)]
struct Runs {
    total_count: usize,
    workflow_runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Step {
    name: String,
    status: String,
    conclusion: Option<String>,
}

#[derive(Deserialize)]
struct Job {
    id: u64,
    run_id: u64,
    run_attempt: u64,
    head_sha: String,
    name: String,
    status: String,
    conclusion: Option<String>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Jobs {
    total_count: usize,
    jobs: Vec<Job>,
}

#[cfg(feature = "github-api")]
fn fetch(agent: &ureq::Agent, token: &str, route: &str) -> Result<Value> {
    super::github_get_json(
        agent,
        &format!("https://api.github.com/repos/{REPOSITORY}/{route}"),
        token,
        "bonsai-release-prerequisites",
        "Release prerequisite API unavailable; no release builds started.",
        "Release prerequisite response unavailable.",
        "Malformed release prerequisite response.",
    )
}

fn latest_run(value: Value, workflow: &str, revision: &str) -> Result<Run> {
    let response: Runs = serde_json::from_value(value).map_err(|error| error.to_string())?;
    // The API filters exact head_sha/main/push and orders newest first. Inspect
    // the latest attempt rather than searching backwards for any older success.
    let run = response
        .workflow_runs
        .into_iter()
        .next()
        .filter(|_| response.total_count > 0)
        .ok_or_else(|| format!("{workflow}: no main-push run for exact source {revision}."))?;
    if run.id == 0
        || run.run_attempt == 0
        || run.path != format!(".github/workflows/{workflow}")
        || run.head_sha != revision
        || run.head_branch != "main"
        || run.event != "push"
        || run.repository.full_name != REPOSITORY
        || run.head_repository.full_name != REPOSITORY
        || run.status != "completed"
        || run.conclusion.as_deref() != Some("success")
    {
        return Err(format!(
            "{workflow}: the latest exact-source main-push attempt must complete successfully before Release; failed, pending, cancelled and missing checks never qualify."
        ));
    }
    Ok(run)
}

fn require_job(jobs: &[Job], name: &str, required_step: &str) -> Result<()> {
    let matches: Vec<_> = jobs.iter().filter(|job| job.name == name).collect();
    let [job] = matches.as_slice() else {
        return Err(format!(
            "Missing or duplicate required prerequisite job: {name}."
        ));
    };
    let steps: Vec<_> = job
        .steps
        .iter()
        .filter(|step| step.name == required_step)
        .collect();
    if job.status != "completed"
        || job.conclusion.as_deref() != Some("success")
        || steps.len() != 1
        || steps[0].status != "completed"
        || steps[0].conclusion.as_deref() != Some("success")
    {
        return Err(format!(
            "Required prerequisite job {name} and its {required_step} step must succeed; skipped or bundle-only work cannot qualify."
        ));
    }
    Ok(())
}

fn check_workflow(
    workflow: &str,
    revision: &str,
    fetch: &mut impl FnMut(&str) -> Result<Value>,
) -> Result<String> {
    let route = format!(
        "actions/workflows/{workflow}/runs?branch=main&event=push&head_sha={revision}&per_page=1"
    );
    let run = latest_run(fetch(&route)?, workflow, revision)?;
    let mut jobs = Vec::new();
    let mut identities = BTreeSet::new();
    for page in 1..=10 {
        let route = format!(
            "actions/runs/{}/attempts/{}/jobs?per_page=100&page={page}",
            run.id, run.run_attempt
        );
        let response: Jobs =
            serde_json::from_value(fetch(&route)?).map_err(|error| error.to_string())?;
        if response.jobs.is_empty() || response.total_count > 1000 {
            return Err("Incomplete or oversized prerequisite job inventory.".into());
        }
        for job in response.jobs {
            if job.id == 0
                || !identities.insert(job.id)
                || job.run_id != run.id
                || job.run_attempt != run.run_attempt
                || job.head_sha != revision
                || job.status != "completed"
                || !matches!(job.conclusion.as_deref(), Some("success" | "skipped"))
            {
                return Err(
                    "Prerequisite job identity or conclusion differs from the successful attempt."
                        .into(),
                );
            }
            jobs.push(job);
        }
        if jobs.len() == response.total_count {
            match workflow {
                "foundation.yml" => {
                    require_job(&jobs, "changes", "Select affected crates and app groups")?;
                    require_job(
                        &jobs,
                        "foundation",
                        "Require every selected Foundation test group",
                    )?;
                }
                "source-checks.yml" => require_job(
                    &jobs,
                    "actions-lock",
                    "Verify every workflow lock and upstream repository identity",
                )?,
                _ => return Err("Unknown release prerequisite workflow.".into()),
            }
            // A rerun can start while its previous attempt's jobs are read.
            // Recheck the newest attempt before accepting the collected jobs.
            let current = latest_run(
                fetch(&format!(
                    "actions/workflows/{workflow}/runs?branch=main&event=push&head_sha={revision}&per_page=1"
                ))?,
                workflow,
                revision,
            )?;
            if current.id != run.id || current.run_attempt != run.run_attempt {
                return Err("Prerequisite attempt changed during validation; Release must be started after checks settle.".into());
            }
            return Ok(format!(
                "{workflow}: successful exact-source run {}, attempt {}",
                run.id, run.run_attempt
            ));
        }
        if jobs.len() > response.total_count {
            return Err("Inconsistent prerequisite job pagination.".into());
        }
    }
    Err("Prerequisite job inventory exceeds supported pagination.".into())
}

pub(super) fn run(arguments: &[OsString]) -> Result<String> {
    let mut cursor = super::ArgCursor::new(arguments, USAGE);
    cursor.skip_command()?;
    let revision = cursor.expect_value("--revision")?;
    cursor.finish()?;
    if !crate::impact::workspace::is_lower_hex(revision, 40)
        || std::env::var("GITHUB_REPOSITORY").as_deref() != Ok(REPOSITORY)
    {
        return Err(
            "Release prerequisites require this repository and an exact lowercase source SHA."
                .into(),
        );
    }
    let token = std::env::var("GITHUB_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
        .ok_or("Release prerequisite lookup requires an Actions read token.")?;
    check_workflows_with_api(revision, &token)
}

#[cfg(feature = "github-api")]
fn check_workflows_with_api(revision: &str, token: &str) -> Result<String> {
    let agent = super::github_agent();
    let mut fetch = |route: &str| fetch(&agent, token, route);
    ["foundation.yml", "source-checks.yml"]
        .into_iter()
        .map(|workflow| check_workflow(workflow, revision, &mut fetch))
        .collect::<Result<Vec<_>>>()
        .map(|summaries| summaries.join("\n"))
}

#[cfg(not(feature = "github-api"))]
fn check_workflows_with_api(revision: &str, token: &str) -> Result<String> {
    let _ = token;
    let mut fetch = |_: &str| -> Result<Value> {
        Err("GitHub API support is disabled in this check-only configuration.".into())
    };
    ["foundation.yml", "source-checks.yml"]
        .into_iter()
        .map(|workflow| check_workflow(workflow, revision, &mut fetch))
        .collect::<Result<Vec<_>>>()
        .map(|summaries| summaries.join("\n"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn runs(workflow: &str) -> Value {
        json!({"total_count":1,"workflow_runs":[{"id":1,"run_attempt":2,
            "path":format!(".github/workflows/{workflow}"),"head_sha":"a".repeat(40),
            "head_branch":"main","event":"push","status":"completed","conclusion":"success",
            "repository":{"full_name":REPOSITORY},"head_repository":{"full_name":REPOSITORY}}]})
    }

    fn job(id: u64, name: &str, step: &str) -> Value {
        json!({"id":id,"run_id":1,"run_attempt":2,"head_sha":"a".repeat(40),
            "name":name,"status":"completed","conclusion":"success",
            "steps":[{"name":step,"status":"completed","conclusion":"success"}]})
    }

    fn jobs() -> Value {
        json!({"total_count":2,"jobs":[
            job(1,"changes","Select affected crates and app groups"),
            job(2,"foundation","Require every selected Foundation test group")]})
    }

    #[test]
    fn release_prerequisites_reject_stale_wrong_and_unsuccessful_runs() -> Result<()> {
        for (key, value) in [
            ("head_sha", json!("b".repeat(40))),
            ("head_branch", json!("feature")),
            ("event", json!("workflow_dispatch")),
            ("status", json!("in_progress")),
            ("conclusion", json!("failure")),
            ("conclusion", json!("cancelled")),
            ("conclusion", json!(null)),
            ("path", json!(".github/workflows/other.yml")),
            ("run_attempt", json!(0)),
            ("id", json!(0)),
            ("head_repository", json!({"full_name":"other/repository"})),
        ] {
            let mut response = runs("foundation.yml");
            response["workflow_runs"][0][key] = value;
            assert!(latest_run(response, "foundation.yml", &"a".repeat(40)).is_err());
        }
        let mut response = runs("foundation.yml");
        let older_success = response["workflow_runs"][0].clone();
        response["workflow_runs"][0]["conclusion"] = json!("failure");
        response["workflow_runs"]
            .as_array_mut()
            .ok_or("fixture runs")?
            .push(older_success);
        assert!(latest_run(response, "foundation.yml", &"a".repeat(40)).is_err());
        assert!(
            latest_run(
                json!({"total_count":0,"workflow_runs":[]}),
                "foundation.yml",
                &"a".repeat(40)
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn release_prerequisites_require_complete_gate_and_attempt_bound_jobs() -> Result<()> {
        let revision = "a".repeat(40);
        let check = |response: Value| {
            check_workflow("foundation.yml", &revision, &mut |route| {
                Ok(if route.contains("/workflows/") {
                    runs("foundation.yml")
                } else {
                    response.clone()
                })
            })
        };
        check(jobs())?;
        for (key, value) in [
            ("run_attempt", json!(1)),
            ("run_id", json!(2)),
            ("head_sha", json!("b".repeat(40))),
            ("conclusion", json!("failure")),
            ("conclusion", json!("skipped")),
            ("name", json!("bundle-only")),
            ("steps", json!([])),
        ] {
            let mut response = jobs();
            response["jobs"][1][key] = value;
            assert!(check(response).is_err());
        }
        let mut response = jobs();
        response["jobs"][1]["steps"][0]["conclusion"] = json!("skipped");
        assert!(check(response).is_err());
        assert!(check(json!({"total_count":0,"jobs":[]})).is_err());
        Ok(())
    }

    #[test]
    fn release_prerequisites_paginate_jobs_and_allow_deliberately_unselected_source_checks()
    -> Result<()> {
        let mut routes = Vec::new();
        check_workflow("foundation.yml", &"a".repeat(40), &mut |route| {
            routes.push(route.to_owned());
            Ok(if route.contains("/workflows/") {
                runs("foundation.yml")
            } else if route.ends_with("page=1") {
                json!({"total_count":2,"jobs":[job(1,"changes","Select affected crates and app groups")]})
            } else {
                json!({"total_count":2,"jobs":[job(2,"foundation","Require every selected Foundation test group")]})
            })
        })?;
        assert_eq!(routes.len(), 4);
        assert!(routes[0].contains("head_sha="));
        assert!(routes[1].contains("/attempts/2/"));
        let mut skipped = job(4, "audit", "Audit locked packages");
        skipped["conclusion"] = json!("skipped");
        check_workflow("source-checks.yml", &"a".repeat(40), &mut |route| {
            Ok(if route.contains("/workflows/") {
                runs("source-checks.yml")
            } else {
                json!({"total_count":2,"jobs":[job(3,"actions-lock","Verify every workflow lock and upstream repository identity"),skipped.clone()]})
            })
        })?;
        Ok(())
    }

    #[test]
    fn release_prerequisites_reject_an_attempt_started_during_job_intake() {
        let mut requests = 0;
        let result = check_workflow("foundation.yml", &"a".repeat(40), &mut |route| {
            Ok(if route.contains("/workflows/") {
                requests += 1;
                let mut response = runs("foundation.yml");
                if requests > 1 {
                    response["workflow_runs"][0]["run_attempt"] = json!(3);
                    response["workflow_runs"][0]["status"] = json!("in_progress");
                    response["workflow_runs"][0]["conclusion"] = json!(null);
                }
                response
            } else {
                jobs()
            })
        });
        assert!(result.is_err());
        assert_eq!(requests, 2);
    }
}
