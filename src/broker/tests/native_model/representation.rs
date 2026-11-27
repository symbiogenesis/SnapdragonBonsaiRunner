//! A bounded complete-response feasibility pair, independent of decode-TPS gates.
#![expect(
    clippy::redundant_pub_crate,
    reason = "The private test module shares profile constants only with its parent harness."
)]

use super::*;

pub(super) const PROFILE: &str = "ptq1-pq2-complete-response";
pub(super) const CASE: &str = "representation::pinned_native_ptq1_pq2_capture_complete_responses";
pub(super) const ROW4_PROFILE: &str = "ptq1-pq2-complete-response-row4";
pub(super) const ROW4_CASE: &str =
    "representation::pinned_native_ptq1_pq2_row4_capture_complete_responses";
const CAP: u32 = 32;
const PROMPT: &str = "Name one tree. Answer with only its name.";
const NEXT_OWNER_RESERVE: Duration = Duration::from_mins(1);
fn profiles(row_workers: u8) -> [AdapterProfile; 2] {
    assert!([1, 4].contains(&row_workers));
    let (name, case) = if row_workers == 1 {
        (PROFILE, CASE)
    } else {
        (ROW4_PROFILE, ROW4_CASE)
    };
    [false, true].map(|repacked| AdapterProfile {
        name,
        case,
        repacked,
        row_workers,
        placement: AdapterPlacement::Cpu,
        optional_requests: false,
        required_scale_history: false,
    })
}

pub(super) fn body() -> Value {
    prompt_request(PROMPT, 1.3, CAP)
}

pub(super) fn model_identity(repacked: bool) -> Value {
    if repacked {
        let export = ptq1_export_lock();
        json!({"filename":export["OutputFilename"], "bytes":export["OutputBytes"],
            "sha256":export["OutputSha256"], "weightRepresentation":"current-repacked-PTQ1_0"})
    } else {
        let mut identity = model_lock()["model"].clone();
        identity["weightRepresentation"] = json!("PQ2_0");
        identity
    }
}

struct CapturedResponse {
    bytes: Vec<u8>,
    elapsed: Duration,
}

async fn measured_response(
    port: u16,
    credential: &str,
    evidence: &Path,
    report: &mut Value,
) -> Option<CapturedResponse> {
    // Parsing waits for owned cleanup. This timer includes
    // prefill, generation, transport and incremental observer/receipt I/O.
    let started = tokio::time::Instant::now();
    let bytes = progress::try_exchange(
        port,
        credential,
        &body(),
        evidence,
        report,
        "representation-complete-response",
    )
    .await?;
    let elapsed = started.elapsed();
    Some(CapturedResponse { bytes, elapsed })
}

pub(super) fn response_observation(bytes: &[u8], elapsed: Duration) -> Value {
    let parsed = parse_stream_response(bytes);
    let mut observed = stream_observation(&parsed.summary, CAP);
    observed["requestMilliseconds"] = json!(elapsed_milliseconds(elapsed));
    observed["visibleContentSha256"] = json!(adapter_digest(parsed.visible_content.as_bytes()));
    observed["visibleContent"] = json!(parsed.visible_content);
    observed["naturallyStopped"] = json!(parsed.summary.finish_reason == "stop");
    observed
}

async fn record_response(
    port: u16,
    credential: &str,
    evidence: &Path,
    deadline: tokio::time::Instant,
    report: &mut Value,
    index: usize,
) -> Option<CapturedResponse> {
    let observed = adapter_work_before_cleanup(
        deadline,
        measured_response(port, credential, evidence, report),
    )
    .await;
    report["samples"][index]["stream"] = Value::Null;
    let captured = match observed {
        Some(Some(captured)) => Some(captured),
        Some(None) => {
            report["pendingReason"] = json!("request-transport-failed");
            None
        }
        None => {
            report["pendingReason"] = json!("request-budget-expired");
            None
        }
    };
    if let Some(captured) = &captured {
        report["samples"][index]["responseBytes"] = json!(captured.bytes.len());
        report["samples"][index]["responseSha256"] = json!(adapter_digest(&captured.bytes));
        report["samples"][index]["responseParsePassed"] = json!(false);
    }
    save_acceptance_report(evidence, report);
    captured
}

fn finish_response_after_cleanup(
    captured: Option<&CapturedResponse>,
    evidence: &Path,
    report: &mut Value,
    index: usize,
) -> bool {
    let cleanup = &report["samples"][index]["cleanup"];
    let cleaned = service_integration::cleanup_passed(cleanup);
    let stream = cleaned
        .then(|| {
            captured.and_then(|captured| {
                service_rollback::response_after_cleanup(&captured.bytes, captured.elapsed, cleanup)
            })
        })
        .flatten();
    let parsed = stream.is_some();
    let response_failure = stream.as_ref().and_then(response_pending_reason);
    report["samples"][index]["responseParsePassed"] = json!(parsed);
    report["samples"][index]["stream"] = stream.unwrap_or(Value::Null);
    if !cleaned {
        report["pendingReason"] = json!("owned-cleanup-unconfirmed");
    } else if captured.is_some() && !parsed {
        report["pendingReason"] = json!("invalid-or-incomplete-response");
    } else if let Some(reason) = response_failure {
        report["pendingReason"] = json!(reason);
    }
    save_acceptance_report(evidence, report);
    cleaned && parsed && response_failure.is_none()
}

fn response_pending_reason(stream: &Value) -> Option<&'static str> {
    let content = stream["visibleContent"].as_str().unwrap_or("");
    (stream["finishReason"] != "stop"
        || stream["naturallyStopped"] != true
        || stream["reportedCompletionTokens"]
            .as_u64()
            .is_none_or(|n| !(1..=u64::from(CAP)).contains(&n))
        || content.trim().is_empty()
        || stream["visibleContentSha256"] != adapter_digest(content.as_bytes())
        || stream["requestMilliseconds"]
            .as_f64()
            .is_none_or(|ms| !ms.is_finite() || ms <= 0.0))
    .then_some("incomplete-or-invalid-complete-response")
}

fn sample_pending_reason(sample: &Value, repacked: bool, row_workers: u8) -> Option<&'static str> {
    if !matches!(row_workers, 1 | 4) {
        return Some("unsupported-worker-policy");
    }
    let runtime_lock: Value =
        serde_json::from_str(include_str!("../../../../config/prism-inference.lock.json")).unwrap();
    let worker_identity = worker_readiness_receipt(
        runtime_lock["version"].as_str(),
        runtime_lock["commit"].as_str(),
        row_workers,
        true,
    );
    let stream = &sample["stream"];
    if [
        "rowWorkerReadinessVersion",
        "rowWorkerReadinessRevision",
        "rowWorkerCountIndependentlyObserved",
        "observedNativeRowWorkers",
    ]
    .iter()
    .any(|field| sample[field] != worker_identity[field])
    {
        return Some("missing-or-mismatched-worker-readiness-identity");
    }
    if sample["modelIdentity"] != model_identity(repacked)
        || sample["configuredNativeRowWorkers"] != row_workers
        || sample["verifiedReadinessObserved"] != true
        || response_pending_reason(stream).is_some()
        || sample["cleanup"]["ownedJobEmpty"] != true
        || sample["cleanup"]["listenerReleased"] != true
    {
        return Some("missing-complete-response-identity-timing-or-cleanup");
    }
    None
}

pub(super) fn original_pq2_rollback_observation(samples: &[Value], row_workers: u8) -> Value {
    // This observation concerns only the original-artifact arm. A later PTQ1
    // failure cannot erase it or turn it into complete-pair/performance evidence.
    let reason = samples
        .first()
        .map_or(Some("original-pq2-arm-not-started"), |sample| {
            sample_pending_reason(sample, false, row_workers)
        });
    json!({"status":if reason.is_none() {"captured-original-pq2-arm"} else {"pending"},
        "reason":reason, "sampleIndex":0, "configuredNativeRowWorkers":row_workers,
        "explicitRepresentationArgument":"--use-original-pq2",
        "gateId":"original-pq2-rollback", "independentGateReview":"pending",
        "scope":"Original PQ2 readiness, complete response and owned cleanup under the compiled release/adapter/admission policy; exact artifact retention and CI identities require independent review",
        "defaultMayChange":false})
}

fn comparison(samples: &[Value], row_workers: u8) -> Value {
    let pending = |reason| {
        json!({"status":"pending", "reason":reason,
        "ptq1OverPq2RequestLatency":null,"performanceComparisonEligible":false})
    };
    if !matches!(row_workers, 1 | 4) || samples.len() != 2 {
        return pending("incomplete-pair");
    }
    for (index, sample) in samples.iter().enumerate() {
        if let Some(reason) = sample_pending_reason(sample, index == 1, row_workers) {
            return pending(reason);
        }
    }
    // Different complete responses remain available for independent review.
    // A raw latency ratio with different output work would be misleading.
    let matched = samples[0]["stream"]["visibleContent"] == samples[1]["stream"]["visibleContent"]
        && samples[0]["stream"]["reportedCompletionTokens"]
            == samples[1]["stream"]["reportedCompletionTokens"];
    let power_conditions = bonsai_cpu_policy::power::compare_samples(samples);
    let power_matched = power_conditions["performanceComparisonEligible"] == true;
    let cpu_placement_matched = samples.iter().all(|sample| {
        sample["configuredPlacement"] == "cpu"
            && sample["cpuPlacementIndependentlyVerified"] == true
            && sample["verifiedProjectionBackend"] == "cpu"
    });
    let state_policy_matched = samples.iter().all(state_policy_matches);
    let eligible = matched && power_matched && cpu_placement_matched && state_policy_matched;
    let observed_ratio = matched.then(|| {
        samples[1]["stream"]["requestMilliseconds"]
            .as_f64()
            .unwrap()
            / samples[0]["stream"]["requestMilliseconds"]
                .as_f64()
                .unwrap()
    });
    json!({"status":"captured-complete-pair", "matchedOutputAndUsage":matched,
        "comparisonKind":"cpu-only-packing", "cpuPlacementMatched":cpu_placement_matched,
        "configuredKvCache":"rsq4", "configuredPromptReuse":"disabled",
        "statePolicyMatched":state_policy_matched,
        "ptq1OverPq2RequestLatency":eligible.then_some(observed_ratio).flatten(),
        "observedUnqualifiedRequestLatencyRatio":observed_ratio,
        "performanceComparisonEligible":eligible,"powerConditions":power_conditions,
        "resources":resource_comparison(samples, matched, power_matched, cpu_placement_matched),
        "fixedOrder":true, "qualityReview":"pending", "repeatedTrialBenefitQualified":false})
}

const RESOURCE_METRICS: [(&str, &str); 4] = [
    (
        "osLifetimePeakJobCommitBytes",
        "OS lifetime peak commitment for the complete owned Job, including descendants and startup",
    ),
    (
        "osLifetimePeakRootWorkingSetBytes",
        "OS lifetime peak root-process residency, including mapped pages and startup, excluding descendants",
    ),
    (
        "sampledPeakRootPrivateBytes",
        "maximum root-process private commitment observed by serving guard samples; startup is excluded",
    ),
    (
        "sampledPeakRootWorkingSetBytes",
        "maximum root-process residency observed by serving guard samples; startup and descendants are excluded",
    ),
];

fn resource_comparison(
    samples: &[Value],
    matched: bool,
    power_matched: bool,
    cpu_placement_matched: bool,
) -> Value {
    let pending = |reason| {
        json!({"status":"pending", "reason":reason, "metrics":[],
            "fullResourceGateQualified":false,"pagingQualified":false,
            "sustainedComfortQualified":false,"resourceBenefitQualified":false})
    };
    if samples.len() != 2 {
        return pending("incomplete-pair");
    }
    if !matched {
        return pending("different-complete-output-or-usage");
    }
    if !power_matched {
        return pending("unknown-or-mismatched-power-conditions");
    }
    if !cpu_placement_matched {
        return pending("missing-or-mismatched-cpu-placement-readiness");
    }
    if !samples.iter().all(state_policy_matches) {
        return pending("missing-or-mismatched-state-policy-readiness");
    }
    let resources = [
        &samples[0]["cleanup"]["backendResources"],
        &samples[1]["cleanup"]["backendResources"],
    ];
    for resource in &resources {
        if resource["successfulGuardSamples"]
            .as_u64()
            .is_none_or(|n| n == 0)
            || resource["guardIntervalMilliseconds"] != 250
            || resource["sampledMinimumSystemAvailablePhysicalBytes"]
                .as_u64()
                .is_none_or(|n| n == 0)
            || RESOURCE_METRICS
                .iter()
                .any(|(field, _)| resource[*field].as_u64().is_none_or(|n| n == 0))
        {
            return pending("missing-positive-resource-observations");
        }
        if resource["sampledPeakRootPrivateBytes"].as_u64().unwrap()
            > resource["osLifetimePeakJobCommitBytes"].as_u64().unwrap()
            || resource["sampledPeakRootWorkingSetBytes"].as_u64().unwrap()
                > resource["osLifetimePeakRootWorkingSetBytes"]
                    .as_u64()
                    .unwrap()
        {
            return pending("inconsistent-sampled-and-lifetime-resource-observations");
        }
    }
    let metrics: Vec<_> = RESOURCE_METRICS
        .iter()
        .map(|(field, scope)| {
            json!({"field":field,"scope":scope,"pq2Bytes":resources[0][*field],
            "ptq1Bytes":resources[1][*field],
            "ptq1OverPq2":resources[1][*field].as_f64().unwrap()
                / resources[0][*field].as_f64().unwrap()})
        })
        .collect();
    let serving_observations: Vec<_> = resources.iter().map(|resource| {
        json!({"successfulGuardSamples":resource["successfulGuardSamples"],
            "guardIntervalMilliseconds":resource["guardIntervalMilliseconds"],
            "sampledMinimumSystemAvailablePhysicalBytes":resource["sampledMinimumSystemAvailablePhysicalBytes"]})
    }).collect();
    json!({"status":"observed-bounded-pair", "metrics":metrics,
        "servingObservationsInRepresentationOrder":serving_observations,
        "scope":"One fixed-order matched complete-answer pair; commitment and residency are non-additive; sampled bounds are not lifetime peaks; system available memory is context, not attributed model consumption; no full-context, paging or sustained-use evidence",
        "fullResourceGateQualified":false,"pagingQualified":false,
        "sustainedComfortQualified":false,"resourceBenefitQualified":false})
}

async fn paired_responses(
    data: &Path,
    evidence: &Path,
    deadline: tokio::time::Instant,
    row_workers: u8,
    report: &mut Value,
) {
    power::begin(report);
    for profile in &profiles(row_workers) {
        // Decide before each startup, charging the first owner and all prior
        // setup against the same deadline. Missing arms may not become jobs.
        if !can_start_owner(deadline) {
            report["pendingReason"] = json!("insufficient-next-owner-and-cleanup-reserve");
            break;
        }
        let index = report["samples"].as_array().unwrap().len();
        report["samples"].as_array_mut().unwrap().push(json!({
            "modelIdentity":model_identity(profile.repacked), "configuredNativeRowWorkers":profile.row_workers,
            "rowWorkerCountIndependentlyObserved":false, "verifiedReadinessObserved":false,
            "startupMilliseconds":null, "stream":null, "cleanup":null}));
        record_worker_readiness(&mut report["samples"][index], profile.row_workers, false);
        record_cpu_placement(&mut report["samples"][index], false);
        power::begin(&mut report["samples"][index]);
        report["phase"] = json!(format!("arm-{index}-startup"));
        save_acceptance_report(evidence, report);
        let started = tokio::time::Instant::now();
        let startup = adapter_work_before_cleanup(
            deadline,
            start_adapter_with_progress(profile, data, evidence, |_| {}),
        )
        .await;
        let owner = match startup {
            Some(Ok(owner)) => owner,
            result => {
                report["pendingReason"] = json!(if result.is_some() {
                    "startup-failed"
                } else {
                    "startup-budget-expired"
                });
                power::finish(&mut report["samples"][index]);
                save_acceptance_report(evidence, report);
                break;
            }
        };
        report["samples"][index]["startupMilliseconds"] =
            json!(elapsed_milliseconds(started.elapsed()));
        record_worker_readiness(&mut report["samples"][index], profile.row_workers, true);
        record_cpu_placement(&mut report["samples"][index], true);
        let port = owner.port();
        let credential = key(&owner);
        let (stop, stopped) = oneshot::channel();
        let running = spawn_local(owner.run(async { stopped.await.map_err(io::Error::other) }));
        report["phase"] = json!(format!("arm-{index}-request"));
        save_acceptance_report(evidence, report);
        let captured = record_response(port, &credential, evidence, deadline, report, index).await;
        report["phase"] = json!(format!("arm-{index}-cleanup"));
        save_acceptance_report(evidence, report);
        let (cleanup, _) =
            service_integration::stop_owner(stop, running, port, true, deadline).await;
        report["samples"][index]["cleanup"] = cleanup;
        save_acceptance_report(evidence, report);
        let completed = finish_response_after_cleanup(captured.as_ref(), evidence, report, index);
        power::finish(&mut report["samples"][index]);
        report["originalPq2Rollback"] =
            original_pq2_rollback_observation(report["samples"].as_array().unwrap(), row_workers);
        save_acceptance_report(evidence, report);
        if !completed {
            break;
        }
    }
    report["comparison"] = comparison(report["samples"].as_array().unwrap(), row_workers);
    report["originalPq2Rollback"] =
        original_pq2_rollback_observation(report["samples"].as_array().unwrap(), row_workers);
    report["passed"] = json!(report["comparison"]["status"] == "captured-complete-pair");
    report["phase"] = json!("completed");
    power::finish(report);
    save_acceptance_report(evidence, report);
}

fn can_start_owner(deadline: tokio::time::Instant) -> bool {
    bonsai_broker::deadline::remaining_until(deadline) > NEXT_OWNER_RESERVE
}

fn record_cpu_placement(sample: &mut Value, verified: bool) {
    // Both representation profiles explicitly request CPU. The source-bound
    // production launcher accepts ownership only after private readiness verifies
    // requested CPU placement, the effective CPU projection backend, the selected
    // rsq4 cache and disabled prefix reuse. Startup verification does not establish
    // executed operations, cache quality, numerical parity or performance.
    sample["configuredPlacement"] = json!("cpu");
    sample["cpuPlacementIndependentlyVerified"] = json!(verified);
    sample["verifiedProjectionBackend"] = json!(verified.then_some("cpu"));
    sample["configuredKvCache"] = json!("rsq4");
    sample["configuredPromptReuse"] = json!("disabled");
    sample["statePolicyIndependentlyVerified"] = json!(verified);
}

fn state_policy_matches(sample: &Value) -> bool {
    sample["configuredKvCache"] == "rsq4"
        && sample["configuredPromptReuse"] == "disabled"
        && sample["statePolicyIndependentlyVerified"] == true
}

#[test]
fn native_representation_pair_requires_completion_identity_and_cleanup() {
    for row_workers in [1, 4] {
        let pair = vec![
            complete_sample(false, row_workers),
            complete_sample(true, row_workers),
        ];
        assert_eq!(
            comparison(&pair, row_workers)["ptq1OverPq2RequestLatency"],
            1.0
        );
        for (path, value) in [
            ("/stream/finishReason", json!("length")),
            ("/stream/reportedCompletionTokens", Value::Null),
            ("/stream/reportedCompletionTokens", json!(0)),
            ("/stream/reportedCompletionTokens", json!(33)),
            ("/stream/visibleContentSha256", json!("0".repeat(64))),
            ("/stream/requestMilliseconds", json!(0)),
            ("/modelIdentity/sha256", json!("0".repeat(64))),
            (
                "/configuredNativeRowWorkers",
                json!(if row_workers == 1 { 4 } else { 1 }),
            ),
            ("/cleanup/ownedJobEmpty", json!(false)),
            ("/cleanup/listenerReleased", json!(false)),
        ] {
            let mut changed = pair.clone();
            *changed[1].pointer_mut(path).unwrap() = value;
            assert_eq!(
                comparison(&changed, row_workers)["status"],
                "pending",
                "{path}"
            );
        }
        let mut changed = pair.clone();
        changed[1]["stream"]["visibleContent"] = json!("Elm");
        changed[1]["stream"]["visibleContentSha256"] = json!(adapter_digest(b"Elm"));
        let result = comparison(&changed, row_workers);
        assert_eq!(result["status"], "captured-complete-pair");
        assert!(result["ptq1OverPq2RequestLatency"].is_null());
        assert_eq!(result["qualityReview"], "pending");
        assert_eq!(comparison(&pair[..1], row_workers)["status"], "pending");
        assert_eq!(
            comparison(&pair, if row_workers == 1 { 4 } else { 1 })["status"],
            "pending"
        );
        assert_eq!(
            comparison(&[pair[1].clone(), pair[0].clone()], row_workers)["status"],
            "pending"
        );
    }
}

pub(super) fn complete_sample(repacked: bool, row_workers: u8) -> Value {
    let power = bonsai_cpu_policy::power::observed_snapshot(Some(true), false, Some(80));
    let mut sample = json!({"modelIdentity":model_identity(repacked),
        "powerBefore":power,"powerAfter":power,
        "configuredNativeRowWorkers":row_workers,"verifiedReadinessObserved":true,
        "stream":{"finishReason":"stop","naturallyStopped":true,"reportedCompletionTokens":3,
            "visibleContent":"Oak","visibleContentSha256":adapter_digest(b"Oak"),"requestMilliseconds":1000},
        "cleanup":{"ownedJobEmpty":true,"listenerReleased":true}});
    sample["cleanup"]["backendResources"] = backend_resource_observation(Some(BackendResources {
        sample_count: 2,
        sampled_peak_root_private_bytes: 800,
        sampled_peak_root_working_set_bytes: 1600,
        sampled_min_available_physical_bytes: Some(4 * 1024 * 1024 * 1024),
        os_peak_job_commit_bytes: Some(1000),
        os_peak_root_working_set_bytes: Some(2000),
    }));
    record_worker_readiness(&mut sample, row_workers, true);
    record_cpu_placement(&mut sample, true);
    sample
}

#[test]
fn native_representation_power_gates_timing_without_erasing_answers_or_rollback() {
    let pair = vec![complete_sample(false, 4), complete_sample(true, 4)];
    let rollback = original_pq2_rollback_observation(&pair, 4);
    assert_eq!(comparison(&pair, 4)["performanceComparisonEligible"], true);
    for (sample, endpoint, observation) in [
        (0, "powerBefore", Value::Null),
        (
            1,
            "powerAfter",
            bonsai_cpu_policy::power::unknown_snapshot(),
        ),
        (
            0,
            "powerAfter",
            bonsai_cpu_policy::power::observed_snapshot(None, false, Some(80)),
        ),
        (
            1,
            "powerBefore",
            bonsai_cpu_policy::power::observed_snapshot(Some(false), false, Some(80)),
        ),
        (
            1,
            "powerAfter",
            bonsai_cpu_policy::power::observed_snapshot(Some(true), true, Some(80)),
        ),
    ] {
        let mut changed = pair.clone();
        changed[sample][endpoint] = observation;
        let compared = comparison(&changed, 4);
        assert_eq!(compared["status"], "captured-complete-pair");
        assert_eq!(compared["performanceComparisonEligible"], false);
        assert!(compared["ptq1OverPq2RequestLatency"].is_null());
        assert_eq!(compared["observedUnqualifiedRequestLatencyRatio"], 1.0);
        assert_eq!(compared["resources"]["status"], "pending");
        assert_eq!(
            compared["resources"]["reason"],
            "unknown-or-mismatched-power-conditions"
        );
        assert_eq!(compared["resources"]["metrics"], json!([]));
        assert_eq!(original_pq2_rollback_observation(&changed, 4), rollback);
    }
    let mut other_source = pair;
    let battery = bonsai_cpu_policy::power::observed_snapshot(Some(false), false, Some(79));
    other_source[1]["powerBefore"] = battery.clone();
    other_source[1]["powerAfter"] = battery;
    assert_eq!(
        comparison(&other_source, 4)["performanceComparisonEligible"],
        false
    );
}

#[test]
fn native_representation_cpu_placement_gates_comparisons_without_erasing_answers() {
    for row_workers in [1, 4] {
        let pair = vec![
            complete_sample(false, row_workers),
            complete_sample(true, row_workers),
        ];
        let matched = comparison(&pair, row_workers);
        assert_eq!(matched["comparisonKind"], "cpu-only-packing");
        assert_eq!(matched["cpuPlacementMatched"], true);
        assert_eq!(matched["configuredKvCache"], "rsq4");
        assert_eq!(matched["configuredPromptReuse"], "disabled");
        assert_eq!(matched["statePolicyMatched"], true);
        let rollback = original_pq2_rollback_observation(&pair, row_workers);
        for index in 0..2 {
            for (field, changed) in [
                ("configuredPlacement", json!("auto")),
                ("cpuPlacementIndependentlyVerified", json!(false)),
                ("verifiedProjectionBackend", json!("hexagon-native-q2")),
            ] {
                for value in [Value::Null, changed] {
                    let mut invalid = pair.clone();
                    invalid[index][field] = value;
                    let compared = comparison(&invalid, row_workers);
                    assert_eq!(compared["status"], "captured-complete-pair");
                    assert_eq!(compared["cpuPlacementMatched"], false);
                    assert_eq!(compared["performanceComparisonEligible"], false);
                    assert!(compared["ptq1OverPq2RequestLatency"].is_null());
                    assert_eq!(compared["observedUnqualifiedRequestLatencyRatio"], 1.0);
                    assert_eq!(compared["resources"]["status"], "pending");
                    assert_eq!(
                        compared["resources"]["reason"],
                        "missing-or-mismatched-cpu-placement-readiness"
                    );
                    assert_eq!(compared["resources"]["metrics"], json!([]));
                    assert_eq!(
                        original_pq2_rollback_observation(&invalid, row_workers),
                        rollback
                    );
                }
            }
            for (field, changed) in [
                ("configuredKvCache", json!("f16")),
                ("configuredPromptReuse", json!("latest")),
                ("statePolicyIndependentlyVerified", json!(false)),
            ] {
                for value in [Value::Null, changed] {
                    let mut invalid = pair.clone();
                    invalid[index][field] = value;
                    let compared = comparison(&invalid, row_workers);
                    assert_eq!(compared["status"], "captured-complete-pair");
                    assert_eq!(compared["statePolicyMatched"], false);
                    assert_eq!(compared["performanceComparisonEligible"], false);
                    assert!(compared["ptq1OverPq2RequestLatency"].is_null());
                    assert_eq!(
                        compared["resources"]["reason"],
                        "missing-or-mismatched-state-policy-readiness"
                    );
                    assert_eq!(compared["resources"]["metrics"], json!([]));
                }
            }
        }
        let mut not_started = json!({});
        record_cpu_placement(&mut not_started, false);
        assert_eq!(not_started["configuredPlacement"], "cpu");
        assert_eq!(not_started["cpuPlacementIndependentlyVerified"], false);
        assert!(not_started["verifiedProjectionBackend"].is_null());
        assert_eq!(not_started["statePolicyIndependentlyVerified"], false);
    }
}

#[test]
fn native_representation_rollback_observation_survives_pending_ptq1_arm() {
    for row_workers in [1, 4] {
        assert_eq!(
            original_pq2_rollback_observation(&[], row_workers)["reason"],
            "original-pq2-arm-not-started"
        );
        let original = complete_sample(false, row_workers);
        let observed =
            original_pq2_rollback_observation(std::slice::from_ref(&original), row_workers);
        assert_eq!(observed["status"], "captured-original-pq2-arm");
        assert_eq!(observed["independentGateReview"], "pending");
        assert_eq!(observed["defaultMayChange"], false);
        let pending_pair = [original.clone(), Value::Null];
        assert_eq!(
            original_pq2_rollback_observation(&pending_pair, row_workers),
            observed
        );
        assert_eq!(comparison(&pending_pair, row_workers)["status"], "pending");
        for (path, value) in [
            ("/modelIdentity/sha256", json!("0".repeat(64))),
            ("/verifiedReadinessObserved", json!(false)),
            ("/observedNativeRowWorkers", Value::Null),
            ("/stream/finishReason", json!("length")),
            ("/stream/reportedCompletionTokens", json!(0)),
            ("/stream/visibleContentSha256", json!("0".repeat(64))),
            ("/cleanup/ownedJobEmpty", json!(false)),
            ("/cleanup/listenerReleased", json!(false)),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(path).unwrap() = value;
            assert_eq!(
                original_pq2_rollback_observation(&[changed], row_workers)["status"],
                "pending",
                "{path}"
            );
        }
        assert_eq!(
            original_pq2_rollback_observation(&[complete_sample(true, row_workers)], row_workers)["status"],
            "pending"
        );
        assert_eq!(
            original_pq2_rollback_observation(&[original], if row_workers == 1 { 4 } else { 1 })["status"],
            "pending"
        );
    }
}

#[test]
fn native_representation_resources_compare_matching_domains_only() {
    for row_workers in [1, 4] {
        let mut pair = vec![
            complete_sample(false, row_workers),
            complete_sample(true, row_workers),
        ];
        for (field, _) in RESOURCE_METRICS {
            let previous = pair[1]["cleanup"]["backendResources"][field]
                .as_u64()
                .unwrap();
            pair[1]["cleanup"]["backendResources"][field] = json!(previous / 2);
        }
        pair[1]["cleanup"]["backendResources"]["successfulGuardSamples"] = json!(3);
        let observed = comparison(&pair, row_workers);
        let resources = &observed["resources"];
        assert_eq!(resources["status"], "observed-bounded-pair");
        let metrics = resources["metrics"].as_array().unwrap();
        assert_eq!(metrics.len(), RESOURCE_METRICS.len());
        for (metric, (field, scope)) in metrics.iter().zip(RESOURCE_METRICS) {
            assert_eq!(metric["field"], field);
            assert_eq!(metric["scope"], scope);
            assert_eq!(
                metric["pq2Bytes"],
                pair[0]["cleanup"]["backendResources"][field]
            );
            assert_eq!(
                metric["ptq1Bytes"],
                pair[1]["cleanup"]["backendResources"][field]
            );
            assert_eq!(metric["ptq1OverPq2"], 0.5);
        }
        let serving = &resources["servingObservationsInRepresentationOrder"];
        assert_eq!(serving[0]["successfulGuardSamples"], 2);
        assert_eq!(serving[1]["successfulGuardSamples"], 3);
        assert_eq!(
            serving[0]["sampledMinimumSystemAvailablePhysicalBytes"],
            4_u64 * 1024 * 1024 * 1024
        );
        for field in [
            "fullResourceGateQualified",
            "pagingQualified",
            "sustainedComfortQualified",
            "resourceBenefitQualified",
        ] {
            assert_eq!(resources[field], false);
        }
        pair[1]["stream"]["reportedCompletionTokens"] = json!(4);
        let unmatched = comparison(&pair, row_workers);
        assert_eq!(unmatched["status"], "captured-complete-pair");
        assert!(unmatched["ptq1OverPq2RequestLatency"].is_null());
        assert_eq!(
            unmatched["resources"]["reason"],
            "different-complete-output-or-usage"
        );
        assert_eq!(unmatched["resources"]["metrics"], json!([]));
    }
}

#[test]
fn native_representation_resources_reject_missing_or_inconsistent_observations() {
    let pair = vec![complete_sample(false, 4), complete_sample(true, 4)];
    assert_eq!(
        resource_comparison(&pair[..1], true, true, true)["reason"],
        "incomplete-pair"
    );
    for index in 0..2 {
        let mut changed = pair.clone();
        changed[index]["cleanup"]["backendResources"] = Value::Null;
        let missing = comparison(&changed, 4);
        assert_eq!(missing["status"], "captured-complete-pair");
        assert_eq!(missing["resources"]["status"], "pending");
        assert_eq!(missing["resources"]["metrics"], json!([]));
        for field in RESOURCE_METRICS.iter().map(|(field, _)| *field).chain([
            "successfulGuardSamples",
            "sampledMinimumSystemAvailablePhysicalBytes",
        ]) {
            for invalid in [Value::Null, json!(0), json!(-1), json!(0.5)] {
                let mut changed = pair.clone();
                changed[index]["cleanup"]["backendResources"][field] = invalid;
                let result = comparison(&changed, 4);
                assert_eq!(
                    result["resources"]["reason"], "missing-positive-resource-observations",
                    "{field}"
                );
                assert_eq!(result["resources"]["metrics"], json!([]));
            }
        }
        for (field, invalid) in [
            ("guardIntervalMilliseconds", Value::Null),
            ("guardIntervalMilliseconds", json!(500)),
            ("sampledPeakRootPrivateBytes", json!(1001)),
            ("sampledPeakRootWorkingSetBytes", json!(2001)),
        ] {
            let mut changed = pair.clone();
            changed[index]["cleanup"]["backendResources"][field] = invalid;
            let result = comparison(&changed, 4);
            assert_eq!(result["resources"]["status"], "pending", "{field}");
            assert_eq!(result["resources"]["metrics"], json!([]));
        }
    }
}

#[test]
fn native_representation_comparison_requires_released_worker_observation() {
    for row_workers in [1, 4] {
        let pair = vec![
            complete_sample(false, row_workers),
            complete_sample(true, row_workers),
        ];
        for (field, mismatched) in [
            ("rowWorkerReadinessVersion", json!("unreviewed-release")),
            ("rowWorkerReadinessRevision", json!("0".repeat(40))),
            (
                "rowWorkerCountIndependentlyObserved",
                json!(
                    !pair[0]["rowWorkerCountIndependentlyObserved"]
                        .as_bool()
                        .unwrap()
                ),
            ),
            (
                "observedNativeRowWorkers",
                json!(if row_workers == 1 { 4 } else { 1 }),
            ),
        ] {
            for index in 0..2 {
                for value in [Value::Null, mismatched.clone()] {
                    let mut changed = pair.clone();
                    changed[index][field] = value;
                    assert_eq!(
                        comparison(&changed, row_workers)["reason"],
                        "missing-or-mismatched-worker-readiness-identity",
                        "{field}, sample {index}"
                    );
                    assert!(
                        comparison(&changed, row_workers)["ptq1OverPq2RequestLatency"].is_null()
                    );
                }
            }
        }
    }
}

#[test]
fn native_representation_profile_is_explicit_and_keeps_original_deadline() {
    for (profile, case) in [(PROFILE, CASE), (ROW4_PROFILE, ROW4_CASE)] {
        assert_eq!(profile_case(profile, false, false), Some(case));
        assert_eq!(profile_case(profile, true, false), None);
        assert_eq!(profile_case(profile, false, true), None);
    }
    assert_eq!(body()["max_completion_tokens"], CAP);
    assert_eq!(body()["enable_thinking"], false);
    assert_eq!(body()["abliteration_scale"], 1.3);
    run_paused(async {
        let deadline = tokio::time::Instant::now() + STOP_WAIT;
        let mut started = false;
        assert_eq!(
            adapter_work_before_cleanup(deadline, async {
                started = true;
            })
            .await,
            None
        );
        assert!(!started);
        assert_eq!(
            bonsai_broker::deadline::remaining_until(deadline),
            STOP_WAIT
        );
    });
}

#[test]
fn native_representation_row4_profile_is_explicit_and_preserves_pair() {
    assert_ne!(PROFILE, ROW4_PROFILE);
    assert_ne!(CASE, ROW4_CASE);
    for (single, parallel) in profiles(1).iter().zip(profiles(4).iter()) {
        assert_eq!(
            (single.name, single.case, single.row_workers),
            (PROFILE, CASE, 1)
        );
        assert_eq!(
            (parallel.name, parallel.case, parallel.row_workers),
            (ROW4_PROFILE, ROW4_CASE, 4)
        );
        assert_eq!(single.repacked, parallel.repacked);
        assert!(
            single.placement == AdapterPlacement::Cpu
                && parallel.placement == AdapterPlacement::Cpu
        );
        assert!(!parallel.optional_requests && !parallel.required_scale_history);
        let mut expected = adapter_arguments(single, "8081");
        *expected.last_mut().unwrap() = "4".into();
        let mut actual = adapter_arguments(parallel, "8081");
        assert_eq!(actual, expected);
        assert_eq!(
            actual
                .array_windows::<2>()
                .filter(|[flag, value]| flag == "--placement" && value == "cpu")
                .count(),
            1
        );
        for (flag, value) in [("--kv-cache", "rsq4"), ("--prompt-reuse", "disabled")] {
            assert_eq!(
                actual
                    .array_windows::<2>()
                    .filter(|[arg, selected]| arg == flag && selected == value)
                    .count(),
                1
            );
        }
        assert_eq!(
            actual.iter().any(|arg| arg == "--use-original-pq2"),
            !parallel.repacked
        );
        assert_eq!(
            actual.iter().any(|arg| arg == "--use-repacked-ptq1"),
            parallel.repacked
        );
        actual.extend(
            [
                "--data-directory",
                "C:/model-not-opened",
                "--evidence-parent",
                "C:/evidence-not-opened",
            ]
            .map(OsString::from),
        );
        assert!(LaunchConfig::parse(actual).is_err());
    }
    assert!(!profiles(4)[0].repacked && profiles(4)[1].repacked);
    assert_eq!(NEXT_OWNER_RESERVE, Duration::from_mins(1));
    assert_eq!(STOP_WAIT, Duration::from_secs(30));
    assert_eq!(CASE_BUDGET, Duration::from_mins(4));
}

#[test]
fn native_representation_profiles_leave_skipped_arms_pending_at_original_deadline() {
    run_paused(async {
        let evidence = TempEvidence::new("Bonsai-Representation-Reserve");
        let deadline = tokio::time::Instant::now() + NEXT_OWNER_RESERVE + Duration::from_secs(1);
        assert!(can_start_owner(deadline));
        tokio::time::advance(Duration::from_secs(1)).await;
        assert!(!can_start_owner(deadline));
        for row_workers in [1, 4] {
            for samples in [vec![], vec![complete_sample(false, row_workers)]] {
                let mut report = json!({"passed":false,"samples":samples});
                paired_responses(
                    Path::new("C:/model-not-opened"),
                    &evidence,
                    deadline,
                    row_workers,
                    &mut report,
                )
                .await;
                assert_eq!(report["samples"], json!(samples));
                assert_eq!(report["passed"], false);
                assert_eq!(report["comparison"]["status"], "pending");
                assert_eq!(
                    report["pendingReason"],
                    "insufficient-next-owner-and-cleanup-reserve"
                );
                assert_eq!(
                    bonsai_broker::deadline::remaining_until(deadline),
                    NEXT_OWNER_RESERVE
                );
                let retained: Value =
                    serde_json::from_slice(&fs::read(evidence.join("acceptance.json")).unwrap())
                        .unwrap();
                assert_eq!(retained, report);
            }
        }
    });
}

#[test]
fn native_representation_complete_bytes_survive_expired_second_request() {
    run_paused(async {
        timeout(Duration::from_secs(10), async {
            let evidence = TempEvidence::new("Bonsai-Representation-Partial");
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let fixture = tokio::spawn(async move {
                for step in 0..2 {
                    let (peer, _) = listener.accept().await.unwrap();
                    let mut peer = BufReader::new(peer);
                    assert_eq!(read_fixture_request(&mut peer).await, body());
                    if step == 0 {
                        let payload = concat!(
                            "data: {\"choices\":[{\"delta\":{\"content\":\"cèdre\\n\"},\"finish_reason\":null}]}\n\n",
                            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":3}}\n\n",
                            "data: [DONE]\n\n"
                        );
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:X}\r\n{payload}\r\n",
                            payload.len()
                        );
                        peer.get_mut().write_all(response.as_bytes()).await.unwrap();
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        peer.get_mut().write_all(b"0\r\n\r\n").await.unwrap();
                        peer.get_mut().shutdown().await.unwrap();
                    } else {
                        assert_eq!(peer.read(&mut [0]).await.unwrap(), 0);
                    }
                }
            });
            let deadline = tokio::time::Instant::now() + STOP_WAIT + Duration::from_millis(300);
            let mut report = json!({"passed":false,"samples":[{},{}]});
            let captured =
                record_response(port, "fixture", &evidence, deadline, &mut report, 0).await;
            assert!(captured.is_some());
            assert!(report["samples"][0]["stream"].is_null());
            report["samples"][0]["cleanup"] =
                json!({"status":"confirmed","ownedJobEmpty":true,"listenerReleased":true});
            assert!(finish_response_after_cleanup(
                captured.as_ref(),
                &evidence,
                &mut report,
                0
            ));
            let first: Value =
                serde_json::from_slice(&fs::read(evidence.join("acceptance.json")).unwrap())
                    .unwrap();
            assert!(
                record_response(port, "fixture", &evidence, deadline, &mut report, 1)
                    .await
                    .is_none()
            );
            fixture.await.unwrap();
            let saved: Value =
                serde_json::from_slice(&fs::read(evidence.join("acceptance.json")).unwrap())
                    .unwrap();
            assert_eq!(saved["samples"][0], first["samples"][0]);
            assert_eq!(saved["samples"][0]["stream"]["visibleContent"], "cèdre\n");
            assert_eq!(
                saved["samples"][0]["stream"]["visibleContentSha256"],
                adapter_digest("cèdre\n".as_bytes())
            );
            assert_eq!(saved["samples"][0]["stream"]["reportedCompletionTokens"], 3);
            assert_eq!(saved["samples"][0]["stream"]["naturallyStopped"], true);
            assert!(
                saved["samples"][0]["stream"]["requestMilliseconds"]
                    .as_f64()
                    .unwrap()
                    >= 20.0
            );
            assert!(saved["samples"][1]["stream"].is_null());
            assert_eq!(saved["pendingReason"], "request-budget-expired");
            assert_eq!(saved["passed"], false);
            assert_eq!(
                comparison(saved["samples"].as_array().unwrap(), 1)["status"],
                "pending"
            );
        })
        .await
        .unwrap();
    });
}

#[test]
fn native_representation_incomplete_first_arm_blocks_next_owner_after_cleanup() {
    let evidence = TempEvidence::new("Bonsai-Representation-Incomplete");
    for (finish_reason, tokens, content, elapsed) in [
        (
            "length",
            Some(u64::from(CAP)),
            "Oak",
            Duration::from_millis(1),
        ),
        ("stop", None, "Oak", Duration::from_millis(1)),
        ("stop", Some(0), "Oak", Duration::from_millis(1)),
        ("stop", Some(1), "  ", Duration::from_millis(1)),
        ("stop", Some(1), "Oak", Duration::ZERO),
    ] {
        let content_event = json!({"choices":[{"delta":{"content":content},"finish_reason":null}]});
        let mut terminal = json!({"choices":[{"delta":{},"finish_reason":finish_reason}]});
        if let Some(tokens) = tokens {
            terminal["usage"] = json!({"completion_tokens":tokens});
        }
        let payload = format!("data: {content_event}\n\ndata: {terminal}\n\ndata: [DONE]\n\n");
        let captured = CapturedResponse {
            bytes: chunked_sse(&payload),
            elapsed,
        };
        let mut report = json!({"passed":false,"samples":[{
            "cleanup":{"status":"confirmed","ownedJobEmpty":true,"listenerReleased":true}}]});
        assert!(!finish_response_after_cleanup(
            Some(&captured),
            &evidence,
            &mut report,
            0
        ));
        assert_eq!(
            report["pendingReason"],
            "incomplete-or-invalid-complete-response"
        );
        assert_eq!(report["samples"][0]["responseParsePassed"], true);
        assert_eq!(
            report["samples"][0]["stream"]["finishReason"],
            finish_reason
        );
        assert_eq!(
            report["samples"][0]["stream"]["reportedCompletionTokens"],
            json!(tokens)
        );
        assert_eq!(report["samples"][0]["stream"]["visibleContent"], content);
        assert_eq!(report["samples"][0]["cleanup"]["ownedJobEmpty"], true);
        let retained: Value =
            serde_json::from_slice(&fs::read(evidence.join("acceptance.json")).unwrap()).unwrap();
        assert_eq!(retained, report);
        assert_eq!(retained["passed"], false);
    }
}

#[test]
fn native_representation_invalid_transport_and_sse_wait_for_cleanup_and_keep_receipts() {
    run_timed(Duration::from_secs(10), async {
        let evidence = TempEvidence::new("Bonsai-Representation-Invalid");
        for wire in [
            b"HTTP/1.1 503 Unavailable\r\n\r\n".as_slice(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nbad\r\n0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nbad\r\n0\r\n",
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let fixture = tokio::spawn(async move {
                let (peer, _) = listener.accept().await.unwrap();
                let mut peer = BufReader::new(peer);
                let _ = read_fixture_request(&mut peer).await;
                peer.get_mut().write_all(wire).await.unwrap();
                peer.get_mut().shutdown().await.unwrap();
            });
            let mut report = json!({"passed":false,"samples":[complete_sample(false, 4)]});
            let captured = record_response(
                port,
                "fixture",
                &evidence,
                tokio::time::Instant::now() + STOP_WAIT + Duration::from_secs(5),
                &mut report,
                0,
            )
            .await;
            fixture.await.unwrap();
            assert!(captured.is_some());
            assert!(report["samples"][0]["stream"].is_null());
            assert_eq!(report["samples"][0]["responseBytes"], wire.len());
            assert_eq!(report["samples"][0]["responseSha256"], adapter_digest(wire));
            report["samples"][0]["cleanup"] =
                json!({"status":"pending","ownedJobEmpty":null,"listenerReleased":null});
            assert!(!finish_response_after_cleanup(
                captured.as_ref(),
                &evidence,
                &mut report,
                0
            ));
            assert_eq!(report["pendingReason"], "owned-cleanup-unconfirmed");
            report["samples"][0]["cleanup"] =
                json!({"status":"confirmed","ownedJobEmpty":true,"listenerReleased":true});
            assert!(!finish_response_after_cleanup(
                captured.as_ref(),
                &evidence,
                &mut report,
                0
            ));
            assert_eq!(report["pendingReason"], "invalid-or-incomplete-response");
            assert!(report["samples"][0]["stream"].is_null());
            let saved: Value =
                serde_json::from_slice(&fs::read(evidence.join("acceptance.json")).unwrap())
                    .unwrap();
            assert_eq!(saved, report);
            assert_eq!(saved["samples"][0]["cleanup"]["ownedJobEmpty"], true);
            assert_eq!(
                saved["samples"][0]["powerBefore"],
                saved["samples"][0]["powerAfter"]
            );
            assert_eq!(saved["passed"], false);
        }
        let mut report = json!({"passed":false,"samples":[{}]});
        assert!(
            record_response(
                0,
                "fixture",
                &evidence,
                tokio::time::Instant::now() + STOP_WAIT + Duration::from_secs(5),
                &mut report,
                0
            )
            .await
            .is_none()
        );
        assert_eq!(report["pendingReason"], "request-transport-failed");
        assert_eq!(report["requestProgress"][0]["transportErrorObserved"], true);
        assert!(report["samples"][0]["stream"].is_null());
    });
}

#[test]
#[ignore = "Actual-model complete-response PQ2/PTQ1 feasibility pair; one bounded main-only device CI job."]
fn pinned_native_ptq1_pq2_capture_complete_responses() {
    capture_complete_responses(1);
}

#[test]
#[ignore = "Actual-model complete-response four-worker PQ2/PTQ1 feasibility pair; one bounded main-only device CI job."]
fn pinned_native_ptq1_pq2_row4_capture_complete_responses() {
    capture_complete_responses(4);
}

fn capture_complete_responses(row_workers: u8) {
    ci_only();
    let [profile, _] = profiles(row_workers);
    let budget = case_budget();
    assert_eq!(
        profile_case(
            &env::var("BONSAI_NATIVE_PROFILE").unwrap(),
            env::var("BONSAI_NATIVE_EXTENDED").as_deref() == Ok("true"),
            env::var("BONSAI_NATIVE_STALLED_CONSUMER").as_deref() == Ok("true")
        ),
        Some(profile.case)
    );
    let data = env::var_os("BONSAI_NATIVE_DATA").unwrap();
    let evidence = env::var_os("BONSAI_NATIVE_EVIDENCE").unwrap();
    let evidence = Path::new(&evidence);
    let mut report = json!({"schemaVersion":1,"case":profile.case,"profile":profile.name,
        "revision":env::var("GITHUB_SHA").unwrap(),"runId":env::var("GITHUB_RUN_ID").unwrap(),
        "runAttempt":env::var("GITHUB_RUN_ATTEMPT").unwrap(),"passed":false,
        "scope":"bounded-single-pair-cpu-only-complete-response-feasibility",
        "comparisonKind":"cpu-only-packing","configuredPlacement":"cpu",
        "configuredKvCache":"rsq4","configuredPromptReuse":"disabled",
        "statePolicyScope":"K4/V4 with reset on every request; supplies no CPU/FP16 numerical or full-context/cache quality qualification",
        "automaticPq2PlacementComparisonQualified":false,
        "request":body(),"requestSha256":adapter_digest(&serde_json::to_vec(&body()).unwrap()),
        "representationOrder":["PQ2_0","current-repacked-PTQ1_0"],"configuredNativeRowWorkers":row_workers,
        "configuredSeed":42,"requestedAdapterScale":1.3,"servingStatePolicy":"reset-every-request",
        "templateSha256":"c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        "sampler":{"temperature":1,"topP":0.95,"topK":20,"minP":0,
            "repeatPenalty":1,"presencePenalty":0,"frequencyPenalty":0},
        "metricDefinition":"complete framed request milliseconds including prefill, generation, transport and incremental observer/receipt I/O; startup recorded separately; no decode timing",
        "aggregateModelCaseBudgetSeconds":budget.as_secs(),"maximumModelCaseBudgetSeconds":maximum_case_budget().as_secs(),
        "wholeJobMaximumSeconds":whole_job_budget().as_secs(),"requiredNativeSandbox":true,"samples":[],"comparison":{"status":"pending"},
        "runtimeLock":runtime_lock().clone(),
        "adapterLock":adapter_lock().clone(),
        "ptq1ExportReceipt":ptq1_export_lock().clone(),
        "decodeTokensPerSecond":null,"performanceQualified":false,"qualityQualified":false,
        "authoringBasisQualified":false,"completeStateParityQualified":false,"defaultMayChange":false,
        "defaultRuntimeQualified":false,"matrixContinuationPermitted":false});
    report["originalPq2Rollback"] = original_pq2_rollback_observation(&[], row_workers);
    save_acceptance_report(evidence, &report);
    run_local_set(async {
        timeout(
            budget,
            paired_responses(
                Path::new(&data),
                evidence,
                tokio::time::Instant::now() + budget,
                row_workers,
                &mut report,
            ),
        )
        .await
        .unwrap();
    });
    assert_eq!(
        report["passed"], true,
        "Complete-response feasibility pair remains pending."
    );
}
