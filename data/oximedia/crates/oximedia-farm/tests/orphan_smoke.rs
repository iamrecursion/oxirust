//! Smoke tests for the 17 orphan modules wired in lib.rs.
//!
//! Each test group covers 2–3 modules through their public APIs, eliminating
//! the need for `#![allow(dead_code)]` silencers.
//!
//! Groupings (module count / test count):
//!  1. capabilities + affinity          (2 tests)
//!  2. blacklist + topology             (3 tests)
//!  3. farm_metrics                     (3 tests)
//!  4. farm_topology                    (2 tests)
//!  5. load_balancer + job_distribution (3 tests)
//!  6. job_progress                     (2 tests)
//!  7. job_retry_policy                 (3 tests)
//!  8. task_affinity                    (3 tests)
//!  9. dag_viz                          (4 tests)
//! 10. cost_accounting                  (2 tests)
//! 11. dashboard_api                    (3 tests)
//! 12. energy                           (4 tests)
//! 13. license_pool                     (2 tests)
//! 14. worker_metrics                   (3 tests)

#![cfg(not(target_arch = "wasm32"))]

// ---------------------------------------------------------------------------
// 1. capabilities + affinity
// ---------------------------------------------------------------------------

#[test]
fn test_capabilities_and_affinity_rule_match() {
    use oximedia_farm::affinity::JobAffinityRule;
    use oximedia_farm::capabilities::WorkerCapabilities;

    let mut caps = WorkerCapabilities::new(1);
    caps.add_codec("av1");
    caps.add_codec("h264");
    caps.add_gpu("nvidia-a100");

    assert!(caps.supports("av1"));
    assert!(caps.supports("H264")); // case-insensitive
    assert!(caps.supports("nvidia-a100"));
    assert!(!caps.supports("vp9"));
    assert_eq!(caps.worker_id(), 1);
    assert_eq!(caps.codecs().len(), 2);
    assert_eq!(caps.gpus().len(), 1);

    let rule_match = JobAffinityRule::new("av1");
    assert!(rule_match.matches(&caps));
    assert_eq!(rule_match.required_capability(), "av1");

    let rule_no_match = JobAffinityRule::new("vp9");
    assert!(!rule_no_match.matches(&caps));
}

#[test]
fn test_affinity_rule_case_normalised() {
    use oximedia_farm::affinity::JobAffinityRule;
    use oximedia_farm::capabilities::WorkerCapabilities;

    let rule = JobAffinityRule::new("NVIDIA-A100");
    assert_eq!(rule.required_capability(), "nvidia-a100");

    let mut caps = WorkerCapabilities::new(99);
    caps.add_gpu("NVIDIA-A100");
    assert!(rule.matches(&caps));
}

// ---------------------------------------------------------------------------
// 2. blacklist + topology
// ---------------------------------------------------------------------------

#[test]
fn test_blacklist_add_query_remove() {
    use oximedia_farm::blacklist::WorkerBlacklist;

    let mut bl = WorkerBlacklist::new();
    assert!(!bl.is_blocked(42));
    assert_eq!(bl.len(), 0);
    assert!(bl.is_empty());

    bl.add(42, "repeated failures");
    assert!(bl.is_blocked(42));
    assert_eq!(bl.reason(42), Some("repeated failures"));
    assert!(!bl.is_blocked(7));

    bl.add(42, "updated reason");
    assert_eq!(bl.reason(42), Some("updated reason"));
    assert_eq!(bl.len(), 1);

    bl.remove(42);
    assert!(!bl.is_blocked(42));
    assert_eq!(bl.reason(42), None);
    assert_eq!(bl.len(), 0);
}

#[test]
fn test_blacklist_filter_available() {
    use oximedia_farm::blacklist::WorkerBlacklist;

    let mut bl = WorkerBlacklist::new();
    bl.add(2, "disk full");
    bl.add(4, "oom");

    let available = bl.filter_available(&[1, 2, 3, 4, 5]);
    assert_eq!(available, vec![1, 3, 5]);
}

#[test]
fn test_topology_add_and_query() {
    use oximedia_farm::topology::{FarmNode, FarmTopology};

    let mut topo = FarmTopology::new();
    assert_eq!(topo.worker_count(), 0);
    assert!(topo.is_empty());

    topo.add_node(FarmNode::new("rack-01", "host-a", 1));
    topo.add_node(FarmNode::new("rack-01", "host-b", 2));
    topo.add_node(FarmNode::new("rack-02", "host-c", 3));

    assert_eq!(topo.worker_count(), 3);

    let rack1 = topo.nodes_in_rack("rack-01");
    assert_eq!(rack1.len(), 2);

    let rack2 = topo.nodes_in_rack("rack-02");
    assert_eq!(rack2.len(), 1);

    assert_eq!(topo.nodes_in_rack("rack-99").len(), 0);

    // Duplicate worker_id is silently ignored
    topo.add_node(FarmNode::new("rack-99", "host-dup", 1));
    assert_eq!(topo.worker_count(), 3);

    // FarmNode helpers
    let node = FarmNode::new("rack-03", "enc-12", 99);
    assert_eq!(node.location(), "rack-03/enc-12");
    assert_eq!(node.rack, "rack-03");
    assert_eq!(node.worker_id, 99);

    // find_node
    assert!(topo.find_node(1).is_some());
    assert!(topo.find_node(999).is_none());

    // Remove
    let removed = topo.remove_node(1);
    assert!(removed.is_some());
    assert_eq!(topo.worker_count(), 2);
}

// ---------------------------------------------------------------------------
// 3. farm_metrics
// ---------------------------------------------------------------------------

#[test]
fn test_farm_metrics_record_and_latest() {
    use oximedia_farm::farm_metrics::{FarmMetricPoint, FarmMetrics, METRIC_QUEUE_DEPTH};

    let mut fm = FarmMetrics::new();
    assert_eq!(fm.latest(METRIC_QUEUE_DEPTH), None);

    fm.record(FarmMetricPoint {
        timestamp_secs: 1000,
        metric_name: METRIC_QUEUE_DEPTH.to_string(),
        value: 42.0,
        worker_id: None,
    });
    assert_eq!(fm.latest(METRIC_QUEUE_DEPTH), Some(42.0));

    fm.record(FarmMetricPoint {
        timestamp_secs: 1010,
        metric_name: METRIC_QUEUE_DEPTH.to_string(),
        value: 55.0,
        worker_id: None,
    });
    assert_eq!(fm.latest(METRIC_QUEUE_DEPTH), Some(55.0));
}

#[test]
fn test_farm_metrics_windowed_average() {
    use oximedia_farm::farm_metrics::{FarmMetricPoint, FarmMetrics, METRIC_WORKER_UTILIZATION};

    let mut fm = FarmMetrics::new();
    for (t, v) in [(100u64, 0.4_f64), (110, 0.6), (120, 0.8)] {
        fm.record(FarmMetricPoint {
            timestamp_secs: t,
            metric_name: METRIC_WORKER_UTILIZATION.to_string(),
            value: v,
            worker_id: None,
        });
    }

    let avg = fm
        .average(METRIC_WORKER_UTILIZATION, 30, 120)
        .expect("windowed avg");
    let expected = (0.4 + 0.6 + 0.8) / 3.0;
    assert!(
        (avg - expected).abs() < 1e-9,
        "avg={avg}, expected={expected}"
    );

    // Window that excludes the oldest point
    let avg_short = fm
        .average(METRIC_WORKER_UTILIZATION, 5, 120)
        .expect("short avg");
    // Only t=120 falls in [115, 120]
    assert!((avg_short - 0.8).abs() < 1e-9);
}

#[test]
fn test_farm_metrics_generate_report() {
    use oximedia_farm::farm_metrics::{
        FarmMetricPoint, FarmMetrics, METRIC_FAILURE_RATE, METRIC_TASK_DURATION_SECS,
        METRIC_WORKER_UTILIZATION,
    };

    let mut fm = FarmMetrics::new();

    // Record task completions (METRIC_TASK_DURATION_SECS) — each point = one task
    for t in [1990u64, 1995, 2000] {
        fm.record(FarmMetricPoint {
            timestamp_secs: t,
            metric_name: METRIC_TASK_DURATION_SECS.to_string(),
            value: 45.0,
            worker_id: None,
        });
    }

    fm.record(FarmMetricPoint {
        timestamp_secs: 2000,
        metric_name: METRIC_FAILURE_RATE.to_string(),
        value: 0.05,
        worker_id: None,
    });

    fm.record(FarmMetricPoint {
        timestamp_secs: 2000,
        metric_name: METRIC_WORKER_UTILIZATION.to_string(),
        value: 0.80,
        worker_id: None,
    });

    let report = fm.generate_report(60, 2000);
    assert_eq!(report.generated_at_secs, 2000);
    assert_eq!(report.total_tasks_completed, 3);
    assert!((report.avg_task_duration_secs - 45.0).abs() < 1e-9);
    // efficiency_pct = (1-0.05) * 0.80 * 100 = 76.0
    assert!((report.efficiency_pct - 76.0_f32).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// 4. farm_topology
// ---------------------------------------------------------------------------

#[test]
fn test_farm_topology_rack_zone_worker_placement() {
    use oximedia_farm::farm_topology::{FarmTopology, Rack, RackId, Zone, ZoneId};
    use oximedia_farm::WorkerId;

    let mut ft = FarmTopology::new();

    let zone_id = ZoneId::new("us-west-2");
    ft.add_zone(
        Zone::new(zone_id.clone(), "US West 2", Some("us-west".to_string())).expect("zone"),
    )
    .expect("add zone");

    let rack_id = RackId::new("rack-01");
    ft.add_rack(Rack::new(rack_id.clone(), zone_id.clone(), "Primary Rack", 0).expect("rack"))
        .expect("add rack");

    let worker = WorkerId::new("w1");
    ft.place_worker(worker.clone(), &rack_id)
        .expect("place worker");

    let rack = ft.rack(&rack_id).expect("rack found");
    assert!(rack.workers.contains(&worker));
    assert_eq!(ft.worker_rack(&worker), Some(&rack_id));

    // Zone lookup works
    let zone = ft.zone(&zone_id).expect("zone found");
    assert_eq!(zone.label, "US West 2");

    // zone/rack identifiers
    assert_eq!(zone_id.as_str(), "us-west-2");
    assert_eq!(rack_id.as_str(), "rack-01");
}

#[test]
fn test_farm_topology_worker_group_join() {
    use oximedia_farm::farm_topology::{FarmTopology, GroupId, WorkerGroup};
    use oximedia_farm::WorkerId;

    let mut ft = FarmTopology::new();

    let g_id = GroupId::new("transcoding");
    ft.add_group(WorkerGroup::new(g_id.clone(), "Transcoding Farm").expect("group"))
        .expect("add group");

    let w = WorkerId::new("enc-42");
    ft.join_group(w.clone(), &g_id).expect("join group");

    let group = ft.group(&g_id).expect("group found");
    assert!(group.members.contains(&w));
    assert_eq!(g_id.as_str(), "transcoding");

    ft.leave_group(&w, &g_id).expect("leave group");
    let group_after = ft.group(&g_id).expect("group still exists");
    assert!(!group_after.members.contains(&w));
}

// ---------------------------------------------------------------------------
// 5. load_balancer + job_distribution
// ---------------------------------------------------------------------------

fn make_worker_capacity(
    id: &str,
    active: u32,
    max: u32,
    latency: f64,
) -> oximedia_farm::load_balancer::WorkerCapacity {
    oximedia_farm::load_balancer::WorkerCapacity {
        worker_id: id.to_string(),
        weight: 1,
        cpu_cores_available: 4.0,
        memory_mb_available: 8192,
        active_jobs: active,
        max_jobs: max,
        avg_latency_ms: latency,
        job_type_affinity: vec![],
        gpu_count: 0,
        gpu_memory_mb_available: 0,
        network_bandwidth_mbps: 1000.0,
        disk_io_mbps: 200.0,
    }
}

#[test]
fn test_load_balancer_round_robin() {
    use oximedia_farm::load_balancer::{LoadBalancer, LoadBalancingStrategy};

    let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
    lb.add_worker(make_worker_capacity("w1", 0, 4, 10.0));
    lb.add_worker(make_worker_capacity("w2", 0, 4, 10.0));

    let first = lb.select("transcode", 1.0, 512).expect("first pick");
    let second = lb.select("transcode", 1.0, 512).expect("second pick");
    // Round-robin: two consecutive picks must differ
    assert_ne!(first, second, "round-robin should alternate workers");
}

#[test]
fn test_load_balancer_least_connections() {
    use oximedia_farm::load_balancer::{LoadBalancer, LoadBalancingStrategy};

    let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastConnections);
    lb.add_worker(make_worker_capacity("busy", 3, 4, 5.0));
    lb.add_worker(make_worker_capacity("idle", 0, 4, 5.0));

    let picked = lb.select("transcode", 1.0, 512).expect("pick");
    assert_eq!(
        picked, "idle",
        "least-connections should pick the idle worker"
    );
}

#[test]
fn test_job_distribution_locality() {
    use oximedia_farm::job_distribution::{FarmJob, JobDistributor};
    use oximedia_farm::load_balancer::LoadBalancingStrategy;

    let mut jd = JobDistributor::new(LoadBalancingStrategy::LeastConnections);
    jd.load_balancer
        .add_worker(make_worker_capacity("local-w", 0, 8, 5.0));
    jd.load_balancer
        .add_worker(make_worker_capacity("remote-w", 0, 8, 50.0));

    jd.register_locality("storage-a", vec!["local-w".to_string()]);

    let job = FarmJob {
        id: "job-1".to_string(),
        job_type: "transcode".to_string(),
        priority: 1,
        input_paths: vec!["/data/input.mov".to_string()],
        required_cpu_cores: 1.0,
        required_memory_mb: 512,
        estimated_duration_secs: 60,
        data_locality_hint: Some("storage-a".to_string()),
        gpu_required: false,
    };

    let decision = jd.distribute(&job).expect("distribution decision");
    assert_eq!(
        decision.assigned_worker, "local-w",
        "locality hint should route to local worker"
    );
    assert_eq!(decision.reason, "locality-preferred");
}

// ---------------------------------------------------------------------------
// 6. job_progress
// ---------------------------------------------------------------------------

#[test]
fn test_job_progress_tracker_eta() {
    use oximedia_farm::job_progress::{JobProgressTracker, ProgressUpdate};
    use oximedia_farm::JobId;

    let job_id = JobId::new();
    let mut tracker = JobProgressTracker::new(job_id);
    // No samples yet → snapshot errors
    assert!(tracker.snapshot().is_err(), "no snapshot before any data");

    tracker
        .record_progress(ProgressUpdate {
            percent: 25.0,
            phase: "encoding".to_string(),
            message: "frame 250/1000".to_string(),
            timestamp_secs: 10,
        })
        .expect("record first progress");

    tracker
        .record_progress(ProgressUpdate {
            percent: 50.0,
            phase: "encoding".to_string(),
            message: "frame 500/1000".to_string(),
            timestamp_secs: 20,
        })
        .expect("record second progress");

    let snap = tracker.snapshot().expect("snapshot after two samples");
    assert_eq!(snap.percent, 50.0);
    assert_eq!(snap.phase, "encoding");
    assert!(
        snap.eta_secs.is_some(),
        "ETA should be computed after two samples"
    );
}

#[test]
fn test_job_progress_percent_clamping() {
    use oximedia_farm::job_progress::ProgressUpdate;

    let over = ProgressUpdate {
        percent: 150.0,
        phase: "done".to_string(),
        message: "".to_string(),
        timestamp_secs: 0,
    };
    assert_eq!(over.clamped_percent(), 100.0);

    let under = ProgressUpdate {
        percent: -10.0,
        phase: "init".to_string(),
        message: "".to_string(),
        timestamp_secs: 0,
    };
    assert_eq!(under.clamped_percent(), 0.0);
}

// ---------------------------------------------------------------------------
// 7. job_retry_policy
// ---------------------------------------------------------------------------

#[test]
fn test_backoff_strategies() {
    use oximedia_farm::job_retry_policy::BackoffStrategy;
    use std::time::Duration;

    // Fixed
    let fixed = BackoffStrategy::Fixed {
        delay: Duration::from_secs(5),
    };
    for attempt in 0..5 {
        assert_eq!(fixed.delay_for(attempt, 0), Duration::from_secs(5));
    }

    // Exponential with cap
    let exp = BackoffStrategy::Exponential {
        base: Duration::from_secs(1),
        max: Duration::from_secs(60),
    };
    assert_eq!(exp.delay_for(0, 0), Duration::from_secs(1));
    assert_eq!(exp.delay_for(1, 0), Duration::from_secs(2));
    assert_eq!(exp.delay_for(2, 0), Duration::from_secs(4));
    assert_eq!(exp.delay_for(10, 0), Duration::from_secs(60)); // capped

    // Linear
    let linear = BackoffStrategy::Linear {
        base: Duration::from_secs(1),
        step: Duration::from_secs(2),
    };
    assert_eq!(linear.delay_for(0, 0), Duration::from_secs(1));
    assert_eq!(linear.delay_for(1, 0), Duration::from_secs(3));
    assert_eq!(linear.delay_for(2, 0), Duration::from_secs(5));
}

#[test]
fn test_attempt_outcome_retryable() {
    use oximedia_farm::job_retry_policy::AttemptOutcome;

    assert!(AttemptOutcome::TransientFailure.is_retryable());
    assert!(AttemptOutcome::Timeout.is_retryable());
    assert!(!AttemptOutcome::Success.is_retryable());
    assert!(!AttemptOutcome::PermanentFailure.is_retryable());
}

#[test]
fn test_retry_policy_evaluate_and_abandon() {
    use oximedia_farm::job_retry_policy::{
        AbandonReason, AttemptOutcome, BackoffStrategy, JobRetryState, RetryDecision, RetryPolicy,
        RetryPolicyConfig,
    };
    use oximedia_farm::{JobId, WorkerId};
    use std::time::Duration;

    let policy = RetryPolicy::new(RetryPolicyConfig {
        max_attempts: 3,
        backoff: BackoffStrategy::Fixed {
            delay: Duration::from_millis(10),
        },
        blacklist_after_consecutive_failures: Some(2),
        retry_deadline: None,
    });

    let job_id = JobId::new();
    let worker = WorkerId::new("w-test");
    let mut state = JobRetryState::new(job_id);

    // Attempt 1: transient failure → should retry
    // record_attempt_started increments attempts_made before evaluate
    state.record_attempt_started(&worker);
    let decision = policy.evaluate(&mut state, &worker, AttemptOutcome::TransientFailure, 0);
    assert!(
        matches!(decision, RetryDecision::Retry { .. }),
        "attempt 1 should retry: {decision:?}"
    );

    // Attempt 2: transient failure → worker should be blacklisted (threshold=2)
    state.record_attempt_started(&worker);
    let decision2 = policy.evaluate(&mut state, &worker, AttemptOutcome::TransientFailure, 0);
    assert!(
        state.is_blacklisted(&worker),
        "worker should be blacklisted after 2 consecutive failures"
    );
    assert!(
        matches!(decision2, RetryDecision::Retry { .. }),
        "attempt 2 should still retry: {decision2:?}"
    );

    // Attempt 3: max_attempts=3 is reached → abandon
    let worker2 = WorkerId::new("w-fresh");
    state.record_attempt_started(&worker2);
    let decision3 = policy.evaluate(&mut state, &worker2, AttemptOutcome::TransientFailure, 0);
    assert!(
        matches!(
            decision3,
            RetryDecision::Abandon {
                reason: AbandonReason::MaxAttemptsExceeded
            }
        ),
        "should be abandoned on attempt 3: {decision3:?}"
    );

    // Permanent failure → immediately abandons regardless of attempts
    let job_id2 = JobId::new();
    let mut state2 = JobRetryState::new(job_id2);
    state2.record_attempt_started(&worker2);
    let dec_perm = policy.evaluate(&mut state2, &worker2, AttemptOutcome::PermanentFailure, 0);
    assert!(
        matches!(
            dec_perm,
            RetryDecision::Abandon {
                reason: AbandonReason::PermanentFailure
            }
        ),
        "permanent failure should immediately abandon: {dec_perm:?}"
    );
}

// ---------------------------------------------------------------------------
// 8. task_affinity
// ---------------------------------------------------------------------------

#[test]
fn test_task_affinity_hard_rules_pass() {
    use oximedia_farm::task_affinity::{
        AffinityMatcher, AffinityPolicy, AffinityRule, WorkerCapabilities,
    };

    let caps = WorkerCapabilities {
        worker_id: "gpu-node-1".to_string(),
        tags: vec!["gpu".to_string(), "high-mem".to_string()],
        memory_gb: 64.0,
        cpu_cores: 32,
        has_gpu: true,
    };

    let policy = AffinityPolicy {
        rules: vec![
            AffinityRule::RequireGpu,
            AffinityRule::RequireMinMemoryGb(32.0),
            AffinityRule::PreferTag("high-mem".to_string()),
        ],
        fallback_any: false,
    };

    assert!(AffinityMatcher::matches_hard(&caps, &policy.rules));
    let score = AffinityMatcher::score(&caps, &policy.rules);
    assert!(score > 0.0, "soft rule satisfied → positive score");
}

#[test]
fn test_task_affinity_hard_rule_fail() {
    use oximedia_farm::task_affinity::{AffinityMatcher, AffinityRule, WorkerCapabilities};

    let caps = WorkerCapabilities {
        worker_id: "cpu-node".to_string(),
        tags: vec![],
        memory_gb: 8.0,
        cpu_cores: 4,
        has_gpu: false,
    };

    assert!(!AffinityMatcher::matches_hard(
        &caps,
        &[AffinityRule::RequireGpu]
    ));
    assert!(!AffinityMatcher::matches_hard(
        &caps,
        &[AffinityRule::RequireMinMemoryGb(16.0)]
    ));
    assert!(!AffinityMatcher::matches_hard(
        &caps,
        &[AffinityRule::RequireMinCpuCores(8)]
    ));
}

#[test]
fn test_task_affinity_exclude_worker() {
    use oximedia_farm::task_affinity::{AffinityMatcher, AffinityRule, WorkerCapabilities};

    let caps = WorkerCapabilities {
        worker_id: "bad-node".to_string(),
        tags: vec![],
        memory_gb: 128.0,
        cpu_cores: 64,
        has_gpu: true,
    };

    // ExcludeWorker should fail even though GPU and memory are ample
    let rules = vec![
        AffinityRule::RequireGpu,
        AffinityRule::ExcludeWorker("bad-node".to_string()),
    ];
    assert!(!AffinityMatcher::matches_hard(&caps, &rules));

    // Verify is_hard / is_soft classification
    assert!(AffinityRule::RequireGpu.is_hard());
    assert!(AffinityRule::RequireTag("x".to_string()).is_hard());
    assert!(AffinityRule::ExcludeWorker("w".to_string()).is_hard());
    assert!(AffinityRule::PreferTag("x".to_string()).is_soft());
    assert!(AffinityRule::CollocateWith("task-1".to_string()).is_soft());
}

// ---------------------------------------------------------------------------
// 9. dag_viz
// ---------------------------------------------------------------------------

#[test]
fn test_dag_viz_dot_output() {
    use oximedia_farm::dag_viz::{DagViz, NodeState, VizFormat};

    let mut viz = DagViz::new("pipeline-test");
    viz.add_node("ingest", "Ingest", NodeState::Completed);
    viz.add_node("transcode", "Transcode", NodeState::Running);
    viz.add_node("package", "Package", NodeState::Pending);
    viz.add_edge("transcode", "ingest")
        .expect("edge transcode→ingest");
    viz.add_edge("package", "transcode")
        .expect("edge package→transcode");

    let dot = viz.render(VizFormat::Dot).expect("render DOT");
    assert!(dot.contains("digraph"), "DOT output must contain 'digraph'");
    assert!(dot.contains("ingest"));
    assert!(dot.contains("transcode"));
    assert_eq!(viz.node_count(), 3);
    assert_eq!(viz.edge_count(), 2);
}

#[test]
fn test_dag_viz_ascii_and_json_output() {
    use oximedia_farm::dag_viz::{DagViz, NodeState, VizFormat};

    let mut viz = DagViz::new("multi-format-test");
    viz.add_node("a", "A", NodeState::Completed);
    viz.add_node("b", "B", NodeState::Pending);
    viz.add_edge("b", "a").expect("edge b→a");

    let ascii = viz.render(VizFormat::Ascii).expect("render ASCII");
    assert!(!ascii.is_empty(), "ASCII output must be non-empty");

    let json = viz.render(VizFormat::Json).expect("render JSON");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert!(parsed.is_object(), "JSON must be an object");
}

#[test]
fn test_dag_viz_mermaid_output() {
    use oximedia_farm::dag_viz::{DagViz, NodeState, VizFormat};

    let mut viz = DagViz::new("mermaid-test");
    viz.add_node("x", "X", NodeState::Failed);
    viz.add_node("y", "Y", NodeState::Blocked);
    viz.add_edge("y", "x").expect("edge y→x");

    let mermaid = viz.render(VizFormat::Mermaid).expect("render Mermaid");
    assert!(
        mermaid.contains("flowchart"),
        "Mermaid output must contain 'flowchart'"
    );
}

#[test]
fn test_dag_viz_cycle_detection_via_add_edge() {
    use oximedia_farm::dag_viz::{DagViz, NodeState, VizError};

    let mut viz = DagViz::new("cycle-test");
    viz.add_node("a", "A", NodeState::Pending);
    viz.add_node("b", "B", NodeState::Pending);
    viz.add_node("c", "C", NodeState::Pending);

    viz.add_edge("b", "a").expect("b→a ok");
    viz.add_edge("c", "b").expect("c→b ok");

    // a→c would create cycle a→c→b→a
    let result = viz.add_edge("a", "c");
    assert!(
        matches!(result, Err(VizError::CycleDetected { .. })),
        "cycle should be detected: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 10. cost_accounting
// ---------------------------------------------------------------------------

#[test]
fn test_cost_rates_compute() {
    use oximedia_farm::cost_accounting::{CostRates, ResourceUsage};
    use oximedia_farm::{JobId, JobType, WorkerId};
    use std::time::Duration;

    let rates = CostRates::default();
    let usage = ResourceUsage::new(
        JobId::new(),
        JobType::VideoTranscode,
        WorkerId::new("enc-1"),
        Duration::from_secs(120),
        240.0, // cpu_seconds
        0.0,   // gpu_seconds
        60.0,  // memory_gb_seconds
    )
    .expect("valid usage");

    let cost = rates.compute(&usage);
    assert!(cost > 0.0, "cost must be positive");
    assert!(cost >= rates.per_job_overhead, "at least overhead cost");
}

#[test]
fn test_cost_ledger_accumulation() {
    use oximedia_farm::cost_accounting::{CostLedger, ResourceUsage};
    use oximedia_farm::{JobId, JobType, WorkerId};
    use std::time::Duration;

    let mut ledger = CostLedger::new();

    let job1 = JobId::new();
    let job2 = JobId::new();

    let (cost1, _anomaly1) = ledger.record(
        ResourceUsage::new(
            job1,
            JobType::VideoTranscode,
            WorkerId::new("w1"),
            Duration::from_secs(60),
            60.0,
            0.0,
            30.0,
        )
        .expect("valid"),
    );

    let (cost2, _anomaly2) = ledger.record(
        ResourceUsage::new(
            job2,
            JobType::AudioTranscode,
            WorkerId::new("w1"),
            Duration::from_secs(30),
            30.0,
            0.0,
            15.0,
        )
        .expect("valid"),
    );

    assert!(cost1 > 0.0);
    assert!(cost2 > 0.0);
    assert!(cost1 > cost2, "longer job should cost more");

    let total = ledger.cost_for_job(job1) + ledger.cost_for_job(job2);
    assert!(total > 0.0);
}

// ---------------------------------------------------------------------------
// 11. dashboard_api
// ---------------------------------------------------------------------------

#[test]
fn test_dashboard_api_parse_request() {
    use oximedia_farm::dashboard_api::HttpRequest;

    let raw = "GET /api/workers HTTP/1.1\r\nHost: localhost\r\n\r\n";
    let req = HttpRequest::parse(raw).expect("parse request");
    assert_eq!(req.method, "GET");
    assert_eq!(req.path, "/api/workers");
    assert_eq!(req.version, "HTTP/1.1");
    assert!(req.headers.contains_key("host"));

    // Bad request → None
    assert!(HttpRequest::parse("").is_none());
    assert!(HttpRequest::parse("BADREQUEST").is_none());
}

#[test]
fn test_dashboard_api_handler_workers() {
    use oximedia_farm::dashboard_api::{
        DashboardApiHandler, DashboardState, HttpRequest, WorkerSnapshot, WorkerStateRepr,
    };

    let state = DashboardState {
        workers: vec![
            WorkerSnapshot {
                id: "enc-1".to_string(),
                state: WorkerStateRepr::Idle,
                active_tasks: 0,
                capacity: 4,
                address: "127.0.0.1:9001".to_string(),
            },
            WorkerSnapshot {
                id: "enc-2".to_string(),
                state: WorkerStateRepr::Busy,
                active_tasks: 2,
                capacity: 4,
                address: "127.0.0.1:9002".to_string(),
            },
        ],
        jobs: vec![],
    };

    let handler = DashboardApiHandler::with_state(state);
    let req = HttpRequest::parse("GET /api/workers HTTP/1.1\r\n\r\n").expect("parse");
    let resp = handler.handle(&req);

    assert_eq!(resp.status_code, 200);
    assert!(resp.body.contains("enc-1"));
    assert!(resp.body.contains("enc-2"));
}

#[test]
fn test_dashboard_api_handler_404_and_405() {
    use oximedia_farm::dashboard_api::{DashboardApiHandler, HttpRequest};

    let handler = DashboardApiHandler::new();

    // Non-existent path → 404
    let req = HttpRequest::parse("GET /api/unknown HTTP/1.1\r\n\r\n").expect("parse");
    let resp = handler.handle(&req);
    assert_eq!(resp.status_code, 404);

    // POST method → 405
    let req_post = HttpRequest::parse("POST /api/workers HTTP/1.1\r\n\r\n").expect("parse");
    let resp_post = handler.handle(&req_post);
    assert_eq!(resp_post.status_code, 405);
}

// ---------------------------------------------------------------------------
// 12. energy
// ---------------------------------------------------------------------------

#[test]
fn test_energy_scheduler_picks_lowest_tdp() {
    use oximedia_farm::capabilities::WorkerCapabilities;
    use oximedia_farm::energy::EnergyAwareScheduler;

    let sched = EnergyAwareScheduler::new();
    let w1 = WorkerCapabilities::new(1);
    let w2 = WorkerCapabilities::new(2);
    let w3 = WorkerCapabilities::new(3);

    // All under threshold=300; w2 has lowest TDP (75 W)
    let candidates = vec![(&w1, 250.0_f32), (&w2, 75.0), (&w3, 150.0)];
    let winner = sched.preferred_worker(&candidates, 300.0);
    assert_eq!(winner, Some(2));
}

#[test]
fn test_energy_scheduler_threshold_filter() {
    use oximedia_farm::capabilities::WorkerCapabilities;
    use oximedia_farm::energy::EnergyAwareScheduler;

    let sched = EnergyAwareScheduler::new();
    let w1 = WorkerCapabilities::new(10);
    let w2 = WorkerCapabilities::new(20);

    // threshold=200 → w1 (300 W) filtered out; w2 (100 W) wins
    let candidates = vec![(&w1, 300.0_f32), (&w2, 100.0)];
    assert_eq!(sched.preferred_worker(&candidates, 200.0), Some(20));
}

#[test]
fn test_energy_scheduler_empty_returns_none() {
    use oximedia_farm::energy::EnergyAwareScheduler;

    let sched = EnergyAwareScheduler::new();
    let empty: Vec<(&oximedia_farm::capabilities::WorkerCapabilities, f32)> = vec![];
    assert_eq!(sched.preferred_worker(&empty, 1000.0), None);
}

#[test]
fn test_energy_scheduler_graceful_degradation() {
    use oximedia_farm::capabilities::WorkerCapabilities;
    use oximedia_farm::energy::EnergyAwareScheduler;

    let sched = EnergyAwareScheduler::new();
    let w1 = WorkerCapabilities::new(1);
    let w2 = WorkerCapabilities::new(2);

    // Both exceed threshold=50 W → fallback to full set; w2 has lower TDP (100 W)
    let candidates = vec![(&w1, 500.0_f32), (&w2, 100.0)];
    assert_eq!(sched.preferred_worker(&candidates, 50.0), Some(2));
}

// ---------------------------------------------------------------------------
// 13. license_pool
// ---------------------------------------------------------------------------

#[test]
fn test_license_pool_checkout_and_checkin() {
    use oximedia_farm::license_pool::{LicensePool, LicensePoolConfig};
    use oximedia_farm::WorkerId;
    use std::time::Duration;

    let config = LicensePoolConfig::new("hevc-encoder", 2, Duration::from_secs(3600));
    let mut pool = LicensePool::new(config).expect("create pool");

    let w1 = WorkerId::new("w1");
    let w2 = WorkerId::new("w2");
    let w3 = WorkerId::new("w3");

    let tok1 = pool.check_out(w1, None, None).expect("first checkout");
    let tok2 = pool.check_out(w2, None, None).expect("second checkout");
    // Pool exhausted
    assert!(
        pool.check_out(w3, None, None).is_err(),
        "pool should be exhausted"
    );
    assert_eq!(pool.active_count(), 2);

    pool.check_in(tok1.id).expect("return token 1");
    assert_eq!(pool.active_count(), 1);

    let _ = tok2; // keep tok2 alive (not returned)
}

#[test]
fn test_license_manager_multi_product() {
    use oximedia_farm::license_pool::{LicenseManager, LicensePoolConfig};
    use oximedia_farm::WorkerId;
    use std::time::Duration;

    let mut mgr = LicenseManager::new();
    mgr.register(LicensePoolConfig::new(
        "av1-encoder",
        3,
        Duration::from_secs(600),
    ))
    .expect("register av1");
    mgr.register(LicensePoolConfig::new(
        "dolby-vision",
        1,
        Duration::from_secs(600),
    ))
    .expect("register dv");

    let w1 = WorkerId::new("enc-1");
    let tok_av1 = mgr
        .check_out("av1-encoder", w1.clone(), None, None)
        .expect("av1 token");

    let w2 = WorkerId::new("enc-2");
    let tok_dv = mgr
        .check_out("dolby-vision", w2.clone(), None, None)
        .expect("dv token");

    // DV pool is full
    let w3 = WorkerId::new("enc-3");
    assert!(
        mgr.check_out("dolby-vision", w3, None, None).is_err(),
        "DV pool exhausted"
    );

    // Return DV seat; now w4 can get it
    mgr.check_in("dolby-vision", tok_dv.id).expect("return DV");
    let w4 = WorkerId::new("enc-4");
    let _tok_dv2 = mgr
        .check_out("dolby-vision", w4, None, None)
        .expect("DV after return");

    let _ = tok_av1;
    let _ = w1;
}

// ---------------------------------------------------------------------------
// 14. worker_metrics
// ---------------------------------------------------------------------------

#[test]
fn test_worker_metrics_ring_buffer_basics() {
    use oximedia_farm::worker_metrics::RingBuffer;

    let mut rb: RingBuffer<u32> = RingBuffer::new(3).expect("valid capacity");
    assert!(rb.is_empty());

    rb.push(10);
    rb.push(20);
    rb.push(30);
    assert_eq!(rb.len(), 3);

    // Overwrite oldest: [20, 30, 40]
    rb.push(40);
    assert_eq!(rb.len(), 3);

    let items: Vec<u32> = rb.iter().copied().collect();
    assert_eq!(items, vec![20, 30, 40]);

    // Zero capacity errors
    let err: oximedia_farm::Result<RingBuffer<u32>> = RingBuffer::new(0);
    assert!(err.is_err(), "zero-capacity ring buffer must error");
}

#[test]
fn test_worker_metrics_record_operations() {
    use oximedia_farm::worker_metrics::{CompletedJobRecord, WorkerMetricsRecord};
    use oximedia_farm::WorkerId;
    use std::time::Duration;

    let w = WorkerId::new("enc-10");
    let mut rec = WorkerMetricsRecord::new(w, 32, 64).expect("create record");

    rec.record_job_assigned();
    assert_eq!(rec.jobs_assigned, 1);

    rec.record_job_completed(CompletedJobRecord {
        wall_time: Duration::from_secs(45),
        succeeded: true,
        frames_processed: 1800,
    });

    assert_eq!(rec.jobs_succeeded, 1);
    assert_eq!(rec.jobs_failed, 0);
    assert_eq!(rec.total_frames_processed, 1800);

    assert!(rec.completion_rate().is_some());
    // error_rate = 0 failed / (0 failed + 1 succeeded) = Some(0.0)
    assert_eq!(
        rec.error_rate(),
        Some(0.0),
        "zero failures → error_rate is Some(0.0)"
    );
    assert!(rec.mean_processing_time().is_some());
}

#[test]
fn test_worker_metrics_registry_summary() {
    use oximedia_farm::worker_metrics::{CompletedJobRecord, WorkerMetricsRegistry};
    use oximedia_farm::WorkerId;
    use std::time::Duration;

    let mut registry = WorkerMetricsRegistry::new(32, 64).expect("create registry");

    let w1 = WorkerId::new("enc-A");
    let w2 = WorkerId::new("enc-B");
    registry.register(w1.clone()).expect("register w1");
    registry.register(w2.clone()).expect("register w2");
    assert_eq!(registry.worker_count(), 2);

    // Record jobs on w1
    {
        let rec = registry.get_mut(&w1).expect("w1 found");
        rec.record_job_assigned();
        rec.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(30),
            succeeded: true,
            frames_processed: 900,
        });
    }

    let summary = registry.farm_wide_summary();
    assert_eq!(summary.worker_count, 2);
    assert_eq!(summary.total_jobs_assigned, 1);
    assert_eq!(summary.total_jobs_succeeded, 1);
    assert_eq!(summary.total_jobs_failed, 0);
}
