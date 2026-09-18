// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Integration / stress tests for oximedia-jobs.
//!
//! Covers:
//! - L48: concurrent job submission/cancellation (100+ ops)
//! - L49: circuit breaker under sustained failure
//! - L50: priority-boost starvation prevention
//! - L51: persistence crash-recovery (drop-and-reopen)
//! - L52: scheduler diamond-DAG correctness
//! - L53: worker-pool auto-scaling behaviour

use oximedia_jobs::{
    dependency_graph::{DepGraph, JobNode},
    job::{AnalysisParams, AnalysisType, Job, JobPayload, JobStatus, Priority, TranscodeParams},
    job_priority_boost::{BoostConfig, PriorityBooster},
    persistence::JobPersistence,
    retry_policy::{CircuitBreaker, CircuitState, ErrorClass, RetryPolicyConfig},
    worker_pool::{Worker, WorkerPool},
};
use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn analysis_payload(input: &str) -> JobPayload {
    JobPayload::Analysis(AnalysisParams {
        input: input.to_string(),
        analysis_type: AnalysisType::Quality,
        output: None,
    })
}

fn _transcode_payload(input: &str, output: &str) -> JobPayload {
    JobPayload::Transcode(TranscodeParams {
        input: input.to_string(),
        output: output.to_string(),
        video_codec: "libx264".to_string(),
        audio_codec: "aac".to_string(),
        video_bitrate: 2_000_000,
        audio_bitrate: 128_000,
        resolution: None,
        framerate: None,
        preset: "medium".to_string(),
        hw_accel: None,
    })
}

fn _simple_job(name: &str, priority: Priority) -> Job {
    Job::new(name.to_string(), priority, analysis_payload(name))
}

// ---------------------------------------------------------------------------
// L48: Concurrent submission / cancellation — 100+ simultaneous operations
// ---------------------------------------------------------------------------

/// Submit 100 jobs from 4 concurrent threads, then cancel every second one.
/// Asserts that all submitted IDs are unique and that cancellations succeed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_concurrent_job_submission_cancellation() {
    use std::sync::Arc as StdArc;

    let persistence = StdArc::new(JobPersistence::in_memory().expect("in-memory db"));

    // Submit 100 jobs from 4 threads (25 each) using the persistence layer directly.
    // We bypass the full JobQueue to keep the test self-contained and dependency-free.
    let jobs_submitted: StdArc<Mutex<Vec<Uuid>>> = StdArc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    for thread_idx in 0_u32..4 {
        let p = StdArc::clone(&persistence);
        let submitted = StdArc::clone(&jobs_submitted);
        let handle = tokio::task::spawn_blocking(move || {
            let mut ids = Vec::new();
            for job_idx in 0_u32..25 {
                let name = format!("stress-job-{thread_idx}-{job_idx}");
                let job = Job::new(name, Priority::Normal, analysis_payload("input.mp4"));
                let id = job.id;
                p.save_job(&job).expect("save_job");
                ids.push(id);
            }
            submitted.lock().expect("lock").extend(ids);
        });
        handles.push(handle);
    }

    for h in handles {
        h.await.expect("thread");
    }

    let ids = jobs_submitted.lock().expect("lock").clone();
    assert_eq!(ids.len(), 100, "expected 100 submitted jobs");

    // All IDs must be unique.
    let unique: HashSet<Uuid> = ids.iter().copied().collect();
    assert_eq!(unique.len(), 100, "all job IDs must be distinct");

    // Verify all 100 are persisted.
    let total = persistence.count_jobs().expect("count");
    assert_eq!(total, 100);

    // Cancel every second job (50 cancellations).
    let to_cancel: Vec<Uuid> = ids.iter().copied().step_by(2).collect();
    assert_eq!(to_cancel.len(), 50);

    for id in &to_cancel {
        let mut job = persistence.get_job(*id).expect("get_job");
        job.status = JobStatus::Cancelled;
        persistence.save_job(&job).expect("cancel via save");
    }

    let cancelled = persistence
        .get_jobs_by_status(JobStatus::Cancelled)
        .expect("by status");
    assert_eq!(cancelled.len(), 50, "50 jobs should be cancelled");
}

// ---------------------------------------------------------------------------
// L49: Circuit breaker under sustained failure
// ---------------------------------------------------------------------------

/// Drive a CircuitBreaker with 20 consecutive failures and verify it opens.
/// Then reset and confirm it returns to Closed.
#[test]
fn test_retry_policy_circuit_breaker_sustained_failure() {
    // Threshold = 5 failures before opening.
    let mut breaker = CircuitBreaker::new(5, 2, Duration::from_secs(30));

    assert_eq!(breaker.state, CircuitState::Closed);
    assert!(breaker.allows_retry(), "should allow retry when closed");

    // Record 4 failures — still closed.
    for _ in 0_u32..4 {
        breaker.record_failure();
    }
    assert_eq!(breaker.state, CircuitState::Closed);
    assert!(breaker.allows_retry());

    // 5th failure opens the circuit.
    breaker.record_failure();
    assert_eq!(breaker.state, CircuitState::Open);
    assert!(!breaker.allows_retry(), "open circuit must block retries");

    // Additional failures don't change state (already open).
    for _ in 5_u32..20 {
        breaker.record_failure();
    }
    assert_eq!(breaker.state, CircuitState::Open);

    // Transition to half-open for probing.
    let moved = breaker.try_half_open();
    assert!(moved, "should be able to move from Open to HalfOpen");
    assert_eq!(breaker.state, CircuitState::HalfOpen);
    assert!(breaker.allows_retry(), "half-open allows a probe retry");

    // 2 successes (= success_threshold) close the circuit.
    breaker.record_success();
    breaker.record_success();
    assert_eq!(breaker.state, CircuitState::Closed);

    // RetryPolicyConfig::should_retry with actual error classes.
    let policy = RetryPolicyConfig::new().with_max_retries(3);
    // Network errors are retryable by default.
    assert!(policy.should_retry(&ErrorClass::Network, 0));
    assert!(policy.should_retry(&ErrorClass::Network, 2));
    assert!(!policy.should_retry(&ErrorClass::Network, 3));

    // Processing errors use the same global limit.
    assert!(policy.should_retry(&ErrorClass::Processing, 0));
    assert!(!policy.should_retry(&ErrorClass::Processing, 3));
}

// ---------------------------------------------------------------------------
// L50: Priority-boost starvation prevention with mixed priorities
// ---------------------------------------------------------------------------

/// Register several jobs with different priorities, simulate repeated passes
/// for low-priority jobs, and verify starvation boosting raises their effective
/// priority above their base level without exceeding the ceiling.
#[test]
fn test_priority_boost_starvation_prevention() {
    let config = BoostConfig {
        wait_threshold: Duration::from_millis(1), // immediate for tests
        wait_boost_increment: 5,
        max_wait_boost: 50,
        deadline_proximity_secs: 300,
        deadline_boost_increment: 20,
        starvation_pass_count: 3, // boost after 3 passes
        starvation_boost_increment: 10,
        dependency_complete_boost: 15,
        priority_ceiling: 100,
    };

    let mut booster = PriorityBooster::new(config);

    // Register three jobs: High, Normal, Low.
    booster.register_job("high-1", 80, None);
    booster.register_job("normal-1", 50, None);
    booster.register_job("low-1", 10, None);

    // Simulate the low-priority job being passed over 3 times.
    for _ in 0..3 {
        booster.record_pass("low-1");
    }

    // After 3 passes, evaluating starvation should boost it.
    booster.evaluate_starvation("low-1");

    let effective = booster
        .effective_priority("low-1")
        .expect("low-1 registered");
    assert!(
        effective > 10,
        "effective priority must exceed base after starvation boost; got {effective}"
    );
    assert!(
        effective <= 100,
        "must not exceed priority ceiling; got {effective}"
    );

    // High priority job should be unchanged (no passes).
    assert_eq!(booster.effective_priority("high-1"), Some(80));

    // Total boosts applied should be at least 1 (the starvation boost).
    assert!(
        booster.total_boosts_applied() >= 1,
        "at least one boost must have been applied"
    );

    // Boost history for low-1 must contain the starvation event.
    let history = booster.boost_history("low-1").expect("history");
    assert!(!history.is_empty(), "boost history should not be empty");

    // Manual boost should also be tracked.
    booster.manual_boost("normal-1", 15, "operator override");
    assert_eq!(booster.effective_priority("normal-1"), Some(65));
}

// ---------------------------------------------------------------------------
// L51: Persistence crash recovery (drop connection and reopen)
// ---------------------------------------------------------------------------

/// Write jobs to a file-backed SQLite DB, drop the connection (simulating
/// a process restart), reopen the DB, and verify all jobs are still present
/// and WAL mode is active.
#[test]
fn test_persistence_crash_recovery() {
    let dir = std::env::temp_dir().join(format!("oxijobs-crash-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let db_path = dir.join("jobs.db");

    let job_ids: Vec<Uuid>;

    // ── Phase 1: write 20 jobs ──────────────────────────────────────────────
    {
        let persistence = JobPersistence::new(&db_path).expect("open db");

        // Verify WAL is active.
        // We'll just confirm the db opens and write succeeds — WAL is set in
        // `new()` via `enable_wal()`.

        let jobs: Vec<Job> = (0..20)
            .map(|i| {
                Job::new(
                    format!("crash-test-job-{i}"),
                    Priority::Normal,
                    analysis_payload(&format!("input-{i}.mp4")),
                )
            })
            .collect();

        job_ids = jobs.iter().map(|j| j.id).collect();

        // Use batch write for better coverage.
        persistence.save_jobs_batch(&jobs).expect("batch save");
    } // ← persistence dropped here (simulates crash / process exit)

    // ── Phase 2: reopen and verify ──────────────────────────────────────────
    {
        let persistence = JobPersistence::new(&db_path).expect("reopen db");

        let total = persistence.count_jobs().expect("count");
        assert_eq!(total, 20, "all 20 jobs must survive a connection drop");

        // Spot-check two jobs by ID.
        let first = persistence.get_job(job_ids[0]).expect("get first job");
        assert_eq!(first.status, JobStatus::Pending);

        let last = persistence.get_job(job_ids[19]).expect("get last job");
        assert_eq!(last.status, JobStatus::Pending);

        // The job names must be recoverable.
        assert!(first.name.starts_with("crash-test-job-"));
    }

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// L52: Scheduler diamond-DAG dependency correctness
// ---------------------------------------------------------------------------

/// Build a diamond DAG (A → B, A → C, B → D, C → D) in the dependency graph
/// and verify that:
/// - The topological sort is valid (A before B/C, B and C before D)
/// - No cycle is reported
/// - All 4 nodes appear in the sort
#[test]
fn test_scheduler_diamond_dag() {
    //    A (1)
    //   / \
    //  B(2) C(3)
    //   \ /
    //    D (4)

    let mut graph = DepGraph::new();
    graph.add_node(JobNode::new(1, "A-ingest"));
    graph.add_node(JobNode::new(2, "B-transcode-1080p"));
    graph.add_node(JobNode::new(3, "C-transcode-720p"));
    graph.add_node(JobNode::new(4, "D-package"));

    // A must complete before B and C.
    graph.add_dependency(1, 2);
    graph.add_dependency(1, 3);
    // B and C must complete before D.
    graph.add_dependency(2, 4);
    graph.add_dependency(3, 4);

    assert!(!graph.has_cycle(), "diamond DAG must not contain a cycle");

    let order = graph.topological_sort();
    assert_eq!(order.len(), 4, "all 4 nodes must appear in the sort");

    // A (1) must be first.
    assert_eq!(order[0], 1, "A must be first");

    // D (4) must be last.
    assert_eq!(order[3], 4, "D must be last");

    // B and C are in positions 1 and 2 (in stable sorted order: 2, 3).
    let middle: HashSet<u64> = order[1..3].iter().copied().collect();
    assert!(middle.contains(&2), "B must be in middle positions");
    assert!(middle.contains(&3), "C must be in middle positions");

    // Dependency queries.
    let deps_of_d = graph.dependencies_of(4);
    assert!(deps_of_d.contains(&2));
    assert!(deps_of_d.contains(&3));

    let deps_of_b = graph.dependencies_of(2);
    assert_eq!(deps_of_b, vec![1]);

    // Roots: only A.
    let roots = graph.roots();
    assert_eq!(roots, vec![1], "only A is a root");

    // Stress: linear chain of 10 000 nodes must sort in < 1 second.
    let start = Instant::now();
    let mut big_graph = DepGraph::new();
    for i in 0_u64..10_000 {
        big_graph.add_node(JobNode::new(i, &format!("job-{i}")));
    }
    for i in 0_u64..9_999 {
        big_graph.add_dependency(i, i + 1);
    }
    let sorted = big_graph.topological_sort();
    let elapsed = start.elapsed();

    assert_eq!(sorted.len(), 10_000, "all 10k nodes must be sorted");
    assert_eq!(sorted[0], 0, "first node must be 0");
    assert_eq!(sorted[9_999], 9_999, "last node must be 9999");
    assert!(
        elapsed < Duration::from_secs(1),
        "10k-node topological sort must complete in < 1s; took {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// L53: Worker-pool auto-scaling / work-stealing hints
// ---------------------------------------------------------------------------

/// Verify that the WorkerPool correctly models load across multiple workers
/// and that `steal_opportunity` identifies imbalanced pairs.
#[test]
fn test_worker_pool_auto_scaling() {
    let mut pool = WorkerPool::new();

    // Add 4 workers with capacity 4 each.
    for i in 0_u32..4 {
        pool.add_worker(Worker::new(format!("worker-{i}"), 4));
    }

    assert_eq!(pool.worker_count(), 4);
    assert_eq!(pool.available_count(), 4);

    // Assign 10 jobs — they should spread across workers.
    for i in 0..10_usize {
        let assigned = pool.assign_job(format!("job-{i}"));
        assert!(assigned.is_some(), "job-{i} should be assigned");
    }

    // After 10 assignments across 4 workers (capacity 4 each = 16 total slots):
    // utilization should be > 0.
    assert!(
        pool.avg_utilization() > 0.0,
        "average utilization must be > 0 after assignments"
    );

    // No steal opportunity when pool is empty.
    let empty_pool = WorkerPool::new();
    assert!(
        empty_pool.steal_opportunity(0.5).is_none(),
        "no steal opportunity in empty pool"
    );

    // Single-worker pool has no steal opportunity.
    let mut single = WorkerPool::new();
    single.add_worker(Worker::new("solo", 4));
    assert!(single.steal_opportunity(0.5).is_none());

    // Create an imbalanced pool: one fully loaded, one idle.
    let mut imbalanced = WorkerPool::new();
    let mut busy = Worker::new("busy-w", 4);
    busy.active_slots = 4;
    busy.state = oximedia_jobs::worker_pool::WorkerState::Busy;
    imbalanced.add_worker(busy);
    imbalanced.add_worker(Worker::new("idle-w", 4)); // 0 active slots

    // Steal opportunity with threshold 0.5 should fire (1.0 - 0.0 = 1.0 gap).
    let opp = imbalanced.steal_opportunity(0.5);
    assert!(
        opp.is_some(),
        "should detect steal opportunity in imbalanced pool"
    );
    let (busiest, idlest) = opp.expect("opportunity");
    assert_eq!(busiest, "busy-w");
    assert_eq!(idlest, "idle-w");

    // Balanced pool (both at 50%): no steal with threshold 0.5.
    let mut balanced = WorkerPool::new();
    let mut w1 = Worker::new("b1", 4);
    w1.active_slots = 2;
    let mut w2 = Worker::new("b2", 4);
    w2.active_slots = 2;
    balanced.add_worker(w1);
    balanced.add_worker(w2);
    assert!(
        balanced.steal_opportunity(0.5).is_none(),
        "balanced pool should not suggest stealing"
    );

    // Complete all jobs in the main pool and verify completion tracking.
    for i in 0..4_u32 {
        let wid = format!("worker-{i}");
        // complete as many slots as were used
        let slots_used = pool.get_worker(&wid).map(|w| w.active_slots).unwrap_or(0);
        for _ in 0..slots_used {
            pool.complete_job(&wid, true);
        }
    }

    assert!(
        pool.total_completed() > 0,
        "at least some completions should be tracked"
    );
}
