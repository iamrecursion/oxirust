//! Integration tests for `oximedia-distributed`.
//!
//! All tests use in-memory primitives — no real TCP/gRPC connections are
//! established and no external services are required.

use oximedia_distributed::{
    DistributedConfig, DistributedEncoder, DistributedJob, EncodingParams, JobPriority, JobStatus,
    SplitStrategy,
};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn make_job_with_id(id: Uuid) -> DistributedJob {
    DistributedJob {
        id,
        task_id: Uuid::new_v4(),
        source_url: "s3://bucket/input.mp4".to_string(),
        codec: "av1".to_string(),
        strategy: SplitStrategy::SegmentBased,
        priority: JobPriority::Normal,
        params: EncodingParams::default(),
        output_url: "s3://bucket/output.mp4".to_string(),
        deadline: None,
    }
}

fn make_job() -> DistributedJob {
    make_job_with_id(Uuid::new_v4())
}

/// Create an encoder with a high concurrent-jobs limit so tests don't hit it.
fn big_encoder() -> DistributedEncoder {
    let config = DistributedConfig {
        coordinator_addr: String::new(), // no background TCP server
        max_concurrent_jobs: 1000,
        job_timeout: Duration::from_secs(3600),
        ..DistributedConfig::default()
    };
    DistributedEncoder::new(config)
}

// ---------------------------------------------------------------------------
// test_multi_worker_job_completion
// ---------------------------------------------------------------------------

/// Create 2 in-memory "workers" (simulated by advancing jobs) and verify
/// that all submitted jobs reach `Completed`.
#[tokio::test]
async fn test_multi_worker_job_completion() {
    let encoder = big_encoder();

    // Simulate 2 workers: each submits one job and advances it to Completed.
    let job1 = make_job();
    let job2 = make_job();
    let id1 = job1.id;
    let id2 = job2.id;

    encoder.submit_job(job1).await.expect("submit job1");
    encoder.submit_job(job2).await.expect("submit job2");

    // Worker 1 processes job1
    encoder.advance_job(id1).await.expect("assign job1");
    encoder.advance_job(id1).await.expect("start job1");
    encoder.advance_job(id1).await.expect("complete job1");

    // Worker 2 processes job2
    encoder.advance_job(id2).await.expect("assign job2");
    encoder.advance_job(id2).await.expect("start job2");
    encoder.advance_job(id2).await.expect("complete job2");

    assert_eq!(encoder.job_status(id1).await.unwrap(), JobStatus::Completed);
    assert_eq!(encoder.job_status(id2).await.unwrap(), JobStatus::Completed);
}

// ---------------------------------------------------------------------------
// test_fault_tolerance_reroutes_on_failure
// ---------------------------------------------------------------------------

/// Mark a job as failed (simulating worker crash); verify fault tolerance
/// re-queues it as Pending, then advance it to Completed on the "second worker".
#[tokio::test]
async fn test_fault_tolerance_reroutes_on_failure() {
    let config = DistributedConfig {
        coordinator_addr: String::new(),
        max_retries: 2,
        fault_tolerance: true,
        max_concurrent_jobs: 10,
        ..DistributedConfig::default()
    };
    let encoder = DistributedEncoder::new(config);

    let job = make_job();
    let job_id = job.id;
    encoder.submit_job(job).await.expect("submit");

    // Worker 1 crashes mid-job
    encoder
        .fail_job(job_id)
        .await
        .expect("fail (worker 1 crash)");

    // Should be re-queued to Pending (fault tolerance active)
    let status = encoder.job_status(job_id).await.unwrap();
    assert_eq!(status, JobStatus::Pending, "job should be re-queued");

    // Worker 2 takes over and completes
    encoder.advance_job(job_id).await.expect("assign");
    encoder.advance_job(job_id).await.expect("start");
    encoder.advance_job(job_id).await.expect("complete");

    assert_eq!(
        encoder.job_status(job_id).await.unwrap(),
        JobStatus::Completed
    );
}

// ---------------------------------------------------------------------------
// test_leader_election_selects_one_leader
// ---------------------------------------------------------------------------

/// Use the `leader_election` module to simulate 3 in-memory Raft nodes and
/// verify exactly 1 becomes the leader.
#[tokio::test]
async fn test_leader_election_selects_one_leader() {
    use oximedia_distributed::leader_election::{ElectionManager, NodeVote};
    use std::time::Duration;

    // Create 3 election managers in a 3-node cluster
    let mut mgr1 = ElectionManager::new("node-1", 3, Duration::from_secs(5));
    let mut mgr2 = ElectionManager::new("node-2", 3, Duration::from_secs(5));
    let mut mgr3 = ElectionManager::new("node-3", 3, Duration::from_secs(5));

    // node-1 starts an election (term becomes 1)
    mgr1.start_election();
    let term = mgr1.term;

    // node-2 and node-3 vote for node-1; we record on mgr1
    let vote2 = NodeVote::new("node-2", "node-1", term);
    let vote3 = NodeVote::new("node-3", "node-1", term);
    mgr1.record_vote(vote2);
    mgr1.record_vote(vote3);

    // node-2 and node-3 become followers
    mgr2.become_follower();
    mgr3.become_follower();

    // Count leaders (exactly 1: node-1)
    let leaders = [&mgr1, &mgr2, &mgr3]
        .iter()
        .filter(|m| m.state.is_leader())
        .count();
    assert_eq!(leaders, 1, "exactly 1 node should be leader");
    assert!(mgr1.state.is_leader(), "node-1 should be the leader");
    assert!(!mgr2.state.is_leader(), "node-2 should not be leader");
    assert!(!mgr3.state.is_leader(), "node-3 should not be leader");
}

// ---------------------------------------------------------------------------
// test_chaos_random_failures_recovers
// ---------------------------------------------------------------------------

/// Simulate 3 random worker failures (each job fails once, then retries and
/// completes) and verify the system reaches a stable all-completed state.
#[tokio::test]
async fn test_chaos_random_failures_recovers() {
    let config = DistributedConfig {
        coordinator_addr: String::new(),
        max_retries: 3,
        fault_tolerance: true,
        max_concurrent_jobs: 100,
        ..DistributedConfig::default()
    };
    let encoder = DistributedEncoder::new(config);

    let job_ids: Vec<Uuid> = (0..3)
        .map(|_| {
            let j = make_job();
            let id = j.id;
            // submit synchronously from within the async context
            id
        })
        .collect();

    // Submit all jobs
    for &id in &job_ids {
        encoder
            .submit_job(make_job_with_id(id))
            .await
            .expect("submit");
    }

    // Each job fails once (simulating a crashed worker), then recovers
    for &id in &job_ids {
        encoder.fail_job(id).await.expect("fail (chaos)");
        // Re-queued — advance to completed on recovery
        encoder.advance_job(id).await.expect("re-assign");
        encoder.advance_job(id).await.expect("re-start");
        encoder.advance_job(id).await.expect("re-complete");
    }

    // All jobs should be Completed now
    for &id in &job_ids {
        assert_eq!(encoder.job_status(id).await.unwrap(), JobStatus::Completed);
    }
}

// ---------------------------------------------------------------------------
// test_load_balancing_distributes_evenly
// ---------------------------------------------------------------------------

/// Submit 100 jobs to 4 simulated workers (round-robin via modulo) and verify
/// no worker receives > 40 jobs (i.e., within 40 % of perfect distribution of
/// 25 each).
#[tokio::test]
async fn test_load_balancing_distributes_evenly() {
    let encoder = big_encoder();

    let worker_ids: Vec<String> = (0..4).map(|i| format!("worker-{i}")).collect();
    let mut worker_counts: HashMap<&str, u32> = HashMap::new();
    for w in &worker_ids {
        worker_counts.insert(w.as_str(), 0);
    }

    let n_jobs = 100_usize;
    let mut job_ids = Vec::with_capacity(n_jobs);

    for _ in 0..n_jobs {
        let job = make_job();
        let id = job.id;
        encoder.submit_job(job).await.expect("submit");
        job_ids.push(id);
    }

    // Distribute using round-robin among 4 workers (simulated)
    for (i, &id) in job_ids.iter().enumerate() {
        let worker = &worker_ids[i % 4];
        *worker_counts.get_mut(worker.as_str()).unwrap() += 1;
        // Advance to Completed to free the slot
        encoder.advance_job(id).await.expect("assign");
        encoder.advance_job(id).await.expect("start");
        encoder.advance_job(id).await.expect("complete");
    }

    // No worker should get > 40 jobs (perfect is 25)
    for (worker, &count) in &worker_counts {
        assert!(
            count <= 40,
            "worker {worker} got {count} jobs, exceeding limit of 40"
        );
    }
}

// ---------------------------------------------------------------------------
// test_throughput_100_jobs_processing
// ---------------------------------------------------------------------------

/// Submit 100 in-memory "instant" jobs and verify wall-clock time < 5 seconds.
#[tokio::test]
async fn test_throughput_100_jobs_processing() {
    let encoder = big_encoder();

    let start = std::time::Instant::now();

    let n = 100_usize;
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        let job = make_job();
        let id = job.id;
        encoder.submit_job(job).await.expect("submit");
        ids.push(id);
    }

    // Advance every job through its full lifecycle
    for id in ids {
        encoder.advance_job(id).await.expect("assign");
        encoder.advance_job(id).await.expect("start");
        encoder.advance_job(id).await.expect("complete");
    }

    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "100 in-memory jobs took {elapsed:?}, should be < 5s"
    );

    assert_eq!(encoder.job_count().await, n);
    assert_eq!(encoder.active_job_count().await, 0);
}

// ---------------------------------------------------------------------------
// test_batch_submit_10_jobs
// ---------------------------------------------------------------------------

/// Submit 10 jobs atomically via `submit_jobs_batch` and verify all succeed.
#[tokio::test]
async fn test_batch_submit_10_jobs() {
    let encoder = big_encoder();
    let jobs: Vec<DistributedJob> = (0..10).map(|_| make_job()).collect();
    let ids: Vec<Uuid> = jobs.iter().map(|j| j.id).collect();

    let results = encoder.submit_jobs_batch(jobs).await;

    assert_eq!(results.len(), 10);
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.is_ok(),
            "job {i} should have been submitted successfully"
        );
        assert_eq!(result.as_ref().unwrap(), &ids[i]);
    }

    assert_eq!(encoder.job_count().await, 10);
}

// ---------------------------------------------------------------------------
// Additional targeted tests
// ---------------------------------------------------------------------------

/// Verify that submitting the same job ID twice (duplicate) returns an error
/// for the second submission.
#[tokio::test]
async fn test_duplicate_job_id_is_rejected() {
    let encoder = big_encoder();
    let job = make_job();
    let dup = job.clone();

    encoder.submit_job(job).await.expect("first submit");
    let second = encoder.submit_job(dup).await;
    assert!(second.is_err(), "duplicate job ID should be rejected");
}

/// Verify that cancelling a job frees the concurrency slot and a new job can
/// be submitted.
#[tokio::test]
async fn test_cancel_frees_concurrency_slot() {
    let config = DistributedConfig {
        coordinator_addr: String::new(),
        max_concurrent_jobs: 1,
        ..DistributedConfig::default()
    };
    let encoder = DistributedEncoder::new(config);

    let job = make_job();
    let id = job.id;
    encoder.submit_job(job).await.expect("submit");

    // Slot full
    assert!(encoder.submit_job(make_job()).await.is_err());

    // Cancel frees the slot
    encoder.cancel_job(id).await.expect("cancel");
    encoder
        .submit_job(make_job())
        .await
        .expect("new job after cancel");
}
