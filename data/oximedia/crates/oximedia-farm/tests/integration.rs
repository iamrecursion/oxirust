//! End-to-end integration tests for oximedia-farm.
//!
//! These tests exercise the full pipeline:
//!   submit job → assign task to worker → complete task → verify output
//!
//! They require the `sqlite` feature (coordinator, persistence, worker modules).

#![cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]

use oximedia_farm::coordinator::{Job, JobQueue, Task};
use oximedia_farm::output_validator::{OutputValidationRules, OutputValidator};
use oximedia_farm::persistence::Database;
use oximedia_farm::worker_health::{JobOutcome, WorkerHealthScorer};
use oximedia_farm::{JobState, JobType, Priority, TaskState, WorkerId};
use std::io::Write;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn make_queue() -> JobQueue {
    let db = Arc::new(Database::in_memory().expect("in-memory db"));
    JobQueue::new(db, 1000, 100)
}

fn write_temp_file(name: &str, content: &[u8]) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(name);
    let mut f = std::fs::File::create(&path).expect("create temp file");
    f.write_all(content).expect("write temp file");
    path
}

// ---------------------------------------------------------------------------
// Test 1: submit → assign → complete → verify output
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_submit_assign_complete_verify_output() {
    let queue = make_queue().await;

    // ── 1. Submit ─────────────────────────────────────────────────────────────
    let job = Job::new(
        JobType::VideoTranscode,
        "/input/source.mp4".to_string(),
        "/output/result.mp4".to_string(),
        Priority::High,
    );
    let job_id = queue.submit_job(job).await.expect("submit ok");

    // ── 2. Verify initial state ───────────────────────────────────────────────
    let job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(job.state, JobState::Queued);
    assert!(!job.tasks.is_empty(), "job should have at least one task");

    let task_id = job.tasks[0].task_id;

    // ── 3. Assign task to a worker ────────────────────────────────────────────
    let worker_id = WorkerId::new("worker-e2e");
    queue
        .assign_task(task_id, worker_id.clone())
        .await
        .expect("assign ok");

    let assigned_job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(assigned_job.tasks[0].state, TaskState::Assigned);

    // ── 4. Start task ─────────────────────────────────────────────────────────
    queue.start_task(task_id).await.expect("start task ok");
    let running_job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(running_job.tasks[0].state, TaskState::Running);

    // ── 5. Report progress ───────────────────────────────────────────────────
    queue.init_job_progress(job_id, 1);
    queue
        .update_task_progress(task_id, 0.5)
        .await
        .expect("progress ok");
    // Verify progress is tracked
    let progress = queue.get_job_progress(job_id);
    assert!(progress.is_some(), "progress should be tracked");
    let progress = progress.expect("progress");
    assert!(
        progress.percent > 0.0,
        "percent should be positive, got {}",
        progress.percent
    );

    // ── 6. Complete task ──────────────────────────────────────────────────────
    queue.complete_task(task_id).await.expect("complete ok");
    let completed_job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(completed_job.tasks[0].state, TaskState::Completed);

    // ── 7. Aggregate job state ───────────────────────────────────────────────
    queue.update_job_states().await.expect("update states ok");
    let final_job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(
        final_job.state,
        JobState::Completed,
        "job should be Completed"
    );

    // ── 8. Validate output file ───────────────────────────────────────────────
    let output_path = write_temp_file("farm_e2e_output_test.mp4", b"fake encoded mp4 output bytes");
    let rules = OutputValidationRules::with_min_size(8).with_extensions(vec!["mp4".to_string()]);
    let validator = OutputValidator::new(rules);
    assert!(
        validator.validate(&output_path).is_ok(),
        "output validation should pass"
    );

    // Clean up
    let _ = std::fs::remove_file(&output_path);
}

// ---------------------------------------------------------------------------
// Test 2: fault path — task fails and is retried
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_task_failure_and_retry() {
    let queue = make_queue().await;

    let job = Job::new(
        JobType::VideoTranscode,
        "/input/broken.mp4".to_string(),
        "/output/broken.mp4".to_string(),
        Priority::Normal,
    );
    let job_id = queue.submit_job(job).await.expect("submit ok");
    let job = queue.get_job(job_id).await.expect("get job");
    let task_id = job.tasks[0].task_id;

    queue
        .assign_task(task_id, WorkerId::new("worker-retry"))
        .await
        .expect("assign ok");
    queue.start_task(task_id).await.expect("start ok");

    // Fail the task (retryable)
    queue.fail_task(task_id, true).await.expect("fail ok");

    // Task should be reset to Pending for retry
    let retrying_job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(
        retrying_job.tasks[0].state,
        TaskState::Pending,
        "retryable failure should reset task to Pending"
    );
}

// ---------------------------------------------------------------------------
// Test 3: worker goes offline → tasks are reassigned
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_worker_offline_triggers_task_reassignment() {
    let queue = make_queue().await;

    let job = Job::new(
        JobType::VideoTranscode,
        "/input/reassign.mp4".to_string(),
        "/output/reassign.mp4".to_string(),
        Priority::Normal,
    );
    let job_id = queue.submit_job(job).await.expect("submit ok");
    let job = queue.get_job(job_id).await.expect("get job");
    let task_id = job.tasks[0].task_id;

    let worker_id = WorkerId::new("worker-offline");
    queue
        .assign_task(task_id, worker_id.clone())
        .await
        .expect("assign ok");

    // Simulate worker going offline
    queue
        .reassign_worker_tasks(&worker_id)
        .await
        .expect("reassign ok");

    let reset_job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(
        reset_job.tasks[0].state,
        TaskState::Pending,
        "tasks should be reset to Pending when worker goes offline"
    );
}

// ---------------------------------------------------------------------------
// Test 4: worker health scoring integration
// ---------------------------------------------------------------------------

#[test]
fn test_worker_health_integration_quarantines_bad_worker() {
    use oximedia_farm::worker_health::{HealthScoringPolicy, WorkerHealthStatus};
    use std::time::Duration;

    let policy = HealthScoringPolicy {
        ewma_alpha: 0.5,
        quarantine_threshold: 0.4,
        recovery_threshold: 0.7,
        quarantine_cool_down: Duration::from_millis(1),
        max_quarantine_duration: Duration::from_millis(500),
        min_samples_for_quarantine: 3,
    };
    let mut scorer = WorkerHealthScorer::new(policy);

    scorer.register_worker("bad-worker");
    scorer.register_worker("good-worker");

    // bad-worker has repeated permanent failures
    for _ in 0..10 {
        scorer.record_outcome("bad-worker", JobOutcome::PermanentFailure);
    }

    // good-worker is always successful
    for _ in 0..5 {
        scorer.record_outcome("good-worker", JobOutcome::Success);
    }

    let bad_rec = scorer.get_record("bad-worker").expect("record");
    assert!(
        matches!(
            bad_rec.status,
            WorkerHealthStatus::Quarantined | WorkerHealthStatus::Evicted
        ),
        "bad-worker should be quarantined or evicted, got {:?}",
        bad_rec.status
    );

    let good_rec = scorer.get_record("good-worker").expect("record");
    assert_eq!(good_rec.status, WorkerHealthStatus::Healthy);

    let assignable = scorer.assignable_workers();
    assert!(assignable.contains(&"good-worker"));
    assert!(!assignable.contains(&"bad-worker"));
}

// ---------------------------------------------------------------------------
// Test 5: multi-task job — all tasks must complete for job to be Completed
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_multi_task_job_requires_all_tasks_completed() {
    let queue = make_queue().await;

    // Build a job with 2 explicit tasks.
    let mut job = Job::new(
        JobType::MultiOutputTranscode,
        "/input/multi.mp4".to_string(),
        "/output/multi/".to_string(),
        Priority::Normal,
    );

    let task1 = Task::new(job.id, "h264".to_string(), vec![], Priority::Normal);
    let task2 = Task::new(job.id, "vp9".to_string(), vec![], Priority::Normal);
    job.tasks = vec![task1, task2];

    let job_id = queue.submit_job(job).await.expect("submit ok");
    let job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(job.tasks.len(), 2, "should have 2 tasks");

    let t1 = job.tasks[0].task_id;
    let t2 = job.tasks[1].task_id;
    let worker = WorkerId::new("worker-multi");

    // Assign and complete task 1.
    queue
        .assign_task(t1, worker.clone())
        .await
        .expect("assign t1");
    queue.start_task(t1).await.expect("start t1");
    queue.complete_task(t1).await.expect("complete t1");

    // Job should NOT be Completed yet.
    queue.update_job_states().await.expect("update states");
    let intermediate = queue.get_job(job_id).await.expect("get job");
    assert_ne!(
        intermediate.state,
        JobState::Completed,
        "job should not be Completed until both tasks are done"
    );

    // Assign and complete task 2.
    queue.assign_task(t2, worker).await.expect("assign t2");
    queue.start_task(t2).await.expect("start t2");
    queue.complete_task(t2).await.expect("complete t2");

    // Now the job should transition to Completed.
    queue.update_job_states().await.expect("update states");
    let final_job = queue.get_job(job_id).await.expect("get job");
    assert_eq!(
        final_job.state,
        JobState::Completed,
        "job should be Completed after all tasks finish"
    );
}
