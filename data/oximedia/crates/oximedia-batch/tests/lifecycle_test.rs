//! Integration tests for the full job lifecycle in `oximedia-batch`.
//!
//! These tests exercise the complete flow from job submission through queueing,
//! status querying, and cancellation via the `BatchEngine` facade (requires
//! the `sqlite` feature and a non-wasm32 target).

#![cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]

use oximedia_batch::{
    job::BatchJob,
    job::BatchOperation,
    operations::FileOperation,
    types::{JobState, RetryPolicy},
    BatchEngine,
};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_db_path() -> NamedTempFile {
    NamedTempFile::new().expect("failed to create tempfile for SQLite DB")
}

/// Create a minimal file-copy job with no retries.
///
/// Disabling retries ensures that if a job fails (e.g. no matching input
/// files) it transitions to `Failed` immediately without waiting for
/// exponential back-off delays.
fn file_copy_job(name: &str) -> BatchJob {
    let mut job = BatchJob::new(
        name.to_string(),
        BatchOperation::FileOp {
            operation: FileOperation::Copy { overwrite: false },
        },
    );
    // Zero retries so the job reaches a terminal state without any delay.
    job.retry = RetryPolicy::none();
    job
}

// ---------------------------------------------------------------------------
// Lifecycle tests (no running engine required)
// ---------------------------------------------------------------------------

/// Verify submit → queued state transitions without starting the execution engine.
///
/// The engine persists job state in SQLite upon submission, so we can
/// verify the complete data path (submit → persist → query) without
/// requiring workers to actually run the job.
#[tokio::test(flavor = "multi_thread")]
async fn test_lifecycle_submit_and_query_status() {
    let db_file = temp_db_path();
    let db_path = db_file
        .path()
        .to_str()
        .expect("tempfile path must be valid UTF-8");

    let engine = BatchEngine::new(db_path, 1).expect("engine must initialise");

    // --- submit ---
    let job = file_copy_job("lifecycle-submit-query");
    let job_id = engine
        .submit_job(job)
        .await
        .expect("job submission must succeed");

    // --- query ---
    let state = engine
        .get_job_status(&job_id)
        .await
        .expect("status must be retrievable after submission");
    assert_eq!(
        state,
        JobState::Queued,
        "job must be in Queued state immediately after submission"
    );
}

/// Verify that multiple jobs submitted in sequence all receive unique IDs
/// and are initially in the `Queued` state.
#[tokio::test(flavor = "multi_thread")]
async fn test_lifecycle_multiple_jobs_are_queued() {
    let db_file = temp_db_path();
    let db_path = db_file
        .path()
        .to_str()
        .expect("tempfile path must be valid UTF-8");

    let engine = BatchEngine::new(db_path, 4).expect("engine must initialise");

    const JOB_COUNT: usize = 5;
    let mut job_ids = Vec::with_capacity(JOB_COUNT);
    for i in 0..JOB_COUNT {
        let job = file_copy_job(&format!("multi-job-{i}"));
        let id = engine
            .submit_job(job)
            .await
            .expect("job submission must succeed");
        job_ids.push(id);
    }

    // All IDs must be unique.
    let unique_ids: std::collections::HashSet<_> = job_ids.iter().collect();
    assert_eq!(unique_ids.len(), JOB_COUNT, "all job IDs must be unique");

    // All must be in Queued state.
    for id in &job_ids {
        let s = engine
            .get_job_status(id)
            .await
            .expect("status must be retrievable");
        assert_eq!(s, JobState::Queued, "job {id} must be Queued");
    }

    // list_jobs must return all submitted jobs.
    let all = engine.list_jobs().expect("list_jobs must succeed");
    assert!(
        all.len() >= JOB_COUNT,
        "list_jobs must include all submitted jobs; found {}",
        all.len()
    );
}

/// Verify that a queued job can be cancelled and is subsequently reported
/// as `Cancelled`.  The cancelled job must still appear in `list_jobs`.
#[tokio::test(flavor = "multi_thread")]
async fn test_lifecycle_cancel_queued_job() {
    let db_file = temp_db_path();
    let db_path = db_file
        .path()
        .to_str()
        .expect("tempfile path must be valid UTF-8");

    let engine = BatchEngine::new(db_path, 1).expect("engine must initialise");

    // --- submit ---
    let job = file_copy_job("cancel-test-job");
    let job_id = engine
        .submit_job(job)
        .await
        .expect("job submission must succeed");

    // Confirm initial Queued state.
    let initial_state = engine
        .get_job_status(&job_id)
        .await
        .expect("status must be retrievable");
    assert_eq!(initial_state, JobState::Queued, "must be Queued initially");

    // --- cancel ---
    engine
        .cancel_job(&job_id)
        .await
        .expect("cancellation must succeed");

    // Verify Cancelled state.
    let cancelled_state = engine
        .get_job_status(&job_id)
        .await
        .expect("status must be retrievable after cancellation");
    assert_eq!(
        cancelled_state,
        JobState::Cancelled,
        "job must be Cancelled after explicit cancellation"
    );

    // Cancelled job must still appear in the database listing.
    let jobs = engine.list_jobs().expect("list_jobs must succeed");
    assert!(
        jobs.iter().any(|j| j.id == job_id),
        "cancelled job must still appear in list_jobs"
    );
}

/// Verify that `BatchEngine::start` and `BatchEngine::stop` complete
/// without hanging when no jobs are present.
#[tokio::test(flavor = "multi_thread")]
async fn test_lifecycle_start_stop_idle_engine() {
    let db_file = temp_db_path();
    let db_path = db_file
        .path()
        .to_str()
        .expect("tempfile path must be valid UTF-8");

    let engine = BatchEngine::new(db_path, 2).expect("engine must initialise");

    // Start the engine.
    engine.start().await.expect("engine must start");

    // Stop immediately (workers are idle but should unblock within ~1 s).
    let stop_result =
        tokio::time::timeout(tokio::time::Duration::from_secs(5), engine.stop()).await;

    assert!(
        stop_result.is_ok(),
        "engine.stop() must complete within 5 s on an idle engine"
    );
    assert!(
        stop_result.expect("timeout must not trigger").is_ok(),
        "engine.stop() must return Ok"
    );
}

/// Full happy-path lifecycle: submit → queue → start engine → cancel →
/// verify cancelled → list includes job.
///
/// We explicitly cancel the job before starting the engine workers to avoid
/// them picking it up (since the job would fail due to missing input files).
/// This tests the complete data path without requiring actual job execution.
#[tokio::test(flavor = "multi_thread")]
async fn test_lifecycle_full_submit_cancel_query() {
    let db_file = temp_db_path();
    let db_path = db_file
        .path()
        .to_str()
        .expect("tempfile path must be valid UTF-8");

    let engine = BatchEngine::new(db_path, 2).expect("engine must initialise");

    // 1. Submit a job.
    let job = file_copy_job("full-lifecycle-job");
    let job_id = engine.submit_job(job).await.expect("submit must succeed");

    // 2. Verify initial state.
    let s = engine.get_job_status(&job_id).await.expect("status query");
    assert_eq!(s, JobState::Queued, "must be Queued after submit");

    // 3. Cancel before starting the engine.
    engine
        .cancel_job(&job_id)
        .await
        .expect("cancel must succeed");

    // 4. Verify cancelled.
    let s = engine.get_job_status(&job_id).await.expect("status query");
    assert_eq!(s, JobState::Cancelled, "must be Cancelled");

    // 5. Job must persist in the DB listing.
    let all = engine.list_jobs().expect("list_jobs must succeed");
    let found = all.iter().any(|j| j.id == job_id);
    assert!(found, "job must appear in list_jobs after cancellation");

    // 6. Start and stop the engine (no jobs to process — idle shutdown).
    engine.start().await.expect("start must succeed");
    let stop_result =
        tokio::time::timeout(tokio::time::Duration::from_secs(5), engine.stop()).await;
    assert!(
        stop_result.is_ok(),
        "engine.stop() must complete within 5 s"
    );
}

/// Verify that cancelling a non-existent job ID returns Ok (idempotent).
#[tokio::test(flavor = "multi_thread")]
async fn test_lifecycle_cancel_unknown_job_is_ok() {
    let db_file = temp_db_path();
    let db_path = db_file
        .path()
        .to_str()
        .expect("tempfile path must be valid UTF-8");

    let engine = BatchEngine::new(db_path, 1).expect("engine must initialise");

    // Cancelling a job that was never submitted must succeed without error.
    let fake_id = oximedia_batch::types::JobId::new();
    let result = engine.cancel_job(&fake_id).await;
    assert!(
        result.is_ok(),
        "cancelling a non-existent job must return Ok"
    );
}

/// Verify that list_jobs returns an empty Vec when no jobs have been submitted.
#[tokio::test(flavor = "multi_thread")]
async fn test_lifecycle_list_jobs_empty_initially() {
    let db_file = temp_db_path();
    let db_path = db_file
        .path()
        .to_str()
        .expect("tempfile path must be valid UTF-8");

    let engine = BatchEngine::new(db_path, 1).expect("engine must initialise");
    let jobs = engine.list_jobs().expect("list_jobs must succeed");
    assert!(
        jobs.is_empty(),
        "no jobs should be listed before any submission"
    );
}
