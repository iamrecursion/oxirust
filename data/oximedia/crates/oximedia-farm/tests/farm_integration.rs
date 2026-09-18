//! End-to-end farm integration tests.
//!
//! These tests exercise:
//!   - Complete job lifecycle: submit → schedule → dispatch → complete → cleanup
//!   - `AutoScaler` scale-up and scale-down behaviour
//!   - `FarmPriorityQueue` / `PriorityJobQueue` ordering under load
//!   - Cloud storage mock round-trips
//!   - Worker health scoring integration
//!
//! All tests are feature-independent (no `sqlite` feature required).

use oximedia_farm::auto_scaler::{AutoScaleConfig, AutoScaler, ScaleDecision};
use oximedia_farm::cloud_storage::{
    CloudCredentials, CloudProvider, CloudStorageClient, CloudStorageConfig, MockCloudBackend,
};
use oximedia_farm::job_queue::{FarmJob, JobPriority, JobQueue};
use oximedia_farm::priority_queue::{FarmPriorityQueue, PriorityEntry, PriorityEntryBuilder};
use oximedia_farm::worker_health::{
    HealthScoringPolicy, JobOutcome, WorkerHealthScorer, WorkerHealthStatus,
};
use oximedia_farm::{JobId, Priority, TaskId, WorkerId};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── helpers ───────────────────────────────────────────────────────────────────

fn default_autoscale_config() -> AutoScaleConfig {
    AutoScaleConfig {
        min_workers: 2,
        max_workers: 20,
        scale_up_threshold: 3.0,
        scale_down_threshold: 0.5,
        cooldown_secs: 60,
        scale_up_step: 2,
        scale_down_step: 1,
    }
}

fn make_entry(job_id: &str, priority: u32) -> PriorityEntry {
    PriorityEntry {
        priority,
        submitted_at: Instant::now(),
        deadline: None,
        job_id: job_id.to_string(),
        estimated_duration: Duration::from_secs(30),
    }
}

fn make_farm_job(id: &str, priority: JobPriority) -> FarmJob {
    FarmJob::new(id, id, priority, None, 60)
}

fn make_mock_cloud_client(
    prefix: &str,
) -> (
    CloudStorageClient,
    Arc<parking_lot::Mutex<MockCloudBackend>>,
) {
    let config = CloudStorageConfig {
        provider: CloudProvider::S3 {
            region: "us-east-1".into(),
            endpoint: None,
        },
        bucket: "farm-test-bucket".into(),
        prefix: prefix.to_string(),
        credentials: CloudCredentials::AccessKey {
            id: "AKIATESTKEY".into(),
            secret: "testsecretkey".into(),
        },
    };
    let backend = Arc::new(parking_lot::Mutex::new(MockCloudBackend::new()));
    let client = CloudStorageClient::with_mock(config, Arc::clone(&backend));
    (client, backend)
}

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("oximedia_farm_integ_{name}"));
    p
}

fn write_temp(name: &str, content: &[u8]) -> std::path::PathBuf {
    let p = temp_path(name);
    std::fs::write(&p, content).expect("write temp");
    p
}

fn health_policy() -> HealthScoringPolicy {
    HealthScoringPolicy {
        ewma_alpha: 0.5,
        quarantine_threshold: 0.35,
        recovery_threshold: 0.70,
        quarantine_cool_down: Duration::from_millis(1),
        max_quarantine_duration: Duration::from_millis(500),
        min_samples_for_quarantine: 3,
    }
}

// ── Job ID / Task ID / Worker ID lifecycle ────────────────────────────────────

#[test]
fn test_job_id_uniqueness_and_display() {
    let ids: Vec<JobId> = (0..100).map(|_| JobId::new()).collect();
    let set: std::collections::HashSet<String> = ids.iter().map(|id| id.to_string()).collect();
    assert_eq!(set.len(), 100, "all 100 JobIds should be unique");
}

#[test]
fn test_task_id_uniqueness() {
    let ids: Vec<TaskId> = (0..50).map(|_| TaskId::new()).collect();
    let set: std::collections::HashSet<String> = ids.iter().map(|id| id.to_string()).collect();
    assert_eq!(set.len(), 50);
}

#[test]
fn test_worker_id_round_trip() {
    let wid = WorkerId::new("farm-worker-42");
    assert_eq!(wid.as_str(), "farm-worker-42");
    assert_eq!(wid.to_string(), "farm-worker-42");
}

// ── Priority enum ─────────────────────────────────────────────────────────────

#[test]
fn test_priority_ordering_all_levels() {
    assert!(Priority::Critical > Priority::High);
    assert!(Priority::High > Priority::Normal);
    assert!(Priority::Normal > Priority::Low);
    assert!(Priority::Low < Priority::Critical);
}

#[test]
fn test_priority_i32_conversions() {
    for v in 0i32..=3 {
        let p = Priority::try_from(v).expect("valid priority");
        assert_eq!(i32::from(p), v);
    }
    assert!(Priority::try_from(4).is_err());
    assert!(Priority::try_from(-1).is_err());
}

// ── FarmPriorityQueue ordering ────────────────────────────────────────────────

#[test]
fn test_priority_queue_dequeues_highest_first() {
    let mut q = FarmPriorityQueue::new();
    q.push(make_entry("low", 1));
    q.push(make_entry("critical", 100));
    q.push(make_entry("mid", 50));

    assert_eq!(q.pop().expect("pop").job_id, "critical");
    assert_eq!(q.pop().expect("pop").job_id, "mid");
    assert_eq!(q.pop().expect("pop").job_id, "low");
}

#[test]
fn test_priority_queue_fifo_within_same_priority() {
    let mut q = FarmPriorityQueue::new();
    let e1 = make_entry("first", 5);
    std::thread::sleep(Duration::from_millis(2));
    let e2 = make_entry("second", 5);
    q.push(e1);
    q.push(e2);
    assert_eq!(q.pop().expect("pop").job_id, "first");
    assert_eq!(q.pop().expect("pop").job_id, "second");
}

#[test]
fn test_priority_queue_capacity_enforced() {
    let mut q = FarmPriorityQueue::with_capacity(3);
    assert!(q.push(make_entry("j1", 1)));
    assert!(q.push(make_entry("j2", 2)));
    assert!(q.push(make_entry("j3", 3)));
    assert!(!q.push(make_entry("j4", 4))); // rejected
    assert_eq!(q.len(), 3);
}

#[test]
fn test_priority_queue_builder() {
    let entry = PriorityEntryBuilder::new("builder-job", 42)
        .estimated_duration(Duration::from_secs(120))
        .build();
    assert_eq!(entry.job_id, "builder-job");
    assert_eq!(entry.priority, 42);
    assert_eq!(entry.estimated_duration, Duration::from_secs(120));
}

#[test]
fn test_priority_queue_drain_expired_leaves_valid() {
    let mut q = FarmPriorityQueue::new();
    let mut expired = make_entry("expired", 5);
    expired.deadline = Some(Instant::now() - Duration::from_secs(1));
    let valid = make_entry("valid", 3);
    q.push(expired);
    q.push(valid);

    let drained = q.drain_expired();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].job_id, "expired");
    assert_eq!(q.len(), 1);
    assert_eq!(q.pop().expect("pop").job_id, "valid");
}

// ── FarmJob / PriorityJobQueue (job_queue module) ─────────────────────────────

#[test]
fn test_farm_job_queue_ordering() {
    let mut q = JobQueue::new();
    q.enqueue(make_farm_job("low-job", JobPriority::Low));
    q.enqueue(make_farm_job("urgent-job", JobPriority::Urgent));
    q.enqueue(make_farm_job("normal-job", JobPriority::Normal));
    q.enqueue(make_farm_job("high-job", JobPriority::High));

    assert_eq!(q.dequeue().expect("dequeue").job_id, "urgent-job");
    assert_eq!(q.dequeue().expect("dequeue").job_id, "high-job");
    assert_eq!(q.dequeue().expect("dequeue").job_id, "normal-job");
    assert_eq!(q.dequeue().expect("dequeue").job_id, "low-job");
}

#[test]
fn test_farm_job_queue_purge_expired() {
    let mut q = JobQueue::new();
    // TTL of 1 ns — will be expired almost immediately
    let expired = FarmJob::new(
        "expired",
        "Expired Job",
        JobPriority::High,
        Some(Duration::from_nanos(1)),
        10,
    );
    // Allow TTL to expire
    std::thread::sleep(Duration::from_millis(2));
    q.enqueue(expired);
    q.enqueue(make_farm_job("alive", JobPriority::Normal));

    let purged = q.purge_expired();
    assert_eq!(purged, 1, "exactly one expired job should be purged");
    assert_eq!(q.count(), 1);
    assert_eq!(q.dequeue().expect("alive job").job_id, "alive");
}

#[test]
fn test_farm_job_is_expired_with_no_ttl() {
    let job = make_farm_job("no-ttl", JobPriority::Normal);
    assert!(!job.is_expired(), "job with no TTL should never expire");
}

#[test]
fn test_farm_job_age_increases() {
    let job = make_farm_job("age-test", JobPriority::Low);
    std::thread::sleep(Duration::from_millis(5));
    assert!(job.age() >= Duration::from_millis(5));
}

// ── AutoScaler scale-up / scale-down lifecycle ────────────────────────────────

#[test]
fn test_autoscaler_scale_up_when_queue_deep() {
    let mut scaler = AutoScaler::new(default_autoscale_config(), 4);
    // 4 workers, 20 jobs → depth_per_worker = 5.0 > 3.0
    let decision = scaler.evaluate(20, 1000);
    assert_eq!(decision, ScaleDecision::ScaleUp(2));
    scaler.apply_decision(&decision, 1000);
    assert_eq!(scaler.current_workers(), 6);
}

#[test]
fn test_autoscaler_scale_down_when_queue_shallow() {
    let mut scaler = AutoScaler::new(default_autoscale_config(), 8);
    // 8 workers, 2 jobs → depth_per_worker = 0.25 < 0.5
    let decision = scaler.evaluate(2, 1000);
    assert_eq!(decision, ScaleDecision::ScaleDown(1));
    scaler.apply_decision(&decision, 1000);
    assert_eq!(scaler.current_workers(), 7);
}

#[test]
fn test_autoscaler_no_change_within_thresholds() {
    let scaler = AutoScaler::new(default_autoscale_config(), 4);
    // 4 workers, 8 jobs → depth_per_worker = 2.0 — between 0.5 and 3.0
    let decision = scaler.evaluate(8, 1000);
    assert_eq!(decision, ScaleDecision::NoChange);
}

#[test]
fn test_autoscaler_cooldown_prevents_thrashing() {
    let mut scaler = AutoScaler::new(default_autoscale_config(), 4);
    let d1 = scaler.evaluate(40, 100);
    assert_eq!(d1, ScaleDecision::ScaleUp(2));
    scaler.apply_decision(&d1, 100);
    // 30 seconds later — still in 60-second cooldown
    assert_eq!(scaler.evaluate(40, 130), ScaleDecision::NoChange);
    // 60 seconds later — cooldown elapsed
    assert_eq!(scaler.evaluate(40, 160), ScaleDecision::ScaleUp(2));
}

#[test]
fn test_autoscaler_never_exceeds_max_workers() {
    let config = AutoScaleConfig {
        min_workers: 1,
        max_workers: 5,
        scale_up_threshold: 1.0,
        scale_up_step: 10, // large step
        ..Default::default()
    };
    let mut scaler = AutoScaler::new(config, 4);
    let d = scaler.evaluate(100, 1000);
    scaler.apply_decision(&d, 1000);
    assert_eq!(scaler.current_workers(), 5, "should clamp at max_workers=5");
}

#[test]
fn test_autoscaler_never_goes_below_min_workers() {
    let config = AutoScaleConfig {
        min_workers: 3,
        max_workers: 20,
        scale_down_threshold: 0.9,
        scale_down_step: 100, // large step
        ..Default::default()
    };
    let mut scaler = AutoScaler::new(config, 5);
    let d = scaler.evaluate(0, 1000);
    scaler.apply_decision(&d, 1000);
    assert_eq!(scaler.current_workers(), 3, "should clamp at min_workers=3");
}

#[test]
fn test_autoscaler_simulate_burst_and_drain() {
    let mut scaler = AutoScaler::new(default_autoscale_config(), 2);

    // Burst: queue fills up — start at t=1000 so last_scale_time=0 is well past cooldown
    let mut now = 1000u64;
    let d = scaler.evaluate(50, now);
    assert_eq!(d, ScaleDecision::ScaleUp(2));
    scaler.apply_decision(&d, now);
    assert_eq!(scaler.current_workers(), 4);

    // Another burst after cooldown
    now += 120;
    let d = scaler.evaluate(50, now);
    assert_eq!(d, ScaleDecision::ScaleUp(2));
    scaler.apply_decision(&d, now);
    assert_eq!(scaler.current_workers(), 6);

    // Queue drains
    now += 120;
    let d = scaler.evaluate(1, now);
    assert_eq!(d, ScaleDecision::ScaleDown(1));
    scaler.apply_decision(&d, now);
    assert_eq!(scaler.current_workers(), 5);

    // Keep draining
    now += 120;
    let d = scaler.evaluate(0, now);
    assert_eq!(d, ScaleDecision::ScaleDown(1));
    scaler.apply_decision(&d, now);
    assert_eq!(scaler.current_workers(), 4);
}

// ── Worker health scoring integration ─────────────────────────────────────────

#[test]
fn test_health_scorer_quarantines_failing_worker() {
    let mut scorer = WorkerHealthScorer::new(health_policy());
    scorer.register_worker("bad-worker");
    scorer.register_worker("healthy-worker");

    for _ in 0..10 {
        scorer.record_outcome("bad-worker", JobOutcome::PermanentFailure);
    }
    for _ in 0..5 {
        scorer.record_outcome("healthy-worker", JobOutcome::Success);
    }

    let bad = scorer.get_record("bad-worker").expect("record");
    assert!(
        matches!(
            bad.status,
            WorkerHealthStatus::Quarantined | WorkerHealthStatus::Evicted
        ),
        "bad-worker expected Quarantined/Evicted, got {:?}",
        bad.status
    );

    let good = scorer.get_record("healthy-worker").expect("record");
    assert_eq!(good.status, WorkerHealthStatus::Healthy);

    let assignable = scorer.assignable_workers();
    assert!(assignable.contains(&"healthy-worker"));
    assert!(!assignable.contains(&"bad-worker"));
}

#[test]
fn test_health_scorer_timeout_penalty() {
    let mut scorer = WorkerHealthScorer::new(health_policy());
    scorer.register_worker("timeout-worker");

    // 10 timeouts should degrade the score significantly
    for _ in 0..10 {
        scorer.record_outcome("timeout-worker", JobOutcome::Timeout);
    }

    let record = scorer.get_record("timeout-worker").expect("record");
    assert!(
        record.score < 0.8,
        "health score should be degraded by timeouts, got {}",
        record.score
    );
}

// ── Cloud storage mock round-trips ────────────────────────────────────────────

#[tokio::test]
async fn test_cloud_full_upload_download_cycle() {
    let (client, _backend) = make_mock_cloud_client("");
    let content = b"farm-integration-test-payload-bytes";
    let src = write_temp("cloud_cycle_src.bin", content);
    let dst = temp_path("cloud_cycle_dst.bin");

    let url = client.upload(&src, "media/test.mp4").await.expect("upload");
    assert!(!url.is_empty());

    client
        .download("media/test.mp4", &dst)
        .await
        .expect("download");
    let downloaded = std::fs::read(&dst).expect("read dst");
    assert_eq!(&downloaded, content);

    // cleanup
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
}

#[tokio::test]
async fn test_cloud_list_after_multiple_uploads() {
    let (client, _backend) = make_mock_cloud_client("jobs/");
    for i in 0..5 {
        let content = format!("content-{i}").into_bytes();
        let path = write_temp(&format!("cloud_list_{i}.bin"), &content);
        client
            .upload(&path, &format!("output_{i}.mp4"))
            .await
            .expect("upload");
        let _ = std::fs::remove_file(&path);
    }

    let objects = client.list("").await.expect("list");
    assert_eq!(objects.len(), 5, "should list exactly 5 uploaded objects");

    // Each object should have non-zero size and a non-empty etag
    for obj in &objects {
        assert!(obj.size > 0, "size should be > 0 for key {}", obj.key);
        assert!(!obj.etag.is_empty(), "etag should not be empty");
    }
}

#[tokio::test]
async fn test_cloud_delete_removes_from_listing() {
    let (client, _backend) = make_mock_cloud_client("");
    let path = write_temp("cloud_del.bin", b"to delete");
    client.upload(&path, "delete_me.mp4").await.expect("upload");
    let _ = std::fs::remove_file(&path);

    let before = client.list("").await.expect("list before");
    assert_eq!(before.len(), 1);

    client.delete("delete_me.mp4").await.expect("delete");

    let after = client.list("").await.expect("list after");
    assert!(after.is_empty(), "listing should be empty after deletion");
}

#[tokio::test]
async fn test_cloud_download_missing_key_returns_not_found() {
    let (client, _backend) = make_mock_cloud_client("");
    let dst = temp_path("cloud_missing.bin");
    let result = client.download("does_not_exist.mp4", &dst).await;
    assert!(result.is_err(), "expected error for missing key");
    let err = result.expect_err("error");
    assert!(
        matches!(err, oximedia_farm::FarmError::NotFound(_)),
        "expected NotFound error, got: {err:?}"
    );
}

// ── Combined scenario: submit → route to cloud → complete ─────────────────────

#[tokio::test]
async fn test_job_lifecycle_with_cloud_storage() {
    // Simulate a full job lifecycle where the job's input is uploaded to cloud
    // storage and the output is downloaded after completion.

    let (client, backend) = make_mock_cloud_client("jobs/");

    // 1. Upload "input" to cloud
    let input_content = b"raw media bytes for transcoding";
    let input_path = write_temp("lifecycle_input.mp4", input_content);
    let upload_url = client
        .upload(&input_path, "input.mp4")
        .await
        .expect("upload input");
    assert!(upload_url.contains("input.mp4"));
    let _ = std::fs::remove_file(&input_path);

    // 2. Verify the input is in cloud storage
    let objects = client.list("").await.expect("list");
    assert_eq!(objects.len(), 1);
    assert!(objects[0].key.contains("input.mp4"));

    // 3. Simulate processing: "transcode" produces an output
    let output_content = b"transcoded output bytes h264";
    backend
        .lock()
        .put("jobs/output.mp4", output_content.as_slice());

    // 4. Download the output
    let out_path = temp_path("lifecycle_output.mp4");
    client
        .download("output.mp4", &out_path)
        .await
        .expect("download output");

    let downloaded = std::fs::read(&out_path).expect("read output");
    assert_eq!(&downloaded, output_content);

    // 5. Clean up
    client.delete("input.mp4").await.expect("delete input");
    let final_objects = client.list("").await.expect("list after cleanup");
    assert_eq!(final_objects.len(), 1, "only output should remain");
    assert!(final_objects[0].key.contains("output.mp4"));

    let _ = std::fs::remove_file(&out_path);
}

// ── Concurrent job ID generation (thread safety) ──────────────────────────────

#[test]
fn test_job_id_generation_thread_safety() {
    use std::sync::Mutex;

    let ids = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    for _ in 0..8 {
        let ids = Arc::clone(&ids);
        let handle = std::thread::spawn(move || {
            let local: Vec<JobId> = (0..100).map(|_| JobId::new()).collect();
            ids.lock().expect("lock").extend(local);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("thread join");
    }

    let guard = ids.lock().expect("lock");
    let set: std::collections::HashSet<String> = guard.iter().map(|id| id.to_string()).collect();
    assert_eq!(
        set.len(),
        800,
        "all 800 concurrently generated JobIds should be unique"
    );
}

// ── JobType display ───────────────────────────────────────────────────────────

#[test]
fn test_job_type_display_variants() {
    use oximedia_farm::JobType;
    assert_eq!(JobType::VideoTranscode.to_string(), "VideoTranscode");
    assert_eq!(JobType::AudioTranscode.to_string(), "AudioTranscode");
    assert_eq!(
        JobType::ThumbnailGeneration.to_string(),
        "ThumbnailGeneration"
    );
    assert_eq!(JobType::QcValidation.to_string(), "QcValidation");
    assert_eq!(JobType::MediaAnalysis.to_string(), "MediaAnalysis");
    assert_eq!(
        JobType::ContentFingerprinting.to_string(),
        "ContentFingerprinting"
    );
    assert_eq!(
        JobType::MultiOutputTranscode.to_string(),
        "MultiOutputTranscode"
    );
}

// ── JobState / TaskState / WorkerState display ────────────────────────────────

#[test]
fn test_state_display_coverage() {
    use oximedia_farm::{JobState, TaskState, WorkerState};
    // Job states
    for (state, expected) in [
        (JobState::Pending, "Pending"),
        (JobState::Queued, "Queued"),
        (JobState::Running, "Running"),
        (JobState::Completed, "Completed"),
        (JobState::Failed, "Failed"),
        (JobState::Cancelled, "Cancelled"),
        (JobState::Paused, "Paused"),
    ] {
        assert_eq!(state.to_string(), expected);
    }
    // Task states
    for (state, expected) in [
        (TaskState::Pending, "Pending"),
        (TaskState::Assigned, "Assigned"),
        (TaskState::Running, "Running"),
        (TaskState::Completed, "Completed"),
        (TaskState::Failed, "Failed"),
    ] {
        assert_eq!(state.to_string(), expected);
    }
    // Worker states
    for (state, expected) in [
        (WorkerState::Idle, "Idle"),
        (WorkerState::Busy, "Busy"),
        (WorkerState::Overloaded, "Overloaded"),
        (WorkerState::Draining, "Draining"),
        (WorkerState::Offline, "Offline"),
    ] {
        assert_eq!(state.to_string(), expected);
    }
}
