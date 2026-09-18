//! Smoke tests for all 14 newly wired batch orphan modules.

// ── audit_log ────────────────────────────────────────────────────────────────

#[test]
fn test_audit_log_creation() {
    use oximedia_batch::audit_log::{AuditAction, AuditLog};
    use oximedia_batch::types::JobId;
    let log = AuditLog::new();
    let job_id = JobId::new();
    log.log_user(job_id.clone(), "alice", AuditAction::JobSubmitted);
    let stats = log.stats();
    assert_eq!(stats.total_entries, 1);
}

#[test]
fn test_audit_log_action_display() {
    use oximedia_batch::audit_log::AuditAction;
    assert_eq!(format!("{}", AuditAction::JobSubmitted), "job_submitted");
    assert_eq!(format!("{}", AuditAction::JobCompleted), "job_completed");
    assert_eq!(format!("{}", AuditAction::JobFailed), "job_failed");
}

// ── batch_analytics ──────────────────────────────────────────────────────────

#[test]
fn test_batch_analytics_ingest_sample() {
    use oximedia_batch::batch_analytics::{AnalyticsConfig, BatchAnalytics, JobSample, Window};
    use oximedia_batch::types::JobState;
    let analytics = BatchAnalytics::new(AnalyticsConfig::default());
    analytics.ingest(JobSample::now("transcode", JobState::Completed, 5.0));
    analytics.ingest(JobSample::now("transcode", JobState::Completed, 7.0));
    let metrics = analytics.metrics(Window::LastHour);
    // At least 2 jobs ingested
    assert_eq!(metrics.total_jobs, 2);
}

// ── chaining ─────────────────────────────────────────────────────────────────

#[test]
fn test_chaining_dependency_order() {
    use oximedia_batch::chaining::BatchJobChain;
    use oximedia_batch::job::{BatchJob, BatchOperation};
    let job_a = BatchJob::new(
        "job-a".to_string(),
        BatchOperation::Transcode {
            preset: "h264".into(),
        },
    );
    let id_a = job_a.id.as_str().to_string();
    let job_b = BatchJob::new(
        "job-b".to_string(),
        BatchOperation::Transcode {
            preset: "h264".into(),
        },
    );

    let mut chain = BatchJobChain::new();
    chain.add(job_a, None);
    chain.add(job_b, Some(id_a.clone()));

    // First job should be ready (no dependency)
    let first = chain.next_ready().expect("job-a ready");
    assert_eq!(first.name, "job-a");

    // Second job should NOT be ready yet
    assert!(chain.next_ready().is_none());

    // Mark first job complete
    chain.mark_complete(&id_a);

    // Now second job is ready
    let second = chain.next_ready().expect("job-b ready");
    assert_eq!(second.name, "job-b");
}

// ── checkpoint ───────────────────────────────────────────────────────────────

#[test]
fn test_checkpoint_new_sequence() {
    use oximedia_batch::checkpoint::Checkpoint;
    let cp = Checkpoint::new(42);
    assert_eq!(cp.sequence, 42);
    assert!(cp.queued_jobs.is_empty());
    assert!(cp.in_progress_jobs.is_empty());
}

#[test]
fn test_checkpoint_roundtrip_json() {
    use oximedia_batch::checkpoint::Checkpoint;
    let cp = Checkpoint::new(1).with_queued(vec!["job-1".to_string(), "job-2".to_string()]);
    let json = serde_json::to_string(&cp).expect("serialize ok");
    let restored: Checkpoint = serde_json::from_str(&json).expect("deserialize ok");
    assert_eq!(restored.sequence, 1);
    assert_eq!(restored.queued_jobs.len(), 2);
}

// ── cluster_discovery ────────────────────────────────────────────────────────

#[test]
fn test_cluster_registry_register() {
    use oximedia_batch::cluster_discovery::{ClusterRegistry, WorkerCapabilities, WorkerNode};
    use std::net::SocketAddr;
    let registry = ClusterRegistry::with_defaults();
    let addr: SocketAddr = "127.0.0.1:9000".parse().expect("addr ok");
    let caps = WorkerCapabilities::default();
    let node = WorkerNode::new("worker-1", "Worker One", addr, caps);
    let is_new = registry.register(node);
    assert!(is_new);
    assert_eq!(registry.node_count(), 1);
}

// ── cost_estimator ───────────────────────────────────────────────────────────

#[test]
fn test_cost_estimator_fallback_prediction() {
    use oximedia_batch::cost_estimator::CostEstimator;
    let estimator = CostEstimator::new();
    // With no history, we get a fallback prediction
    let pred = estimator
        .predict("transcode:h264", 100_000_000)
        .expect("predict ok");
    assert!(pred.predicted_duration_secs > 0.0);
    assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
}

// ── dead_letter_queue ────────────────────────────────────────────────────────

#[test]
fn test_dead_letter_queue_push() {
    use oximedia_batch::dead_letter_queue::{DeadLetterEntry, DeadLetterQueue, DeadLetterReason};
    use oximedia_batch::types::JobId;
    let dlq = DeadLetterQueue::with_defaults();
    let entry = DeadLetterEntry::new(
        JobId::new(),
        "test-job",
        DeadLetterReason::RetriesExhausted { attempts: 3 },
        "all retries failed",
        3,
    );
    dlq.push(entry);
    assert_eq!(dlq.len(), 1);
}

// ── error_recovery ───────────────────────────────────────────────────────────

#[test]
fn test_error_recovery_skip_mode() {
    use oximedia_batch::error_recovery::BatchErrorRecovery;
    let mut rec = BatchErrorRecovery::new(true);
    rec.record_failure(1, "file not found");
    assert!(rec.should_continue());
    assert_eq!(rec.failure_count(), 1);
}

#[test]
fn test_error_recovery_abort_mode() {
    use oximedia_batch::error_recovery::BatchErrorRecovery;
    let mut rec = BatchErrorRecovery::new(false);
    rec.record_failure(2, "codec error");
    assert!(!rec.should_continue());
}

// ── graceful_shutdown ────────────────────────────────────────────────────────

#[test]
fn test_graceful_shutdown_phase_transitions() {
    use oximedia_batch::graceful_shutdown::{
        GracefulShutdown, GracefulShutdownConfig, ShutdownPhase,
    };
    let gs = GracefulShutdown::new(GracefulShutdownConfig::default());
    assert_eq!(gs.phase(), ShutdownPhase::Running);
    // In Running phase, new jobs should not be rejected
    assert!(!gs.should_reject_new_jobs());
}

// ── job_deps ─────────────────────────────────────────────────────────────────

#[test]
fn test_job_deps_ready_after_completion() {
    use oximedia_batch::job_deps::{DependencyStatus, JobDependencyManager};
    use oximedia_batch::types::JobId;
    let mut mgr = JobDependencyManager::default();
    let id_a = JobId::from_string("job-a".to_string());
    let id_b = JobId::from_string("job-b".to_string());
    mgr.register_job(&id_a);
    mgr.register_job(&id_b);
    mgr.add_dependency(&id_b, &id_a).expect("B depends on A");
    assert_eq!(
        mgr.status(&id_b).expect("status ok"),
        DependencyStatus::Pending
    );
    mgr.mark_completed(&id_a).expect("complete A");
    assert_eq!(
        mgr.status(&id_b).expect("status ok"),
        DependencyStatus::Ready
    );
}

// ── notification_hub ─────────────────────────────────────────────────────────

#[test]
fn test_notification_hub_subscribe_and_publish() {
    use oximedia_batch::notification_hub::{
        EventFilter, HubConfig, JobEvent, NotificationHub, NotificationTarget, WebhookSpec,
    };
    use oximedia_batch::types::{JobId, JobState};
    let hub = NotificationHub::new(HubConfig::default(), None);
    hub.subscribe(
        NotificationTarget::Webhook(WebhookSpec::new("https://example.com/hook")),
        EventFilter::All,
    );
    let event = JobEvent::new(JobId::new(), "test-job", JobState::Completed);
    hub.publish(event);
    let stats = hub.stats();
    assert_eq!(stats.events_published, 1);
}

// ── progress_agg ─────────────────────────────────────────────────────────────

#[test]
fn test_progress_agg_complete_tracking() {
    use oximedia_batch::progress_agg::BatchProgressAggregator;
    let mut agg = BatchProgressAggregator::new(10);
    agg.complete(3);
    assert!((agg.percent() - 30.0).abs() < 1e-4);
    assert!(!agg.is_done());
    agg.complete(7);
    assert!(agg.is_done());
}

// ── quota_enforcer ────────────────────────────────────────────────────────────

#[test]
fn test_quota_enforcer_concurrent_limit() {
    use oximedia_batch::quota_enforcer::{QuotaEnforcer, QuotaLimits};
    let enforcer = QuotaEnforcer::new();
    enforcer.set_limits(
        "user-1",
        QuotaLimits {
            max_concurrent_jobs: Some(2),
            max_daily_jobs: None,
            max_storage_bytes: None,
            max_cpu_hours: None,
        },
    );
    // First two jobs should be admitted
    assert!(enforcer.check_can_admit("user-1").is_ok());
    enforcer.charge_concurrent_job("user-1");
    assert!(enforcer.check_can_admit("user-1").is_ok());
    enforcer.charge_concurrent_job("user-1");
    // Third job should be rejected (concurrent limit = 2)
    assert!(enforcer.check_can_admit("user-1").is_err());
    // After releasing one, admission is OK again
    enforcer.release_concurrent_job("user-1");
    assert!(enforcer.check_can_admit("user-1").is_ok());
}

// ── quota ────────────────────────────────────────────────────────────────────

#[test]
fn test_quota_consume_and_release() {
    use oximedia_batch::quota::ResourceQuota;
    let mut quota = ResourceQuota::new(4.0, 8192);
    assert!(quota.consume(2.0, 2048));
    assert!((quota.available_cpu() - 2.0).abs() < 1e-4);
    quota.release(2.0, 2048);
    assert!(quota.is_idle());
}
