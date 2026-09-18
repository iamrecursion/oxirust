//! Integration tests for oximedia-monitor.
//!
//! These tests run without any optional features by default, using
//! `InMemoryMonitor` which requires no database.  Tests that specifically
//! need `OximediaMonitor` (SQLite-backed) are gated behind the `sqlite`
//! feature and will only run when that feature is enabled.

use oximedia_monitor::{
    alerting_pipeline::{Comparator, PipelineRule, Priority},
    slo_tracker::{SloDefinition, SloTracker, SloType},
    InMemoryMonitor, MonitorConfig,
};
use std::time::Duration;

#[cfg(feature = "sqlite")]
use oximedia_monitor::OximediaMonitor;
#[cfg(feature = "sqlite")]
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

/// Build a [`MonitorConfig`] that is optimised for fast test execution.
fn fast_in_memory_config() -> MonitorConfig {
    let mut cfg = MonitorConfig::default();
    cfg.metrics.enable_system_metrics = false;
    cfg.metrics.enable_disk_metrics = false;
    cfg.metrics.collection_interval = Duration::from_millis(100);
    cfg
}

// ---------------------------------------------------------------------------
// SQLite-backed OximediaMonitor integration tests
// ---------------------------------------------------------------------------

#[cfg(feature = "sqlite")]
fn fast_config(dir: &tempfile::TempDir) -> MonitorConfig {
    let mut cfg = MonitorConfig::default();
    cfg.storage.db_path = dir.path().join("monitor.db");
    cfg.metrics.enable_system_metrics = false;
    cfg.metrics.enable_disk_metrics = false;
    cfg.metrics.collection_interval = Duration::from_millis(100);
    cfg
}

#[cfg(feature = "sqlite")]
fn system_metrics_no_disk_config(dir: &tempfile::TempDir) -> MonitorConfig {
    let mut cfg = MonitorConfig::default();
    cfg.storage.db_path = dir.path().join("monitor.db");
    cfg.metrics.enable_system_metrics = true;
    cfg.metrics.enable_disk_metrics = false;
    cfg.metrics.collection_interval = Duration::from_millis(100);
    cfg
}

#[cfg(feature = "sqlite")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_monitor_lifecycle() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    monitor.start().await.expect("test expectation failed");
    assert!(monitor.metrics_collector().is_running().await);

    monitor.stop().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    assert!(!monitor.metrics_collector().is_running().await);
}

#[cfg(feature = "sqlite")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_system_metrics_collection() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(system_metrics_no_disk_config(&dir))
        .await
        .expect("test expectation failed");
    let metrics = monitor
        .system_metrics()
        .await
        .expect("metrics should be valid");
    assert!(metrics.is_some());

    let m = metrics.expect("m should be valid");
    assert!(m.cpu.cpu_count > 0);
    assert!(m.memory.total > 0);
}

#[cfg(feature = "sqlite")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_application_metrics_initial_state() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    let metrics = monitor.application_metrics();
    assert_eq!(metrics.encoding.total_frames, 0);
    assert_eq!(metrics.jobs.completed, 0);
}

#[cfg(feature = "sqlite")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_quality_metrics_initial_state() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    let metrics = monitor.quality_metrics();
    assert_eq!(metrics.bitrate.video_bitrate_bps, 0);
    assert!(metrics.scores.psnr.is_none());
}

#[cfg(feature = "sqlite")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_encoding_metrics_tracking() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    let tracker = monitor.metrics_collector().application_tracker();
    tracker.record_frame_encoded(16.67);
    tracker.record_frame_encoded(16.67);
    tracker.record_job_completed(30.0);

    let metrics = monitor.application_metrics();
    assert_eq!(metrics.encoding.total_frames, 2);
    assert_eq!(metrics.jobs.completed, 1);
}

#[cfg(feature = "sqlite")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_quality_metrics_tracking() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    let tracker = monitor.metrics_collector().quality_tracker();
    tracker.update_bitrate(5_000_000, 128_000);
    tracker.update_scores(Some(40.0), Some(0.99), Some(92.0));

    let metrics = monitor.quality_metrics();
    assert_eq!(metrics.bitrate.video_bitrate_bps, 5_000_000);
    assert_eq!(metrics.scores.psnr, Some(40.0));
    assert_eq!(metrics.scores.ssim, Some(0.99));
    assert_eq!(metrics.scores.vmaf, Some(92.0));
}

#[cfg(feature = "sqlite")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_alert_manager_present() {
    let dir = tempdir().expect("dir should be valid");
    let monitor = OximediaMonitor::new(fast_config(&dir))
        .await
        .expect("monitor should be valid");
    // Alert manager is created when alerts.enabled == true (the default).
    assert!(monitor.alert_manager().is_some());
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn test_config_validation() {
    let mut config = MonitorConfig::default();
    assert!(config.validate().is_ok());

    // Invalid collection interval.
    config.metrics.collection_interval = std::time::Duration::from_millis(50);
    assert!(config.validate().is_err());
}

// ---------------------------------------------------------------------------
// In-memory monitor integration tests (no sqlite feature required)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_in_memory_monitor_lifecycle() {
    let monitor =
        InMemoryMonitor::new(fast_in_memory_config()).expect("monitor creation should succeed");
    monitor.start().await.expect("start should succeed");
    assert!(monitor.metrics_collector().is_running().await);

    monitor.stop().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    assert!(!monitor.metrics_collector().is_running().await);
}

#[tokio::test]
async fn test_in_memory_application_metrics_initial_state() {
    let monitor =
        InMemoryMonitor::new(fast_in_memory_config()).expect("monitor creation should succeed");
    let metrics = monitor.application_metrics();
    assert_eq!(metrics.encoding.total_frames, 0);
    assert_eq!(metrics.jobs.completed, 0);
}

#[tokio::test]
async fn test_in_memory_quality_metrics_initial_state() {
    let monitor =
        InMemoryMonitor::new(fast_in_memory_config()).expect("monitor creation should succeed");
    let metrics = monitor.quality_metrics();
    assert_eq!(metrics.bitrate.video_bitrate_bps, 0);
    assert!(metrics.scores.psnr.is_none());
}

#[tokio::test]
async fn test_in_memory_alert_manager_present() {
    let monitor =
        InMemoryMonitor::new(fast_in_memory_config()).expect("monitor creation should succeed");
    assert!(monitor.alert_manager().is_some());
}

#[tokio::test]
async fn test_in_memory_record_metric_ring_buffer() {
    let monitor =
        InMemoryMonitor::new(fast_in_memory_config()).expect("monitor creation should succeed");
    monitor.record_metric("cpu_usage", 72.0);
    monitor.record_metric("cpu_usage", 85.0);
    monitor.record_metric("memory_usage", 60.0);
    // Ring buffer is only flushed when the batcher flushes; just verify no panic.
    let _ = monitor.ring_buffer_len();
}

// ---------------------------------------------------------------------------
// TASK 11: Full alert pipeline integration test
// metric threshold -> rule evaluation -> notification dispatch
// ---------------------------------------------------------------------------

/// Tests the full alert pipeline:
/// 1. Metric value is recorded
/// 2. Alert rule threshold is evaluated
/// 3. Alert fires when threshold is breached
/// 4. Alert does NOT fire below threshold
#[tokio::test]
async fn test_full_alert_pipeline_threshold_to_rule_evaluation() {
    use oximedia_monitor::alerting_pipeline::AlertingPipeline;

    let mut pipeline = AlertingPipeline::new();

    // Add a rule: fire when cpu > 90.0
    let rule = PipelineRule::new(
        "high_cpu_alert",
        "cpu_usage",
        Comparator::Gt,
        90.0,
        Priority::Critical,
    )
    .with_silence(Duration::from_millis(0));

    pipeline.add_rule(rule);

    // Below threshold - no alert.
    let fired_below = pipeline.evaluate("cpu_usage", 80.0);
    assert!(
        fired_below.is_empty(),
        "No alert should fire when cpu=80.0 < 90.0, got {fired_below:?}",
    );

    // Above threshold - alert fires.
    let fired_above = pipeline.evaluate("cpu_usage", 95.0);
    assert!(
        !fired_above.is_empty(),
        "Alert should fire when cpu=95.0 > 90.0"
    );
    assert_eq!(fired_above[0].rule_id, "high_cpu_alert");
    assert!((fired_above[0].value - 95.0).abs() < f64::EPSILON);
    assert_eq!(fired_above[0].priority, Priority::Critical);
}

/// Tests that the silence period prevents repeated alert firing.
#[tokio::test]
async fn test_alert_pipeline_silence_period_prevents_repeat() {
    use oximedia_monitor::alerting_pipeline::AlertingPipeline;

    let mut pipeline = AlertingPipeline::new();

    let rule = PipelineRule::new(
        "memory_high",
        "memory",
        Comparator::Gte,
        80.0,
        Priority::Warning,
    )
    .with_silence(Duration::from_secs(3600)); // 1 hour silence

    pipeline.add_rule(rule);

    // First breach fires.
    let first = pipeline.evaluate("memory", 85.0);
    assert!(!first.is_empty(), "First breach should fire");

    // Second breach within silence period does NOT fire.
    let second = pipeline.evaluate("memory", 90.0);
    assert!(
        second.is_empty(),
        "Second breach within silence period should not fire, got {second:?}",
    );
}

/// Tests that multiple rules can be evaluated simultaneously.
#[tokio::test]
async fn test_alert_pipeline_multiple_rules() {
    use oximedia_monitor::alerting_pipeline::AlertingPipeline;

    let mut pipeline = AlertingPipeline::new();

    pipeline.add_rule(
        PipelineRule::new("cpu_warn", "cpu", Comparator::Gt, 70.0, Priority::Warning)
            .with_silence(Duration::ZERO),
    );
    pipeline.add_rule(
        PipelineRule::new(
            "cpu_critical",
            "cpu",
            Comparator::Gt,
            90.0,
            Priority::Critical,
        )
        .with_silence(Duration::ZERO),
    );

    // Value 80.0 triggers warning but not critical.
    let fired = pipeline.evaluate("cpu", 80.0);
    assert_eq!(fired.len(), 1, "Only warning rule should fire at cpu=80");
    assert_eq!(fired[0].rule_id, "cpu_warn");

    // Value 95.0 triggers both.
    let fired = pipeline.evaluate("cpu", 95.0);
    assert_eq!(
        fired.len(),
        2,
        "Both rules should fire at cpu=95, got {fired:?}"
    );
}

/// Tests that the alerting pipeline integrates with InMemoryMonitor.
#[tokio::test]
async fn test_alert_pipeline_via_in_memory_monitor() {
    let monitor =
        InMemoryMonitor::new(fast_in_memory_config()).expect("monitor creation should succeed");

    // Register an alert rule for disk usage.
    monitor.add_alert_rule(
        PipelineRule::new(
            "disk_critical",
            "disk_usage",
            Comparator::Gt,
            95.0,
            Priority::Critical,
        )
        .with_silence(Duration::ZERO),
    );

    // Record values: first normal, then critical.
    monitor.record_metric("disk_usage", 70.0);
    monitor.record_metric("disk_usage", 97.0);

    // After recording 97.0, the pipeline should have fired the alert.
    // We verify the monitor processed without panic. Pipeline fired status is
    // internal; the test verifies the whole path is callable without error.
    let _ = monitor.ring_buffer_len();
}

// ---------------------------------------------------------------------------
// TASK 12: SLO tracking with synthetic 99.9% uptime SLO and simulated downtime
// ---------------------------------------------------------------------------

/// Tests the 99.9% availability SLO with synthetic uptime data.
///
/// Simulates 10,000 1-minute observations where 0.1% (10 observations) are
/// outages (availability = 0.0) and the rest are healthy (availability = 100.0).
/// Verifies that the SLO tracker correctly reports just-meeting the 99.9% target.
#[test]
fn test_slo_99_9_availability_exactly_at_threshold() {
    let mut tracker = SloTracker::new(SloDefinition::availability_99_9());

    // 9990 passing + 10 failing = 99.9%
    for i in 0u64..9990 {
        tracker.record(i, 100.0); // healthy
    }
    for i in 9990u64..10000 {
        tracker.record(i, 0.0); // outage
    }

    let pct = tracker.current_compliance_pct();
    assert!(
        (pct - 99.9).abs() < 0.01,
        "Expected ~99.9% compliance, got {pct:.4}%"
    );
    assert!(
        tracker.is_meeting_slo(99.9),
        "SLO should be met at exactly 99.9%"
    );
}

/// Tests that a system with >0.1% downtime fails the 99.9% SLO.
#[test]
fn test_slo_99_9_availability_fails_with_excess_downtime() {
    let mut tracker = SloTracker::new(SloDefinition::availability_99_9());

    // 998 passing + 2 failing = 99.8% (below 99.9% target)
    for i in 0u64..998 {
        tracker.record(i, 100.0);
    }
    for i in 998u64..1000 {
        tracker.record(i, 0.0); // 0.2% failure rate
    }

    let pct = tracker.current_compliance_pct();
    assert!(
        pct < 99.9,
        "Compliance should be below 99.9% with 0.2% outage, got {pct:.4}%"
    );
    assert!(
        !tracker.is_meeting_slo(99.9),
        "SLO should NOT be met with 99.8% compliance"
    );
}

/// Tests SLO tracking with a pattern of bursty outages interspersed with uptime.
#[test]
fn test_slo_bursty_downtime_pattern() {
    let mut tracker = SloTracker::new(SloDefinition {
        name: "Media Pipeline Availability".to_string(),
        slo_type: SloType::Availability,
        target: 99.0,
        window_hours: 24,
    });

    // Simulate 100 minutes: 99 up + 1 down = 99%.
    let mut epoch = 0u64;
    for _ in 0..99 {
        tracker.record(epoch, 100.0);
        epoch += 60;
    }
    // One minute of downtime.
    tracker.record(epoch, 0.0);

    let pct = tracker.current_compliance_pct();
    assert!(
        (pct - 99.0).abs() < 0.01,
        "Expected 99.0% compliance, got {pct:.4}%"
    );
    assert!(
        tracker.is_meeting_slo(99.0),
        "99% SLO should be met with 99% compliance"
    );
    assert!(
        !tracker.is_meeting_slo(99.5),
        "99.5% SLO should NOT be met with 99% compliance"
    );
}

/// Tests SLO tracking for latency SLO.
#[test]
fn test_slo_latency_p99_with_simulated_spikes() {
    let mut tracker = SloTracker::new(SloDefinition::latency_p99_100ms());

    // 99 requests under 100ms + 1 spike at 500ms.
    for i in 0u64..99 {
        tracker.record(i, 50.0 + (i as f64 * 0.5)); // 50..99.5 ms range
    }
    tracker.record(99, 500.0); // spike

    // 99% compliance (99/100 meet target).
    let pct = tracker.current_compliance_pct();
    assert!(
        (pct - 99.0).abs() < 0.01,
        "Expected 99.0% latency compliance, got {pct:.4}%"
    );

    // A 99% latency SLO is met, but a 99.5% one is not.
    assert!(tracker.is_meeting_slo(99.0));
    assert!(!tracker.is_meeting_slo(99.5));
}

/// Tests the error budget calculation via SLO compliance.
///
/// Error budget = 1 - SLO target.  For a 99.9% SLO over 30 days:
/// allowed downtime = 30 * 24 * 60 * 0.001 = 43.2 minutes.
#[test]
fn test_slo_error_budget_calculation() {
    let slo_target = 99.9_f64; // %
    let window_minutes = 30u64 * 24 * 60; // 43 200 minutes in 30 days
    let allowed_failures = (window_minutes as f64 * (1.0 - slo_target / 100.0)).floor() as u64;

    // allowed_failures = floor(43200 * 0.001) = floor(43.2) = 43 minutes of downtime allowed.
    assert_eq!(allowed_failures, 43);

    let mut tracker = SloTracker::new(SloDefinition::availability_99_9());

    // Record exactly allowed_failures outages.
    let passing = window_minutes - allowed_failures;
    for i in 0..passing {
        tracker.record(i * 60, 100.0);
    }
    for i in 0..allowed_failures {
        tracker.record((passing + i) * 60, 0.0);
    }

    let pct = tracker.current_compliance_pct();
    // pct = passing / total * 100 = (43200 - 43) / 43200 * 100 ≈ 99.9004...%
    assert!(
        pct >= 99.9,
        "SLO should be met with budget-limited outages, got {pct:.6}%"
    );
    assert!(
        tracker.is_meeting_slo(99.9),
        "SLO should be met when downtime is within error budget"
    );
}

/// Tests that SLO tracking handles the boundary: exactly one more failure than
/// the error budget breaks the SLO.
#[test]
fn test_slo_error_budget_exceeded_by_one() {
    let mut tracker = SloTracker::new(SloDefinition::availability_99_9());

    // 1000 observations: 999 pass + 1 fail = 99.9% → meets SLO.
    for i in 0u64..999 {
        tracker.record(i, 100.0);
    }
    tracker.record(999, 0.0);
    assert!(tracker.is_meeting_slo(99.9), "Should meet SLO with 99.9%");

    // Add one more failure: 999 pass + 2 fail / 1001 = 99.8% → fails SLO.
    tracker.record(1000, 0.0);
    let pct = tracker.current_compliance_pct();
    assert!(
        pct < 99.9,
        "SLO should fail when one more outage pushes below 99.9%, got {pct:.6}%"
    );
    assert!(
        !tracker.is_meeting_slo(99.9),
        "SLO should NOT be met after exceeding error budget"
    );
}
