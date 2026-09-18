//! Smoke tests for newly-wired orphan modules in `oximedia-rights`.

// ── revenue ───────────────────────────────────────────────────────────────────
#[test]
fn revenue_share_distribute_equal() {
    use oximedia_rights::revenue::RevenueShare;
    let mut share = RevenueShare::new();
    share.add_party(1, 1.0);
    share.add_party(2, 1.0);
    share.add_party(3, 1.0);
    let result = share.distribute(300.0);
    assert_eq!(result.len(), 3);
    for (_, amount) in &result {
        assert!(
            (amount - 100.0).abs() < 0.01,
            "each party should get 100.0, got {amount}"
        );
    }
}

#[test]
fn revenue_share_empty_distributes_nothing() {
    use oximedia_rights::revenue::RevenueShare;
    let share = RevenueShare::new();
    let result = share.distribute(1000.0);
    assert!(result.is_empty());
}

// ── music_cue_sheet ───────────────────────────────────────────────────────────
#[test]
fn cue_sheet_total_music_duration() {
    use oximedia_rights::music_cue_sheet::{CueEntry, CueSheet, ProductionType, UsageType};
    let mut cue_sheet = CueSheet::new(
        "prod-001",
        "Test Production",
        ProductionType::Television,
        "Acme Studios",
        "US",
        2400.0,
    )
    .expect("CueSheet::new should succeed");
    let entry = CueEntry::new(1, "Test Theme", UsageType::BackgroundInstrumental, 90.0)
        .expect("CueEntry::new should succeed");
    cue_sheet.add_cue(entry);
    assert!((cue_sheet.total_music_duration_secs() - 90.0).abs() < 0.01);
    assert!(cue_sheet.music_ratio() > 0.0);
}

// ── rights_renewal ────────────────────────────────────────────────────────────
#[test]
fn renewal_scheduler_empty() {
    use oximedia_rights::rights_renewal::RenewalScheduler;
    let scheduler = RenewalScheduler::with_defaults();
    assert_eq!(scheduler.candidate_count(), 0);
}

#[test]
fn renewal_status_terminal_variants() {
    use oximedia_rights::rights_renewal::RenewalStatus;
    assert!(RenewalStatus::Renewed.is_terminal());
    assert!(RenewalStatus::Lapsed.is_terminal());
    assert!(!RenewalStatus::Pending.is_terminal());
}

// ── conflict ──────────────────────────────────────────────────────────────────
#[test]
fn rights_conflict_detector_no_overlap() {
    use oximedia_rights::conflict::{RightsConflictDetector, RightsWindow};
    // Two windows that don't overlap in territory.
    let w1 = RightsWindow::new(1, "US", 0, 1000);
    let w2 = RightsWindow::new(2, "UK", 0, 1000);
    let conflicts = RightsConflictDetector::find_overlaps(&[w1, w2]);
    assert!(
        conflicts.is_empty(),
        "different territories should not conflict"
    );
}

#[test]
fn rights_conflict_detector_finds_overlap() {
    use oximedia_rights::conflict::{RightsConflictDetector, RightsWindow};
    // Two windows for the same territory with overlapping times.
    let w1 = RightsWindow::new(1, "US", 0, 2000);
    let w2 = RightsWindow::new(2, "US", 1000, 3000);
    let conflicts = RightsConflictDetector::find_overlaps(&[w1, w2]);
    assert!(
        !conflicts.is_empty(),
        "same territory overlapping windows should conflict"
    );
}

// ── expiry_alert ──────────────────────────────────────────────────────────────
#[test]
fn expiry_alerter_no_alerts_far_future() {
    use oximedia_rights::expiry_alert::{Right, RightsExpiryAlerter};
    // Expires 1 year from "now" (ts=0 → we treat 0 as now, expiry=365*86400)
    let right = Right::new(1, "Film License", 365 * 86_400);
    let alerts = RightsExpiryAlerter::check(&[right], 0, 7);
    assert!(
        alerts.is_empty(),
        "expiry > 7 days from now should not alert"
    );
}

#[test]
fn expiry_alerter_alerts_imminent_expiry() {
    use oximedia_rights::expiry_alert::{Right, RightsExpiryAlerter};
    // Expires in 3 days from now_ts.
    let now_ts = 1_000_000u64;
    let expires_ts = now_ts + 3 * 86_400;
    let right = Right::new(1, "Music License", expires_ts);
    let alerts = RightsExpiryAlerter::check(&[right], now_ts, 7);
    assert_eq!(alerts.len(), 1, "imminent expiry should produce one alert");
}

// ── embargo_manager ───────────────────────────────────────────────────────────
#[test]
fn embargo_manager_empty() {
    use oximedia_rights::embargo_manager::EmbargoManager;
    let mgr = EmbargoManager::new();
    assert!(mgr.is_empty());
    assert_eq!(mgr.len(), 0);
}

// ── clearance_notifications ───────────────────────────────────────────────────
#[test]
fn clearance_notification_log_empty() {
    use oximedia_rights::clearance_notifications::NotificationLog;
    let log = NotificationLog::new();
    assert_eq!(log.total_count(), 0);
}

// ── compliance_report ─────────────────────────────────────────────────────────
#[test]
fn compliance_report_builder_accessible() {
    use oximedia_rights::compliance_report::ComplianceReportBuilder;
    // Builder should be constructible without panicking.
    let builder =
        ComplianceReportBuilder::new("rpt-001", "OxiMedia Platform", "2026-05-30", "Legal Team");
    let _ = std::hint::black_box(builder);
}

// ── sync_licensing ────────────────────────────────────────────────────────────
#[test]
fn sync_license_manager_empty() {
    use oximedia_rights::sync_licensing::SyncLicenseManager;
    let mgr = SyncLicenseManager::new();
    assert!(mgr.is_empty());
    assert_eq!(mgr.len(), 0);
}

// ── usage_tracking ────────────────────────────────────────────────────────────
#[test]
fn usage_tracker_initial_state() {
    use oximedia_rights::usage_tracking::UsageTracker;
    let tracker = UsageTracker::new();
    assert!(tracker.events().is_empty());
}

// ── pool_config ───────────────────────────────────────────────────────────────
#[test]
fn pool_config_builder_default() {
    use oximedia_rights::pool_config::PoolConfig;
    let cfg = PoolConfig::builder().build();
    assert!(cfg.max_connections > 0);
    assert!(cfg.min_idle <= cfg.max_connections);
}

// ── rights_audit ──────────────────────────────────────────────────────────────
#[test]
fn rights_audit_log_empty() {
    use oximedia_rights::rights_audit::RightsAuditLog;
    let log = RightsAuditLog::new();
    assert!(log.is_empty());
    assert_eq!(log.len(), 0);
}

// ── query_cache ───────────────────────────────────────────────────────────────
#[test]
fn query_cache_config_default() {
    use oximedia_rights::query_cache::CacheConfig;
    let cfg = CacheConfig::default();
    assert!(cfg.ttl_seconds > 0);
    assert!(cfg.is_enabled());
}

// ── performance_rights ────────────────────────────────────────────────────────
#[test]
fn performance_rights_tracker_empty() {
    use oximedia_rights::performance_rights::PerformanceRightsTracker;
    let tracker = PerformanceRightsTracker::new();
    assert_eq!(tracker.record_count(), 0);
    assert!(tracker.unreported_records().is_empty());
}
