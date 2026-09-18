//! Archive health dashboard data generation.
//!
//! Aggregates integrity scan results, retention status, dedup metrics, and
//! catalog statistics into a unified health dashboard. Supports trend
//! tracking over time for monitoring archive health progression.

use crate::dedup_archive::DedupIndex;
use crate::integrity_scan::{IntegrityScan, ScanHealthMetrics};
use crate::retention_schedule::{RetentionClass, RetentionSchedule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Time-series data point
// ---------------------------------------------------------------------------

/// A single data point in a time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp in Unix milliseconds.
    pub timestamp_ms: u64,
    /// Value at this timestamp.
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point.
    #[must_use]
    pub fn new(timestamp_ms: u64, value: f64) -> Self {
        Self {
            timestamp_ms,
            value,
        }
    }
}

/// A named time series of data points.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Series name (e.g., "health_score", "total_files").
    pub name: String,
    /// Unit description (e.g., "percent", "count", "bytes").
    pub unit: String,
    /// Data points sorted by timestamp ascending.
    pub points: Vec<DataPoint>,
}

impl TimeSeries {
    /// Create a new empty time series.
    #[must_use]
    pub fn new(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            points: Vec::new(),
        }
    }

    /// Add a data point.
    pub fn add(&mut self, timestamp_ms: u64, value: f64) {
        self.points.push(DataPoint::new(timestamp_ms, value));
    }

    /// Number of data points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the series is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get the latest value.
    #[must_use]
    pub fn latest_value(&self) -> Option<f64> {
        self.points.last().map(|p| p.value)
    }

    /// Get the minimum value in the series.
    #[must_use]
    pub fn min_value(&self) -> Option<f64> {
        self.points
            .iter()
            .map(|p| p.value)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the maximum value in the series.
    #[must_use]
    pub fn max_value(&self) -> Option<f64> {
        self.points
            .iter()
            .map(|p| p.value)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Compute the average value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average(&self) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }
        let sum: f64 = self.points.iter().map(|p| p.value).sum();
        Some(sum / self.points.len() as f64)
    }

    /// Compute the trend (slope) using simple linear regression.
    ///
    /// Returns `None` if fewer than 2 points.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn trend_slope(&self) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }

        let n = self.points.len() as f64;
        let sum_x: f64 = self.points.iter().map(|p| p.timestamp_ms as f64).sum();
        let sum_y: f64 = self.points.iter().map(|p| p.value).sum();
        let sum_xy: f64 = self
            .points
            .iter()
            .map(|p| p.timestamp_ms as f64 * p.value)
            .sum();
        let sum_x2: f64 = self
            .points
            .iter()
            .map(|p| (p.timestamp_ms as f64).powi(2))
            .sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return Some(0.0);
        }
        Some((n * sum_xy - sum_x * sum_y) / denom)
    }
}

// ---------------------------------------------------------------------------
// Archive health status
// ---------------------------------------------------------------------------

/// Overall health status of the archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All checks passed; no issues detected.
    Healthy,
    /// Minor issues detected (warnings present).
    Degraded,
    /// Significant issues detected (corruption or missing files).
    AtRisk,
    /// Critical: immediate attention required.
    Critical,
}

impl HealthStatus {
    /// Numeric severity (0 = healthy, 3 = critical).
    #[must_use]
    pub const fn severity(&self) -> u8 {
        match self {
            Self::Healthy => 0,
            Self::Degraded => 1,
            Self::AtRisk => 2,
            Self::Critical => 3,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::Healthy => "HEALTHY",
            Self::Degraded => "DEGRADED",
            Self::AtRisk => "AT_RISK",
            Self::Critical => "CRITICAL",
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Dashboard sections
// ---------------------------------------------------------------------------

/// Integrity section of the health dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegritySection {
    /// Last scan health score (0.0 to 1.0).
    pub health_score: f64,
    /// Total files scanned.
    pub total_scanned: usize,
    /// Files found OK.
    pub ok_count: usize,
    /// Files found corrupted.
    pub corrupted_count: usize,
    /// Files missing.
    pub missing_count: usize,
    /// Files modified.
    pub modified_count: usize,
    /// Total bytes scanned.
    pub total_bytes_scanned: u64,
    /// Last scan duration in milliseconds.
    pub last_scan_duration_ms: u64,
}

impl From<&ScanHealthMetrics> for IntegritySection {
    fn from(m: &ScanHealthMetrics) -> Self {
        Self {
            health_score: m.health_score(),
            total_scanned: m.total_scanned,
            ok_count: m.ok_count,
            corrupted_count: m.corrupted_count,
            missing_count: m.missing_count,
            modified_count: m.modified_count,
            total_bytes_scanned: m.total_bytes_scanned,
            last_scan_duration_ms: m.duration_ms,
        }
    }
}

/// Storage section of the health dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSection {
    /// Total logical storage used (bytes).
    pub logical_bytes: u64,
    /// Total physical storage used (bytes, after dedup).
    pub physical_bytes: u64,
    /// Deduplication ratio.
    pub dedup_ratio: f64,
    /// Bytes saved by deduplication.
    pub bytes_saved: u64,
    /// Number of unique content items.
    pub unique_items: usize,
    /// Total references (unique + duplicates).
    pub total_references: u64,
}

/// Retention section of the health dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionSection {
    /// Total entries in retention schedule.
    pub total_entries: usize,
    /// Entries eligible for deletion right now.
    pub eligible_for_deletion: usize,
    /// Entries under legal hold.
    pub legal_hold_count: usize,
    /// Breakdown by retention class.
    pub class_breakdown: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Health dashboard
// ---------------------------------------------------------------------------

/// Complete archive health dashboard snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDashboard {
    /// Timestamp of this snapshot (Unix ms).
    pub timestamp_ms: u64,
    /// Overall health status.
    pub status: HealthStatus,
    /// Integrity metrics.
    pub integrity: IntegritySection,
    /// Storage metrics.
    pub storage: StorageSection,
    /// Retention metrics.
    pub retention: RetentionSection,
    /// Summary alerts.
    pub alerts: Vec<DashboardAlert>,
}

/// An alert raised by the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAlert {
    /// Alert severity.
    pub severity: HealthStatus,
    /// Short summary.
    pub message: String,
    /// Affected component.
    pub component: String,
}

impl HealthDashboard {
    /// Format the dashboard as a human-readable summary.
    #[must_use]
    pub fn to_summary_string(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Archive Health Dashboard ===\n");
        out.push_str(&format!("Status: {}\n\n", self.status));

        out.push_str("-- Integrity --\n");
        out.push_str(&format!(
            "  Health score:    {:.1}%\n",
            self.integrity.health_score * 100.0
        ));
        out.push_str(&format!(
            "  Files scanned:   {}\n",
            self.integrity.total_scanned
        ));
        out.push_str(&format!("  OK:              {}\n", self.integrity.ok_count));
        out.push_str(&format!(
            "  Corrupted:       {}\n",
            self.integrity.corrupted_count
        ));
        out.push_str(&format!(
            "  Missing:         {}\n",
            self.integrity.missing_count
        ));

        out.push_str("\n-- Storage --\n");
        out.push_str(&format!(
            "  Logical size:    {} bytes\n",
            self.storage.logical_bytes
        ));
        out.push_str(&format!(
            "  Physical size:   {} bytes\n",
            self.storage.physical_bytes
        ));
        out.push_str(&format!(
            "  Dedup ratio:     {:.2}x\n",
            self.storage.dedup_ratio
        ));
        out.push_str(&format!(
            "  Bytes saved:     {} bytes\n",
            self.storage.bytes_saved
        ));

        out.push_str("\n-- Retention --\n");
        out.push_str(&format!(
            "  Total entries:   {}\n",
            self.retention.total_entries
        ));
        out.push_str(&format!(
            "  Eligible delete: {}\n",
            self.retention.eligible_for_deletion
        ));
        out.push_str(&format!(
            "  Legal holds:     {}\n",
            self.retention.legal_hold_count
        ));

        if !self.alerts.is_empty() {
            out.push_str(&format!("\n-- Alerts ({}) --\n", self.alerts.len()));
            for alert in &self.alerts {
                out.push_str(&format!(
                    "  [{}] {}: {}\n",
                    alert.severity, alert.component, alert.message
                ));
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Dashboard builder
// ---------------------------------------------------------------------------

/// Builds a health dashboard snapshot from various data sources.
pub struct DashboardBuilder {
    timestamp_ms: u64,
    integrity: Option<IntegritySection>,
    storage: Option<StorageSection>,
    retention: Option<RetentionSection>,
}

impl DashboardBuilder {
    /// Create a new builder for the given timestamp.
    #[must_use]
    pub fn new(timestamp_ms: u64) -> Self {
        Self {
            timestamp_ms,
            integrity: None,
            storage: None,
            retention: None,
        }
    }

    /// Add integrity metrics from a scan.
    #[must_use]
    pub fn with_integrity_scan(mut self, scan: &IntegrityScan) -> Self {
        let metrics = scan.metrics();
        self.integrity = Some(IntegritySection::from(&metrics));
        self
    }

    /// Add integrity metrics directly.
    #[must_use]
    pub fn with_integrity(mut self, section: IntegritySection) -> Self {
        self.integrity = Some(section);
        self
    }

    /// Add storage metrics from a dedup index.
    #[must_use]
    pub fn with_dedup_index(mut self, index: &DedupIndex) -> Self {
        let stats = index.stats();
        let logical = index.estimated_logical_storage();
        let physical = index.estimated_physical_storage();
        let ratio = if physical == 0 {
            1.0
        } else {
            logical as f64 / physical as f64
        };

        self.storage = Some(StorageSection {
            logical_bytes: logical,
            physical_bytes: physical,
            dedup_ratio: ratio,
            bytes_saved: stats.bytes_saved,
            unique_items: stats.unique_entries,
            total_references: stats.total_references,
        });
        self
    }

    /// Add storage metrics directly.
    #[must_use]
    pub fn with_storage(mut self, section: StorageSection) -> Self {
        self.storage = Some(section);
        self
    }

    /// Add retention metrics from a schedule.
    #[must_use]
    pub fn with_retention_schedule(mut self, schedule: &RetentionSchedule, now_ms: u64) -> Self {
        let eligible = schedule.eligible_for_deletion(now_ms).len();
        let holds = schedule.legal_holds().len();

        let mut class_breakdown = HashMap::new();
        for class in [
            RetentionClass::Temporary,
            RetentionClass::Standard,
            RetentionClass::LongTerm,
            RetentionClass::Permanent,
        ] {
            let count = schedule.by_class(class).len();
            if count > 0 {
                class_breakdown.insert(class.label().to_string(), count);
            }
        }

        self.retention = Some(RetentionSection {
            total_entries: schedule.len(),
            eligible_for_deletion: eligible,
            legal_hold_count: holds,
            class_breakdown,
        });
        self
    }

    /// Add retention metrics directly.
    #[must_use]
    pub fn with_retention(mut self, section: RetentionSection) -> Self {
        self.retention = Some(section);
        self
    }

    /// Build the dashboard.
    #[must_use]
    pub fn build(self) -> HealthDashboard {
        let integrity = self.integrity.unwrap_or(IntegritySection {
            health_score: 1.0,
            total_scanned: 0,
            ok_count: 0,
            corrupted_count: 0,
            missing_count: 0,
            modified_count: 0,
            total_bytes_scanned: 0,
            last_scan_duration_ms: 0,
        });

        let storage = self.storage.unwrap_or(StorageSection {
            logical_bytes: 0,
            physical_bytes: 0,
            dedup_ratio: 1.0,
            bytes_saved: 0,
            unique_items: 0,
            total_references: 0,
        });

        let retention = self.retention.unwrap_or(RetentionSection {
            total_entries: 0,
            eligible_for_deletion: 0,
            legal_hold_count: 0,
            class_breakdown: HashMap::new(),
        });

        // Generate alerts
        let mut alerts = Vec::new();

        if integrity.corrupted_count > 0 {
            alerts.push(DashboardAlert {
                severity: HealthStatus::Critical,
                message: format!("{} corrupted file(s) detected", integrity.corrupted_count),
                component: "integrity".to_string(),
            });
        }
        if integrity.missing_count > 0 {
            alerts.push(DashboardAlert {
                severity: HealthStatus::AtRisk,
                message: format!("{} missing file(s)", integrity.missing_count),
                component: "integrity".to_string(),
            });
        }
        if integrity.health_score < 0.99 && integrity.total_scanned > 0 {
            alerts.push(DashboardAlert {
                severity: HealthStatus::Degraded,
                message: format!(
                    "integrity health score below 99%: {:.1}%",
                    integrity.health_score * 100.0
                ),
                component: "integrity".to_string(),
            });
        }
        if retention.eligible_for_deletion > 100 {
            alerts.push(DashboardAlert {
                severity: HealthStatus::Degraded,
                message: format!(
                    "{} assets eligible for deletion — consider running enforcement",
                    retention.eligible_for_deletion
                ),
                component: "retention".to_string(),
            });
        }

        // Determine overall status
        let status = if alerts.iter().any(|a| a.severity == HealthStatus::Critical) {
            HealthStatus::Critical
        } else if alerts.iter().any(|a| a.severity == HealthStatus::AtRisk) {
            HealthStatus::AtRisk
        } else if alerts.iter().any(|a| a.severity == HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        HealthDashboard {
            timestamp_ms: self.timestamp_ms,
            status,
            integrity,
            storage,
            retention,
            alerts,
        }
    }
}

// ---------------------------------------------------------------------------
// Trend tracker
// ---------------------------------------------------------------------------

/// Tracks health metrics over time for trend analysis.
#[derive(Debug, Default)]
pub struct HealthTrend {
    /// Health score over time.
    pub health_score: TimeSeries,
    /// Total files tracked over time.
    pub total_files: TimeSeries,
    /// Corrupted file count over time.
    pub corrupted_count: TimeSeries,
    /// Storage utilization over time.
    pub storage_bytes: TimeSeries,
    /// Dedup ratio over time.
    pub dedup_ratio: TimeSeries,
}

impl HealthTrend {
    /// Create a new empty trend tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            health_score: TimeSeries::new("health_score", "fraction"),
            total_files: TimeSeries::new("total_files", "count"),
            corrupted_count: TimeSeries::new("corrupted_count", "count"),
            storage_bytes: TimeSeries::new("storage_bytes", "bytes"),
            dedup_ratio: TimeSeries::new("dedup_ratio", "ratio"),
        }
    }

    /// Record a dashboard snapshot into the trend data.
    pub fn record(&mut self, dashboard: &HealthDashboard) {
        let ts = dashboard.timestamp_ms;
        self.health_score.add(ts, dashboard.integrity.health_score);
        self.total_files
            .add(ts, dashboard.integrity.total_scanned as f64);
        self.corrupted_count
            .add(ts, dashboard.integrity.corrupted_count as f64);
        self.storage_bytes
            .add(ts, dashboard.storage.logical_bytes as f64);
        self.dedup_ratio.add(ts, dashboard.storage.dedup_ratio);
    }

    /// Get the trend direction for the health score.
    #[must_use]
    pub fn health_trend_direction(&self) -> TrendDirection {
        match self.health_score.trend_slope() {
            None => TrendDirection::Stable,
            Some(slope) if slope > 1e-15 => TrendDirection::Improving,
            Some(slope) if slope < -1e-15 => TrendDirection::Declining,
            Some(_) => TrendDirection::Stable,
        }
    }

    /// Total number of snapshots recorded.
    #[must_use]
    pub fn snapshot_count(&self) -> usize {
        self.health_score.len()
    }
}

/// Direction of a trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Metric is improving.
    Improving,
    /// Metric is stable.
    Stable,
    /// Metric is declining.
    Declining,
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Improving => write!(f, "IMPROVING"),
            Self::Stable => write!(f, "STABLE"),
            Self::Declining => write!(f, "DECLINING"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrity_scan::{FileScanRecord, IntegrityScan};

    #[test]
    fn test_time_series_basic() {
        let mut ts = TimeSeries::new("test", "count");
        ts.add(1000, 10.0);
        ts.add(2000, 20.0);
        ts.add(3000, 15.0);

        assert_eq!(ts.len(), 3);
        assert_eq!(ts.latest_value(), Some(15.0));
        assert_eq!(ts.min_value(), Some(10.0));
        assert_eq!(ts.max_value(), Some(20.0));
        assert!((ts.average().expect("avg") - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_series_empty() {
        let ts = TimeSeries::new("empty", "x");
        assert!(ts.is_empty());
        assert_eq!(ts.latest_value(), None);
        assert_eq!(ts.min_value(), None);
        assert_eq!(ts.average(), None);
        assert_eq!(ts.trend_slope(), None);
    }

    #[test]
    fn test_time_series_trend_slope_increasing() {
        let mut ts = TimeSeries::new("inc", "val");
        ts.add(1, 1.0);
        ts.add(2, 2.0);
        ts.add(3, 3.0);
        let slope = ts.trend_slope().expect("slope");
        assert!(slope > 0.0, "slope should be positive: {slope}");
    }

    #[test]
    fn test_time_series_trend_slope_decreasing() {
        let mut ts = TimeSeries::new("dec", "val");
        ts.add(1, 3.0);
        ts.add(2, 2.0);
        ts.add(3, 1.0);
        let slope = ts.trend_slope().expect("slope");
        assert!(slope < 0.0, "slope should be negative: {slope}");
    }

    #[test]
    fn test_health_status_severity() {
        assert!(HealthStatus::Critical.severity() > HealthStatus::AtRisk.severity());
        assert!(HealthStatus::AtRisk.severity() > HealthStatus::Degraded.severity());
        assert!(HealthStatus::Degraded.severity() > HealthStatus::Healthy.severity());
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "HEALTHY");
        assert_eq!(HealthStatus::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_dashboard_builder_minimal() {
        let dashboard = DashboardBuilder::new(1000).build();
        assert_eq!(dashboard.status, HealthStatus::Healthy);
        assert!(dashboard.alerts.is_empty());
    }

    #[test]
    fn test_dashboard_builder_with_scan() {
        let mut scan = IntegrityScan::with_defaults(0);
        scan.add_record(FileScanRecord::new("/ok.mxf", "aa", "aa", 100, 1));
        scan.add_record(FileScanRecord::new("/bad.mxf", "aa", "xx", 200, 2));
        scan.finish(100);

        let dashboard = DashboardBuilder::new(1000)
            .with_integrity_scan(&scan)
            .build();

        assert_eq!(dashboard.integrity.total_scanned, 2);
        assert_eq!(dashboard.integrity.ok_count, 1);
        assert_eq!(dashboard.integrity.corrupted_count, 1);
        assert_eq!(dashboard.status, HealthStatus::Critical);
        assert!(!dashboard.alerts.is_empty());
    }

    #[test]
    fn test_dashboard_builder_with_dedup() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"content_a", "/a.mxf");
        idx.ingest(b"content_a", "/b.mxf");
        idx.ingest(b"content_b", "/c.mxf");

        let dashboard = DashboardBuilder::new(1000).with_dedup_index(&idx).build();

        assert!(dashboard.storage.dedup_ratio > 1.0);
        assert!(dashboard.storage.bytes_saved > 0);
    }

    #[test]
    fn test_dashboard_builder_with_retention() {
        let mut sched = RetentionSchedule::new();
        sched.add(crate::retention_schedule::RetentionEntry::new(
            "temp-001",
            RetentionClass::Temporary,
            0,
            Some(100),
            false,
        ));
        sched.add(crate::retention_schedule::RetentionEntry::new(
            "perm-001",
            RetentionClass::Permanent,
            0,
            None,
            false,
        ));
        sched.add(crate::retention_schedule::RetentionEntry::new(
            "held-001",
            RetentionClass::Standard,
            0,
            None,
            true,
        ));

        let dashboard = DashboardBuilder::new(5000)
            .with_retention_schedule(&sched, 5000)
            .build();

        assert_eq!(dashboard.retention.total_entries, 3);
        assert_eq!(dashboard.retention.legal_hold_count, 1);
    }

    #[test]
    fn test_dashboard_healthy_no_alerts() {
        let mut scan = IntegrityScan::with_defaults(0);
        for i in 0..10 {
            scan.add_record(FileScanRecord::new(
                format!("/file_{i}.mxf"),
                "aa",
                "aa",
                100,
                1,
            ));
        }
        scan.finish(100);

        let dashboard = DashboardBuilder::new(1000)
            .with_integrity_scan(&scan)
            .build();

        assert_eq!(dashboard.status, HealthStatus::Healthy);
        assert!(dashboard.alerts.is_empty());
    }

    #[test]
    fn test_dashboard_missing_files_alert() {
        let mut scan = IntegrityScan::with_defaults(0);
        scan.add_record(FileScanRecord::new("/ok.mxf", "aa", "aa", 100, 1));
        scan.add_record(FileScanRecord::missing("/gone.mxf", 2));
        scan.finish(100);

        let dashboard = DashboardBuilder::new(1000)
            .with_integrity_scan(&scan)
            .build();

        assert!(dashboard
            .alerts
            .iter()
            .any(|a| a.message.contains("missing")));
    }

    #[test]
    fn test_dashboard_summary_string() {
        let dashboard = DashboardBuilder::new(1000).build();
        let summary = dashboard.to_summary_string();
        assert!(summary.contains("Archive Health Dashboard"));
        assert!(summary.contains("HEALTHY"));
    }

    #[test]
    fn test_health_trend_record() {
        let mut trend = HealthTrend::new();

        let d1 = DashboardBuilder::new(1000).build();
        let d2 = DashboardBuilder::new(2000).build();
        trend.record(&d1);
        trend.record(&d2);

        assert_eq!(trend.snapshot_count(), 2);
    }

    #[test]
    fn test_health_trend_improving() {
        let mut trend = HealthTrend::new();

        // Simulate improving health scores
        for i in 0..5 {
            let section = IntegritySection {
                health_score: 0.8 + (i as f64 * 0.05),
                total_scanned: 100,
                ok_count: 80 + i * 5,
                corrupted_count: 20 - i * 5,
                missing_count: 0,
                modified_count: 0,
                total_bytes_scanned: 1000,
                last_scan_duration_ms: 100,
            };
            let dashboard = DashboardBuilder::new((i + 1) as u64 * 1000)
                .with_integrity(section)
                .build();
            trend.record(&dashboard);
        }

        assert_eq!(trend.health_trend_direction(), TrendDirection::Improving);
    }

    #[test]
    fn test_health_trend_declining() {
        let mut trend = HealthTrend::new();

        for i in 0..5 {
            let section = IntegritySection {
                health_score: 1.0 - (i as f64 * 0.1),
                total_scanned: 100,
                ok_count: 100 - i * 10,
                corrupted_count: i * 10,
                missing_count: 0,
                modified_count: 0,
                total_bytes_scanned: 1000,
                last_scan_duration_ms: 100,
            };
            let dashboard = DashboardBuilder::new((i + 1) as u64 * 1000)
                .with_integrity(section)
                .build();
            trend.record(&dashboard);
        }

        assert_eq!(trend.health_trend_direction(), TrendDirection::Declining);
    }

    #[test]
    fn test_health_trend_stable() {
        let mut trend = HealthTrend::new();

        for i in 0..5 {
            let section = IntegritySection {
                health_score: 1.0,
                total_scanned: 100,
                ok_count: 100,
                corrupted_count: 0,
                missing_count: 0,
                modified_count: 0,
                total_bytes_scanned: 1000,
                last_scan_duration_ms: 100,
            };
            let dashboard = DashboardBuilder::new((i + 1) as u64 * 1000)
                .with_integrity(section)
                .build();
            trend.record(&dashboard);
        }

        assert_eq!(trend.health_trend_direction(), TrendDirection::Stable);
    }

    #[test]
    fn test_trend_direction_display() {
        assert_eq!(TrendDirection::Improving.to_string(), "IMPROVING");
        assert_eq!(TrendDirection::Stable.to_string(), "STABLE");
        assert_eq!(TrendDirection::Declining.to_string(), "DECLINING");
    }

    #[test]
    fn test_integrity_section_from_metrics() {
        let metrics = ScanHealthMetrics {
            total_scanned: 100,
            ok_count: 95,
            corrupted_count: 3,
            missing_count: 2,
            modified_count: 0,
            total_bytes_scanned: 1_000_000,
            duration_ms: 500,
        };
        let section = IntegritySection::from(&metrics);
        assert_eq!(section.total_scanned, 100);
        assert_eq!(section.corrupted_count, 3);
        assert!((section.health_score - 0.95).abs() < f64::EPSILON);
    }
}
