#![allow(dead_code)]

//! Archive statistics and reporting.
//!
//! Collects and summarises statistics about archived assets, including
//! format distribution, size metrics, integrity check results, and trends.

use std::collections::HashMap;
use std::fmt;

/// The overall health status of an archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveHealth {
    /// All assets pass integrity checks.
    Healthy,
    /// Some assets have warnings (e.g. approaching retention limits).
    Warning,
    /// One or more assets have integrity failures.
    Degraded,
    /// Archive is in a critical state.
    Critical,
}

impl fmt::Display for ArchiveHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Warning => write!(f, "Warning"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Size bucket for categorising assets by file size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SizeBucket {
    /// Less than 1 MB.
    Tiny,
    /// 1 MB to 100 MB.
    Small,
    /// 100 MB to 1 GB.
    Medium,
    /// 1 GB to 10 GB.
    Large,
    /// More than 10 GB.
    Huge,
}

impl SizeBucket {
    /// Classify a byte count into a bucket.
    pub fn from_bytes(bytes: u64) -> Self {
        const MB: u64 = 1_000_000;
        const GB: u64 = 1_000_000_000;
        if bytes < MB {
            Self::Tiny
        } else if bytes < 100 * MB {
            Self::Small
        } else if bytes < GB {
            Self::Medium
        } else if bytes < 10 * GB {
            Self::Large
        } else {
            Self::Huge
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Tiny => "<1 MB",
            Self::Small => "1–100 MB",
            Self::Medium => "100 MB–1 GB",
            Self::Large => "1–10 GB",
            Self::Huge => ">10 GB",
        }
    }
}

/// Statistics about format distribution in the archive.
#[derive(Debug, Clone, Default)]
pub struct FormatDistribution {
    /// Map from format name to asset count.
    counts: HashMap<String, u64>,
    /// Map from format name to total bytes.
    bytes: HashMap<String, u64>,
}

impl FormatDistribution {
    /// Create a new empty distribution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an asset with the given format and size.
    pub fn record(&mut self, format: impl Into<String>, size_bytes: u64) {
        let fmt = format.into();
        *self.counts.entry(fmt.clone()).or_insert(0) += 1;
        *self.bytes.entry(fmt).or_insert(0) += size_bytes;
    }

    /// Get the asset count for a format.
    pub fn count(&self, format: &str) -> u64 {
        self.counts.get(format).copied().unwrap_or(0)
    }

    /// Get the total bytes for a format.
    pub fn bytes(&self, format: &str) -> u64 {
        self.bytes.get(format).copied().unwrap_or(0)
    }

    /// Total number of distinct formats.
    pub fn format_count(&self) -> usize {
        self.counts.len()
    }

    /// Total asset count across all formats.
    pub fn total_assets(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Total bytes across all formats.
    pub fn total_bytes(&self) -> u64 {
        self.bytes.values().sum()
    }

    /// Return formats sorted by count descending.
    pub fn top_formats(&self, limit: usize) -> Vec<(String, u64)> {
        let mut pairs: Vec<_> = self.counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(limit);
        pairs
    }

    /// Percentage of total assets that use a given format.
    #[allow(clippy::cast_precision_loss)]
    pub fn percentage(&self, format: &str) -> f64 {
        let total = self.total_assets();
        if total == 0 {
            return 0.0;
        }
        let count = self.count(format);
        count as f64 / total as f64 * 100.0
    }
}

/// Integrity check result counters.
#[derive(Debug, Clone, Default)]
pub struct IntegrityStats {
    /// Number of assets that passed integrity checks.
    pub passed: u64,
    /// Number of assets that failed integrity checks.
    pub failed: u64,
    /// Number of assets skipped (e.g. unavailable).
    pub skipped: u64,
    /// Number of assets not yet checked.
    pub unchecked: u64,
}

impl IntegrityStats {
    /// Create a new empty stats struct.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total assets considered.
    pub fn total(&self) -> u64 {
        self.passed + self.failed + self.skipped + self.unchecked
    }

    /// Pass rate as a percentage.
    #[allow(clippy::cast_precision_loss)]
    pub fn pass_rate(&self) -> f64 {
        let checked = self.passed + self.failed;
        if checked == 0 {
            return 0.0;
        }
        self.passed as f64 / checked as f64 * 100.0
    }

    /// Failure rate as a percentage.
    #[allow(clippy::cast_precision_loss)]
    pub fn fail_rate(&self) -> f64 {
        let checked = self.passed + self.failed;
        if checked == 0 {
            return 0.0;
        }
        self.failed as f64 / checked as f64 * 100.0
    }

    /// Determine the health status from these stats.
    pub fn health(&self) -> ArchiveHealth {
        if self.failed == 0 && self.unchecked == 0 {
            ArchiveHealth::Healthy
        } else if self.failed == 0 {
            ArchiveHealth::Warning
        } else if self.fail_rate() < 5.0 {
            ArchiveHealth::Degraded
        } else {
            ArchiveHealth::Critical
        }
    }
}

/// A complete snapshot of archive statistics.
#[derive(Debug, Clone)]
pub struct ArchiveSnapshot {
    /// Snapshot label or identifier.
    pub label: String,
    /// Total number of assets.
    pub total_assets: u64,
    /// Total storage bytes used.
    pub total_bytes: u64,
    /// Format distribution.
    pub format_dist: FormatDistribution,
    /// Size distribution buckets.
    pub size_dist: HashMap<SizeBucket, u64>,
    /// Integrity stats.
    pub integrity: IntegrityStats,
}

impl ArchiveSnapshot {
    /// Create a new empty snapshot with a label.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            total_assets: 0,
            total_bytes: 0,
            format_dist: FormatDistribution::new(),
            size_dist: HashMap::new(),
            integrity: IntegrityStats::new(),
        }
    }

    /// Record an asset into this snapshot.
    pub fn record_asset(&mut self, format: impl Into<String>, size_bytes: u64) {
        self.total_assets += 1;
        self.total_bytes += size_bytes;
        self.format_dist.record(format, size_bytes);
        let bucket = SizeBucket::from_bytes(size_bytes);
        *self.size_dist.entry(bucket).or_insert(0) += 1;
    }

    /// Average asset size in bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_asset_bytes(&self) -> f64 {
        if self.total_assets == 0 {
            return 0.0;
        }
        self.total_bytes as f64 / self.total_assets as f64
    }

    /// Overall health derived from integrity stats.
    pub fn health(&self) -> ArchiveHealth {
        self.integrity.health()
    }

    /// How many distinct formats are in this snapshot.
    pub fn distinct_formats(&self) -> usize {
        self.format_dist.format_count()
    }

    /// Total bytes as human-readable string (e.g. "1.23 GB").
    #[allow(clippy::cast_precision_loss)]
    pub fn human_total_bytes(&self) -> String {
        const KB: f64 = 1_000.0;
        const MB: f64 = 1_000_000.0;
        const GB: f64 = 1_000_000_000.0;
        const TB: f64 = 1_000_000_000_000.0;
        let b = self.total_bytes as f64;
        if b >= TB {
            format!("{:.2} TB", b / TB)
        } else if b >= GB {
            format!("{:.2} GB", b / GB)
        } else if b >= MB {
            format!("{:.2} MB", b / MB)
        } else if b >= KB {
            format!("{:.2} KB", b / KB)
        } else {
            format!("{b} B")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bucket_tiny() {
        assert_eq!(SizeBucket::from_bytes(500), SizeBucket::Tiny);
    }

    #[test]
    fn test_size_bucket_small() {
        assert_eq!(SizeBucket::from_bytes(5_000_000), SizeBucket::Small);
    }

    #[test]
    fn test_size_bucket_medium() {
        assert_eq!(SizeBucket::from_bytes(500_000_000), SizeBucket::Medium);
    }

    #[test]
    fn test_size_bucket_large() {
        assert_eq!(SizeBucket::from_bytes(5_000_000_000), SizeBucket::Large);
    }

    #[test]
    fn test_size_bucket_huge() {
        assert_eq!(SizeBucket::from_bytes(50_000_000_000), SizeBucket::Huge);
    }

    #[test]
    fn test_size_bucket_label() {
        assert_eq!(SizeBucket::Tiny.label(), "<1 MB");
        assert_eq!(SizeBucket::Huge.label(), ">10 GB");
    }

    #[test]
    fn test_format_dist_record() {
        let mut fd = FormatDistribution::new();
        fd.record("mkv", 1000);
        fd.record("mkv", 2000);
        fd.record("flac", 500);
        assert_eq!(fd.count("mkv"), 2);
        assert_eq!(fd.bytes("mkv"), 3000);
        assert_eq!(fd.count("flac"), 1);
    }

    #[test]
    fn test_format_dist_totals() {
        let mut fd = FormatDistribution::new();
        fd.record("mkv", 1000);
        fd.record("flac", 500);
        assert_eq!(fd.total_assets(), 2);
        assert_eq!(fd.total_bytes(), 1500);
        assert_eq!(fd.format_count(), 2);
    }

    #[test]
    fn test_format_dist_top() {
        let mut fd = FormatDistribution::new();
        fd.record("mkv", 100);
        fd.record("mkv", 100);
        fd.record("flac", 100);
        let top = fd.top_formats(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "mkv");
    }

    #[test]
    fn test_format_dist_percentage() {
        let mut fd = FormatDistribution::new();
        fd.record("mkv", 100);
        fd.record("flac", 100);
        assert!((fd.percentage("mkv") - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_integrity_pass_rate() {
        let mut is = IntegrityStats::new();
        is.passed = 90;
        is.failed = 10;
        assert!((is.pass_rate() - 90.0).abs() < f64::EPSILON);
        assert!((is.fail_rate() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_integrity_health_healthy() {
        let is = IntegrityStats {
            passed: 100,
            failed: 0,
            skipped: 0,
            unchecked: 0,
        };
        assert_eq!(is.health(), ArchiveHealth::Healthy);
    }

    #[test]
    fn test_integrity_health_warning() {
        let is = IntegrityStats {
            passed: 90,
            failed: 0,
            skipped: 0,
            unchecked: 10,
        };
        assert_eq!(is.health(), ArchiveHealth::Warning);
    }

    #[test]
    fn test_integrity_health_degraded() {
        let is = IntegrityStats {
            passed: 99,
            failed: 1,
            skipped: 0,
            unchecked: 0,
        };
        assert_eq!(is.health(), ArchiveHealth::Degraded);
    }

    #[test]
    fn test_integrity_health_critical() {
        let is = IntegrityStats {
            passed: 80,
            failed: 20,
            skipped: 0,
            unchecked: 0,
        };
        assert_eq!(is.health(), ArchiveHealth::Critical);
    }

    #[test]
    fn test_snapshot_record() {
        let mut snap = ArchiveSnapshot::new("test");
        snap.record_asset("mkv", 1_000_000);
        snap.record_asset("flac", 500_000);
        assert_eq!(snap.total_assets, 2);
        assert_eq!(snap.total_bytes, 1_500_000);
        assert_eq!(snap.distinct_formats(), 2);
    }

    #[test]
    fn test_snapshot_avg() {
        let mut snap = ArchiveSnapshot::new("test");
        snap.record_asset("mkv", 1000);
        snap.record_asset("mkv", 3000);
        assert!((snap.avg_asset_bytes() - 2000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_snapshot_human_bytes() {
        let mut snap = ArchiveSnapshot::new("t");
        snap.total_bytes = 1_500_000_000;
        assert_eq!(snap.human_total_bytes(), "1.50 GB");
    }

    #[test]
    fn test_archive_health_display() {
        assert_eq!(ArchiveHealth::Healthy.to_string(), "Healthy");
        assert_eq!(ArchiveHealth::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_empty_integrity_rates() {
        let is = IntegrityStats::new();
        assert!((is.pass_rate() - 0.0).abs() < f64::EPSILON);
        assert!((is.fail_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_snapshot_avg() {
        let snap = ArchiveSnapshot::new("empty");
        assert!((snap.avg_asset_bytes() - 0.0).abs() < f64::EPSILON);
    }
}
