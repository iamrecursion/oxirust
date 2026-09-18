#![allow(dead_code)]
//! Transfer statistics — recording upload/download events and computing throughput metrics.

use std::time::{Duration, SystemTime};

/// Direction of a data transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    /// Data flowing from client to store.
    Upload,
    /// Data flowing from store to client.
    Download,
    /// Internal copy or replication.
    Copy,
}

impl TransferDirection {
    /// Returns a short label string.
    pub fn label(&self) -> &'static str {
        match self {
            TransferDirection::Upload => "upload",
            TransferDirection::Download => "download",
            TransferDirection::Copy => "copy",
        }
    }
}

/// A single transfer event record.
#[derive(Debug, Clone)]
pub struct TransferRecord {
    /// Unique identifier for this transfer.
    pub id: String,
    /// Direction of the transfer.
    pub direction: TransferDirection,
    /// Number of bytes transferred.
    pub bytes: u64,
    /// Duration of the transfer.
    pub duration: Duration,
    /// When the transfer started.
    pub started_at: SystemTime,
    /// Whether the transfer succeeded.
    pub success: bool,
    /// Object key involved.
    pub key: String,
}

impl TransferRecord {
    /// Create a new transfer record.
    pub fn new(
        id: impl Into<String>,
        direction: TransferDirection,
        key: impl Into<String>,
        bytes: u64,
        duration: Duration,
        success: bool,
    ) -> Self {
        Self {
            id: id.into(),
            direction,
            key: key.into(),
            bytes,
            duration,
            started_at: SystemTime::now(),
            success,
        }
    }

    /// Throughput in megabytes per second. Returns 0.0 if duration is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_mbps(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        (self.bytes as f64) / secs / (1024.0 * 1024.0)
    }

    /// Returns throughput in kbps (kilobytes per second).
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_kbps(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        (self.bytes as f64) / secs / 1024.0
    }
}

/// Aggregated transfer statistics over a collection of records.
#[derive(Debug, Default, Clone)]
pub struct TransferStats {
    records: Vec<TransferRecord>,
}

impl TransferStats {
    /// Create a new empty stats collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a transfer record.
    pub fn add_record(&mut self, record: TransferRecord) {
        self.records.push(record);
    }

    /// Total number of recorded transfers.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Number of successful transfers.
    pub fn success_count(&self) -> usize {
        self.records.iter().filter(|r| r.success).count()
    }

    /// Number of failed transfers.
    pub fn failure_count(&self) -> usize {
        self.records.iter().filter(|r| !r.success).count()
    }

    /// Total bytes transferred (successful + failed combined).
    pub fn total_bytes(&self) -> u64 {
        self.records.iter().map(|r| r.bytes).sum()
    }

    /// Total bytes for successful transfers only.
    pub fn successful_bytes(&self) -> u64 {
        self.records
            .iter()
            .filter(|r| r.success)
            .map(|r| r.bytes)
            .sum()
    }

    /// Average throughput in MB/s across all records. Returns 0.0 if no records.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_throughput_mbps(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .records
            .iter()
            .map(TransferRecord::throughput_mbps)
            .sum();
        total / self.records.len() as f64
    }

    /// Peak throughput in MB/s across all records.
    pub fn peak_throughput_mbps(&self) -> f64 {
        self.records
            .iter()
            .map(TransferRecord::throughput_mbps)
            .fold(0.0_f64, f64::max)
    }

    /// Total wall-clock time spent on transfers.
    pub fn total_duration(&self) -> Duration {
        self.records.iter().map(|r| r.duration).sum()
    }

    /// Success rate as a fraction [0.0, 1.0]. Returns 0.0 if no records.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        self.success_count() as f64 / self.records.len() as f64
    }

    /// Filter records by direction.
    pub fn records_by_direction(&self, dir: TransferDirection) -> Vec<&TransferRecord> {
        self.records.iter().filter(|r| r.direction == dir).collect()
    }

    /// Clear all recorded data.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_record(id: &str, bytes: u64, millis: u64, success: bool) -> TransferRecord {
        TransferRecord::new(
            id,
            TransferDirection::Upload,
            "test/key",
            bytes,
            Duration::from_millis(millis),
            success,
        )
    }

    #[test]
    fn test_transfer_direction_label() {
        assert_eq!(TransferDirection::Upload.label(), "upload");
        assert_eq!(TransferDirection::Download.label(), "download");
        assert_eq!(TransferDirection::Copy.label(), "copy");
    }

    #[test]
    fn test_throughput_mbps_basic() {
        // 10 MB in 1 second => 10 MB/s
        let r = make_record("r1", 10 * 1024 * 1024, 1000, true);
        let tp = r.throughput_mbps();
        assert!((tp - 10.0).abs() < 0.01, "expected ~10 MB/s, got {tp}");
    }

    #[test]
    fn test_throughput_mbps_zero_duration() {
        let r = make_record("r2", 1024, 0, true);
        assert_eq!(r.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_throughput_kbps() {
        // 1 MB in 1 second => 1024 KB/s
        let r = make_record("r3", 1024 * 1024, 1000, true);
        let kbps = r.throughput_kbps();
        assert!((kbps - 1024.0).abs() < 0.1);
    }

    #[test]
    fn test_transfer_stats_empty() {
        let stats = TransferStats::new();
        assert_eq!(stats.record_count(), 0);
        assert_eq!(stats.total_bytes(), 0);
        assert_eq!(stats.avg_throughput_mbps(), 0.0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_transfer_stats_add_record() {
        let mut stats = TransferStats::new();
        stats.add_record(make_record("a", 1024, 500, true));
        assert_eq!(stats.record_count(), 1);
    }

    #[test]
    fn test_total_bytes() {
        let mut stats = TransferStats::new();
        stats.add_record(make_record("a", 100, 100, true));
        stats.add_record(make_record("b", 200, 200, false));
        assert_eq!(stats.total_bytes(), 300);
    }

    #[test]
    fn test_successful_bytes() {
        let mut stats = TransferStats::new();
        stats.add_record(make_record("a", 100, 100, true));
        stats.add_record(make_record("b", 200, 200, false));
        assert_eq!(stats.successful_bytes(), 100);
    }

    #[test]
    fn test_success_and_failure_count() {
        let mut stats = TransferStats::new();
        stats.add_record(make_record("a", 10, 10, true));
        stats.add_record(make_record("b", 10, 10, true));
        stats.add_record(make_record("c", 10, 10, false));
        assert_eq!(stats.success_count(), 2);
        assert_eq!(stats.failure_count(), 1);
    }

    #[test]
    fn test_success_rate() {
        let mut stats = TransferStats::new();
        stats.add_record(make_record("a", 10, 10, true));
        stats.add_record(make_record("b", 10, 10, false));
        let rate = stats.success_rate();
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_avg_throughput_mbps() {
        let mut stats = TransferStats::new();
        // 10 MB in 1s => 10 MB/s
        stats.add_record(TransferRecord::new(
            "x",
            TransferDirection::Download,
            "k",
            10 * 1024 * 1024,
            Duration::from_secs(1),
            true,
        ));
        // 20 MB in 1s => 20 MB/s
        stats.add_record(TransferRecord::new(
            "y",
            TransferDirection::Download,
            "k2",
            20 * 1024 * 1024,
            Duration::from_secs(1),
            true,
        ));
        let avg = stats.avg_throughput_mbps();
        assert!((avg - 15.0).abs() < 0.1, "expected ~15 MB/s avg, got {avg}");
    }

    #[test]
    fn test_peak_throughput_mbps() {
        let mut stats = TransferStats::new();
        stats.add_record(make_record("a", 10 * 1024 * 1024, 1000, true));
        stats.add_record(make_record("b", 50 * 1024 * 1024, 1000, true));
        let peak = stats.peak_throughput_mbps();
        assert!(
            (peak - 50.0).abs() < 0.1,
            "expected ~50 MB/s peak, got {peak}"
        );
    }

    #[test]
    fn test_records_by_direction() {
        let mut stats = TransferStats::new();
        stats.add_record(make_record("a", 10, 10, true)); // Upload
        stats.add_record(TransferRecord::new(
            "b",
            TransferDirection::Download,
            "k",
            10,
            Duration::from_millis(10),
            true,
        ));
        assert_eq!(
            stats.records_by_direction(TransferDirection::Upload).len(),
            1
        );
        assert_eq!(
            stats
                .records_by_direction(TransferDirection::Download)
                .len(),
            1
        );
        assert_eq!(stats.records_by_direction(TransferDirection::Copy).len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut stats = TransferStats::new();
        stats.add_record(make_record("a", 100, 100, true));
        stats.clear();
        assert_eq!(stats.record_count(), 0);
        assert_eq!(stats.total_bytes(), 0);
    }
}
