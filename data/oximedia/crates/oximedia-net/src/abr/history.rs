//! Segment download history and rolling window bandwidth statistics.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Segment download history with rolling window statistics
// ─────────────────────────────────────────────────────────────────────────────

/// A single segment download record.
#[derive(Debug, Clone, Copy)]
pub struct SegmentDownloadRecord {
    /// Segment sequence number.
    pub sequence: u64,
    /// Quality level index of the downloaded segment.
    pub quality_index: usize,
    /// Bytes downloaded.
    pub bytes: usize,
    /// Download duration.
    pub duration: Duration,
    /// Segment playback duration (how much media time it represents).
    pub segment_duration: Duration,
    /// Timestamp when download completed.
    pub timestamp: Instant,
    /// Computed throughput in bytes/sec.
    pub throughput: f64,
}

impl SegmentDownloadRecord {
    /// Creates a new segment record.
    #[must_use]
    pub fn new(
        sequence: u64,
        quality_index: usize,
        bytes: usize,
        duration: Duration,
        segment_duration: Duration,
    ) -> Self {
        let throughput = if duration.as_secs_f64() > 0.0 {
            bytes as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        Self {
            sequence,
            quality_index,
            bytes,
            duration,
            segment_duration,
            timestamp: Instant::now(),
            throughput,
        }
    }
}

/// Rolling window statistics over recent segment downloads.
#[derive(Debug, Clone, Default)]
pub struct DownloadWindowStats {
    /// Mean throughput (bytes/sec) over the window.
    pub mean_throughput: f64,
    /// Standard deviation of throughput.
    pub std_throughput: f64,
    /// Coefficient of variation (std/mean) — higher means more variable network.
    pub cv_throughput: f64,
    /// Minimum throughput (bytes/sec) in the window.
    pub min_throughput: f64,
    /// Maximum throughput (bytes/sec) in the window.
    pub max_throughput: f64,
    /// Number of samples in the window.
    pub count: usize,
    /// Total bytes downloaded in the window.
    pub total_bytes: u64,
    /// Percentage of time spent downloading (download duration / wall time).
    pub download_ratio: f64,
}

/// History of segment downloads with rolling window statistical analysis.
#[derive(Debug)]
pub struct SegmentDownloadHistory {
    /// Stored records, newest at the back.
    records: VecDeque<SegmentDownloadRecord>,
    /// Maximum number of records to retain.
    capacity: usize,
    /// Next sequence number to assign.
    next_sequence: u64,
}

impl SegmentDownloadHistory {
    /// Creates a new download history with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            next_sequence: 0,
        }
    }

    /// Adds a new download record.
    pub fn add(
        &mut self,
        quality_index: usize,
        bytes: usize,
        duration: Duration,
        segment_duration: Duration,
    ) -> u64 {
        let seq = self.next_sequence;
        self.next_sequence += 1;

        let record =
            SegmentDownloadRecord::new(seq, quality_index, bytes, duration, segment_duration);
        if self.records.len() >= self.capacity {
            self.records.pop_front();
        }
        self.records.push_back(record);
        seq
    }

    /// Returns statistics over the most recent `window` records.
    #[must_use]
    pub fn stats(&self, window: usize) -> DownloadWindowStats {
        if self.records.is_empty() {
            return DownloadWindowStats::default();
        }

        let take = window.min(self.records.len());
        let start = self.records.len() - take;
        let recent: Vec<&SegmentDownloadRecord> = self.records.iter().skip(start).collect();

        let throughputs: Vec<f64> = recent.iter().map(|r| r.throughput).collect();
        let count = throughputs.len();
        let mean = throughputs.iter().sum::<f64>() / count as f64;
        let variance =
            throughputs.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / count.max(2) as f64;
        let std = variance.sqrt();
        let cv = if mean > 0.0 { std / mean } else { 0.0 };
        let min = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = throughputs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let total_bytes: u64 = recent.iter().map(|r| r.bytes as u64).sum();

        // Download ratio: total download time / elapsed wall time
        let total_download_secs: f64 = recent.iter().map(|r| r.duration.as_secs_f64()).sum();
        let wall_time = if let (Some(first), Some(last)) = (recent.first(), recent.last()) {
            let elapsed = last.timestamp.duration_since(first.timestamp).as_secs_f64();
            elapsed.max(total_download_secs)
        } else {
            total_download_secs
        };
        let download_ratio = if wall_time > 0.0 {
            (total_download_secs / wall_time).min(1.0)
        } else {
            0.0
        };

        DownloadWindowStats {
            mean_throughput: mean,
            std_throughput: std,
            cv_throughput: cv,
            min_throughput: min.max(0.0),
            max_throughput: max.max(0.0),
            count,
            total_bytes,
            download_ratio,
        }
    }

    /// Returns all stored records (oldest first).
    #[must_use]
    pub fn records(&self) -> &VecDeque<SegmentDownloadRecord> {
        &self.records
    }

    /// Returns the number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if no records exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Resets the history.
    pub fn reset(&mut self) {
        self.records.clear();
        self.next_sequence = 0;
    }
}
