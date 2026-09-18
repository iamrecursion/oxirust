#![allow(dead_code)]
//! Performance statistics and counters for the acceleration layer.
//!
//! Tracks kernel execution times, memory transfer throughput,
//! task completion rates, and other metrics for profiling and
//! monitoring GPU/CPU workloads.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single recorded timing measurement for a kernel or operation.
#[derive(Debug, Clone, Copy)]
pub struct TimingSample {
    /// Duration of the operation.
    pub duration: Duration,
    /// Timestamp when the operation started.
    pub started_at: Instant,
}

impl TimingSample {
    /// Creates a new timing sample.
    #[must_use]
    pub fn new(duration: Duration, started_at: Instant) -> Self {
        Self {
            duration,
            started_at,
        }
    }

    /// Returns the duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }

    /// Returns the duration in microseconds.
    #[must_use]
    pub fn duration_us(&self) -> f64 {
        self.duration.as_secs_f64() * 1_000_000.0
    }
}

/// Rolling statistics tracker for a named operation.
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Operation name.
    pub name: String,
    /// Total number of invocations.
    pub invocation_count: u64,
    /// Total cumulative duration.
    pub total_duration: Duration,
    /// Minimum observed duration.
    pub min_duration: Option<Duration>,
    /// Maximum observed duration.
    pub max_duration: Option<Duration>,
    /// Recent timing samples (bounded).
    recent_samples: Vec<TimingSample>,
    /// Maximum number of recent samples to keep.
    max_samples: usize,
}

impl OperationStats {
    /// Creates a new operation statistics tracker.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            invocation_count: 0,
            total_duration: Duration::ZERO,
            min_duration: None,
            max_duration: None,
            recent_samples: Vec::new(),
            max_samples: 100,
        }
    }

    /// Creates a new tracker with a custom sample buffer size.
    #[must_use]
    pub fn with_max_samples(name: &str, max_samples: usize) -> Self {
        Self {
            max_samples: max_samples.max(1),
            ..Self::new(name)
        }
    }

    /// Records a completed operation.
    pub fn record(&mut self, duration: Duration, started_at: Instant) {
        self.invocation_count += 1;
        self.total_duration += duration;

        self.min_duration = Some(match self.min_duration {
            Some(min) => min.min(duration),
            None => duration,
        });
        self.max_duration = Some(match self.max_duration {
            Some(max) => max.max(duration),
            None => duration,
        });

        if self.recent_samples.len() >= self.max_samples {
            self.recent_samples.remove(0);
        }
        self.recent_samples
            .push(TimingSample::new(duration, started_at));
    }

    /// Returns the average duration, or `None` if no samples recorded.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_duration(&self) -> Option<Duration> {
        if self.invocation_count == 0 {
            return None;
        }
        Some(self.total_duration / self.invocation_count as u32)
    }

    /// Returns the average duration in milliseconds, or `None` if no samples.
    #[must_use]
    pub fn average_ms(&self) -> Option<f64> {
        self.average_duration().map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Returns the minimum duration in milliseconds, or `None`.
    #[must_use]
    pub fn min_ms(&self) -> Option<f64> {
        self.min_duration.map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Returns the maximum duration in milliseconds, or `None`.
    #[must_use]
    pub fn max_ms(&self) -> Option<f64> {
        self.max_duration.map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Returns the number of recent samples.
    #[must_use]
    pub fn recent_count(&self) -> usize {
        self.recent_samples.len()
    }

    /// Returns the median of the recent samples, or `None` if empty.
    #[must_use]
    pub fn recent_median_ms(&self) -> Option<f64> {
        if self.recent_samples.is_empty() {
            return None;
        }
        let mut durations: Vec<f64> = self
            .recent_samples
            .iter()
            .map(TimingSample::duration_ms)
            .collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = durations.len() / 2;
        if durations.len() % 2 == 0 && durations.len() >= 2 {
            Some((durations[mid - 1] + durations[mid]) / 2.0)
        } else {
            Some(durations[mid])
        }
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.invocation_count = 0;
        self.total_duration = Duration::ZERO;
        self.min_duration = None;
        self.max_duration = None;
        self.recent_samples.clear();
    }
}

/// Memory transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    /// Host (CPU) to device (GPU).
    HostToDevice,
    /// Device (GPU) to host (CPU).
    DeviceToHost,
    /// Device to device (intra-GPU or multi-GPU).
    DeviceToDevice,
}

impl TransferDirection {
    /// Returns a short label for the direction.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::HostToDevice => "H2D",
            Self::DeviceToHost => "D2H",
            Self::DeviceToDevice => "D2D",
        }
    }
}

/// Tracks memory transfer statistics.
#[derive(Debug, Clone)]
pub struct TransferStats {
    /// Transfer direction.
    pub direction: TransferDirection,
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Total number of transfers.
    pub transfer_count: u64,
    /// Total time spent on transfers.
    pub total_duration: Duration,
}

impl TransferStats {
    /// Creates a new transfer statistics tracker.
    #[must_use]
    pub fn new(direction: TransferDirection) -> Self {
        Self {
            direction,
            total_bytes: 0,
            transfer_count: 0,
            total_duration: Duration::ZERO,
        }
    }

    /// Records a completed transfer.
    pub fn record(&mut self, bytes: u64, duration: Duration) {
        self.total_bytes += bytes;
        self.transfer_count += 1;
        self.total_duration += duration;
    }

    /// Returns the average throughput in bytes per second, or `None`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn throughput_bps(&self) -> Option<f64> {
        let secs = self.total_duration.as_secs_f64();
        if secs <= 0.0 {
            return None;
        }
        Some(self.total_bytes as f64 / secs)
    }

    /// Returns the throughput in megabytes per second, or `None`.
    #[must_use]
    pub fn throughput_mbps(&self) -> Option<f64> {
        self.throughput_bps().map(|bps| bps / (1024.0 * 1024.0))
    }

    /// Returns the average transfer size in bytes, or `None`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_transfer_size(&self) -> Option<f64> {
        if self.transfer_count == 0 {
            return None;
        }
        Some(self.total_bytes as f64 / self.transfer_count as f64)
    }

    /// Resets all transfer statistics.
    pub fn reset(&mut self) {
        self.total_bytes = 0;
        self.transfer_count = 0;
        self.total_duration = Duration::ZERO;
    }
}

/// Aggregated acceleration statistics across all operations.
#[derive(Debug)]
pub struct AccelStatistics {
    /// Per-operation timing statistics.
    operations: HashMap<String, OperationStats>,
    /// Per-direction transfer statistics.
    transfers: HashMap<TransferDirection, TransferStats>,
    /// Total tasks submitted.
    pub tasks_submitted: u64,
    /// Total tasks completed successfully.
    pub tasks_completed: u64,
    /// Total tasks that failed.
    pub tasks_failed: u64,
    /// Timestamp when stats collection started.
    pub started_at: Instant,
}

impl AccelStatistics {
    /// Creates a new statistics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            transfers: HashMap::new(),
            tasks_submitted: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            started_at: Instant::now(),
        }
    }

    /// Records an operation timing.
    pub fn record_operation(&mut self, name: &str, duration: Duration, started_at: Instant) {
        self.operations
            .entry(name.to_string())
            .or_insert_with(|| OperationStats::new(name))
            .record(duration, started_at);
    }

    /// Records a memory transfer.
    pub fn record_transfer(
        &mut self,
        direction: TransferDirection,
        bytes: u64,
        duration: Duration,
    ) {
        self.transfers
            .entry(direction)
            .or_insert_with(|| TransferStats::new(direction))
            .record(bytes, duration);
    }

    /// Increments the submitted task counter.
    pub fn record_task_submitted(&mut self) {
        self.tasks_submitted += 1;
    }

    /// Increments the completed task counter.
    pub fn record_task_completed(&mut self) {
        self.tasks_completed += 1;
    }

    /// Increments the failed task counter.
    pub fn record_task_failed(&mut self) {
        self.tasks_failed += 1;
    }

    /// Returns how long statistics have been collected.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns the task success rate (0.0 to 1.0), or `None` if no tasks.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn success_rate(&self) -> Option<f64> {
        let total = self.tasks_completed + self.tasks_failed;
        if total == 0 {
            return None;
        }
        Some(self.tasks_completed as f64 / total as f64)
    }

    /// Returns statistics for a named operation, if available.
    #[must_use]
    pub fn get_operation(&self, name: &str) -> Option<&OperationStats> {
        self.operations.get(name)
    }

    /// Returns transfer statistics for a direction, if available.
    #[must_use]
    pub fn get_transfer(&self, direction: TransferDirection) -> Option<&TransferStats> {
        self.transfers.get(&direction)
    }

    /// Returns the names of all tracked operations.
    #[must_use]
    pub fn operation_names(&self) -> Vec<&str> {
        self.operations.keys().map(String::as_str).collect()
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.operations.clear();
        self.transfers.clear();
        self.tasks_submitted = 0;
        self.tasks_completed = 0;
        self.tasks_failed = 0;
        self.started_at = Instant::now();
    }
}

impl Default for AccelStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Profiling / Timing Overlay
// ──────────────────────────────────────────────────────────────────────────────

/// A scope-based timer that records elapsed time when dropped.
///
/// Typical usage:
/// ```ignore
/// let _timer = profiler.start_timer("scale_image");
/// // ... operation ...
/// // timer automatically records on drop
/// ```
pub struct ScopedTimer {
    operation_name: String,
    started_at: Instant,
    /// Channel to send the result back to the profiler.
    /// We use a callback approach since we can't hold &mut on the profiler.
    result_sender: Option<Box<dyn FnOnce(String, Duration, Instant) + Send>>,
}

impl ScopedTimer {
    /// Manually stop the timer and return the elapsed duration.
    /// After calling this, the drop handler will not record again.
    pub fn stop(mut self) -> Duration {
        let elapsed = self.started_at.elapsed();
        if let Some(sender) = self.result_sender.take() {
            sender(self.operation_name.clone(), elapsed, self.started_at);
        }
        elapsed
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if let Some(sender) = self.result_sender.take() {
            let elapsed = self.started_at.elapsed();
            sender(self.operation_name.clone(), elapsed, self.started_at);
        }
    }
}

/// Entry in the profiling timeline.
#[derive(Debug, Clone)]
pub struct ProfileEntry {
    /// Name of the operation.
    pub operation: String,
    /// Duration of the operation.
    pub duration: Duration,
    /// When the operation started.
    pub started_at: Instant,
    /// Frame number (if applicable).
    pub frame_index: Option<u64>,
    /// Custom tag for grouping.
    pub tag: Option<String>,
}

impl ProfileEntry {
    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }
}

/// A profiler that records per-operation timing for building overlays
/// and performance reports.
#[derive(Debug)]
pub struct AccelProfiler {
    /// Timeline of recorded profile entries (bounded ring buffer).
    entries: Vec<ProfileEntry>,
    /// Maximum entries to retain.
    max_entries: usize,
    /// Current frame index for tagging entries.
    current_frame: u64,
    /// Whether profiling is enabled (can be toggled at runtime).
    enabled: bool,
    /// Per-operation aggregate stats (computed on demand).
    aggregates: HashMap<String, OperationStats>,
}

impl AccelProfiler {
    /// Creates a new profiler with the given max entry count.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: max_entries.max(1),
            current_frame: 0,
            enabled: true,
            aggregates: HashMap::new(),
        }
    }

    /// Creates a profiler with default settings (10000 entries).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(10_000)
    }

    /// Whether the profiler is currently recording.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable profiling.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Advance to the next frame.
    pub fn next_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Current frame index.
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Records a completed operation manually.
    pub fn record(
        &mut self,
        operation: &str,
        duration: Duration,
        started_at: Instant,
        tag: Option<&str>,
    ) {
        if !self.enabled {
            return;
        }

        let entry = ProfileEntry {
            operation: operation.to_string(),
            duration,
            started_at,
            frame_index: Some(self.current_frame),
            tag: tag.map(String::from),
        };

        // Update aggregates
        self.aggregates
            .entry(operation.to_string())
            .or_insert_with(|| OperationStats::new(operation))
            .record(duration, started_at);

        // Ring buffer behavior
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// Returns the number of profile entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns all profile entries.
    #[must_use]
    pub fn entries(&self) -> &[ProfileEntry] {
        &self.entries
    }

    /// Returns entries for a specific frame.
    #[must_use]
    pub fn entries_for_frame(&self, frame: u64) -> Vec<&ProfileEntry> {
        self.entries
            .iter()
            .filter(|e| e.frame_index == Some(frame))
            .collect()
    }

    /// Returns entries for a specific operation.
    #[must_use]
    pub fn entries_for_operation(&self, operation: &str) -> Vec<&ProfileEntry> {
        self.entries
            .iter()
            .filter(|e| e.operation == operation)
            .collect()
    }

    /// Returns aggregate stats for all tracked operations.
    #[must_use]
    pub fn operation_stats(&self) -> &HashMap<String, OperationStats> {
        &self.aggregates
    }

    /// Returns aggregate stats for a specific operation.
    #[must_use]
    pub fn get_operation_stats(&self, operation: &str) -> Option<&OperationStats> {
        self.aggregates.get(operation)
    }

    /// Generates a human-readable timing overlay/report for the most recent frame.
    #[must_use]
    pub fn frame_overlay(&self) -> String {
        let frame = self.current_frame;
        let frame_entries = self.entries_for_frame(frame);

        if frame_entries.is_empty() {
            return format!("Frame {frame}: no operations recorded");
        }

        let total_ms: f64 = frame_entries.iter().map(|e| e.duration_ms()).sum();
        let mut lines = vec![format!(
            "Frame {frame} ({} ops, {total_ms:.2}ms total):",
            frame_entries.len()
        )];

        for entry in &frame_entries {
            let pct = if total_ms > 0.0 {
                entry.duration_ms() / total_ms * 100.0
            } else {
                0.0
            };
            let tag_str = entry
                .tag
                .as_deref()
                .map(|t| format!(" [{t}]"))
                .unwrap_or_default();
            lines.push(format!(
                "  {}{}: {:.3}ms ({:.1}%)",
                entry.operation,
                tag_str,
                entry.duration_ms(),
                pct,
            ));
        }

        lines.join("\n")
    }

    /// Generates a summary report of all operations across all frames.
    #[must_use]
    pub fn summary_report(&self) -> String {
        if self.aggregates.is_empty() {
            return "No operations recorded".to_string();
        }

        let mut lines = vec![format!(
            "Profiler Summary ({} operations, {} entries):",
            self.aggregates.len(),
            self.entries.len()
        )];

        let mut ops: Vec<(&String, &OperationStats)> = self.aggregates.iter().collect();
        ops.sort_by(|a, b| {
            b.1.total_duration
                .partial_cmp(&a.1.total_duration)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (name, stats) in ops {
            let avg = stats.average_ms().unwrap_or(0.0);
            let min = stats.min_ms().unwrap_or(0.0);
            let max = stats.max_ms().unwrap_or(0.0);
            lines.push(format!(
                "  {name}: {count} calls, avg={avg:.3}ms, min={min:.3}ms, max={max:.3}ms",
                count = stats.invocation_count,
            ));
        }

        lines.join("\n")
    }

    /// Returns the percentile value (0..100) for a given operation's recent durations.
    ///
    /// Returns `None` if the operation has no recorded data.
    #[must_use]
    pub fn percentile_ms(&self, operation: &str, percentile: f64) -> Option<f64> {
        let mut durations: Vec<f64> = self
            .entries
            .iter()
            .filter(|e| e.operation == operation)
            .map(ProfileEntry::duration_ms)
            .collect();

        if durations.is_empty() {
            return None;
        }

        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((percentile / 100.0) * (durations.len() as f64 - 1.0))
            .round()
            .max(0.0) as usize;
        let idx = idx.min(durations.len() - 1);
        Some(durations[idx])
    }

    /// Clears all recorded entries and aggregates.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.aggregates.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_sample() {
        let now = Instant::now();
        let sample = TimingSample::new(Duration::from_millis(50), now);
        assert!((sample.duration_ms() - 50.0).abs() < 0.01);
        assert!((sample.duration_us() - 50000.0).abs() < 10.0);
    }

    #[test]
    fn test_operation_stats_empty() {
        let stats = OperationStats::new("test");
        assert_eq!(stats.invocation_count, 0);
        assert!(stats.average_duration().is_none());
        assert!(stats.min_ms().is_none());
        assert!(stats.max_ms().is_none());
        assert!(stats.recent_median_ms().is_none());
    }

    #[test]
    fn test_operation_stats_record() {
        let mut stats = OperationStats::new("scale");
        let now = Instant::now();
        stats.record(Duration::from_millis(10), now);
        stats.record(Duration::from_millis(20), now);
        stats.record(Duration::from_millis(30), now);
        assert_eq!(stats.invocation_count, 3);
        let avg = stats.average_ms().expect("avg should be valid");
        assert!((avg - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_operation_stats_min_max() {
        let mut stats = OperationStats::new("conv");
        let now = Instant::now();
        stats.record(Duration::from_millis(5), now);
        stats.record(Duration::from_millis(15), now);
        stats.record(Duration::from_millis(10), now);
        assert!((stats.min_ms().expect("min_ms should succeed") - 5.0).abs() < 0.01);
        assert!((stats.max_ms().expect("max_ms should succeed") - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_operation_stats_median_odd() {
        let mut stats = OperationStats::new("test");
        let now = Instant::now();
        stats.record(Duration::from_millis(10), now);
        stats.record(Duration::from_millis(30), now);
        stats.record(Duration::from_millis(20), now);
        // Sorted: 10, 20, 30 -> median = 20
        let median = stats.recent_median_ms().expect("median should be valid");
        assert!((median - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_operation_stats_median_even() {
        let mut stats = OperationStats::new("test");
        let now = Instant::now();
        stats.record(Duration::from_millis(10), now);
        stats.record(Duration::from_millis(20), now);
        // Sorted: 10, 20 -> median = 15
        let median = stats.recent_median_ms().expect("median should be valid");
        assert!((median - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_operation_stats_reset() {
        let mut stats = OperationStats::new("op");
        let now = Instant::now();
        stats.record(Duration::from_millis(1), now);
        stats.reset();
        assert_eq!(stats.invocation_count, 0);
        assert!(stats.average_duration().is_none());
    }

    #[test]
    fn test_operation_stats_sample_eviction() {
        let mut stats = OperationStats::with_max_samples("op", 3);
        let now = Instant::now();
        for i in 0..5 {
            stats.record(Duration::from_millis(i * 10), now);
        }
        assert_eq!(stats.recent_count(), 3);
    }

    #[test]
    fn test_transfer_direction_label() {
        assert_eq!(TransferDirection::HostToDevice.label(), "H2D");
        assert_eq!(TransferDirection::DeviceToHost.label(), "D2H");
        assert_eq!(TransferDirection::DeviceToDevice.label(), "D2D");
    }

    #[test]
    fn test_transfer_stats_record() {
        let mut stats = TransferStats::new(TransferDirection::HostToDevice);
        stats.record(1024, Duration::from_millis(1));
        stats.record(2048, Duration::from_millis(2));
        assert_eq!(stats.total_bytes, 3072);
        assert_eq!(stats.transfer_count, 2);
    }

    #[test]
    fn test_transfer_stats_throughput() {
        let mut stats = TransferStats::new(TransferDirection::DeviceToHost);
        stats.record(1_000_000, Duration::from_secs(1));
        let bps = stats.throughput_bps().expect("bps should be valid");
        assert!((bps - 1_000_000.0).abs() < 1.0);
        let mbps = stats.throughput_mbps().expect("mbps should be valid");
        assert!(mbps > 0.0);
    }

    #[test]
    fn test_transfer_stats_empty_throughput() {
        let stats = TransferStats::new(TransferDirection::HostToDevice);
        assert!(stats.throughput_bps().is_none());
        assert!(stats.average_transfer_size().is_none());
    }

    #[test]
    fn test_transfer_stats_reset() {
        let mut stats = TransferStats::new(TransferDirection::HostToDevice);
        stats.record(100, Duration::from_millis(1));
        stats.reset();
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.transfer_count, 0);
    }

    #[test]
    fn test_accel_statistics_record_operations() {
        let mut s = AccelStatistics::new();
        let now = Instant::now();
        s.record_operation("scale", Duration::from_millis(5), now);
        s.record_operation("scale", Duration::from_millis(10), now);
        let op = s.get_operation("scale").expect("op should be valid");
        assert_eq!(op.invocation_count, 2);
    }

    #[test]
    fn test_accel_statistics_tasks() {
        let mut s = AccelStatistics::new();
        s.record_task_submitted();
        s.record_task_submitted();
        s.record_task_completed();
        s.record_task_failed();
        assert_eq!(s.tasks_submitted, 2);
        assert_eq!(s.tasks_completed, 1);
        assert_eq!(s.tasks_failed, 1);
        assert!((s.success_rate().expect("success_rate should succeed") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_accel_statistics_success_rate_none() {
        let s = AccelStatistics::new();
        assert!(s.success_rate().is_none());
    }

    #[test]
    fn test_accel_statistics_reset() {
        let mut s = AccelStatistics::new();
        let now = Instant::now();
        s.record_operation("op", Duration::from_millis(1), now);
        s.record_task_submitted();
        s.reset();
        assert!(s.operation_names().is_empty());
        assert_eq!(s.tasks_submitted, 0);
    }

    #[test]
    fn test_accel_statistics_operation_names() {
        let mut s = AccelStatistics::new();
        let now = Instant::now();
        s.record_operation("alpha", Duration::from_millis(1), now);
        s.record_operation("beta", Duration::from_millis(1), now);
        let names = s.operation_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    // ── AccelProfiler tests ────────────────────────────────────────────────

    #[test]
    fn test_profiler_new() {
        let profiler = AccelProfiler::new(100);
        assert!(profiler.is_enabled());
        assert_eq!(profiler.entry_count(), 0);
        assert_eq!(profiler.current_frame(), 0);
    }

    #[test]
    fn test_profiler_with_defaults() {
        let profiler = AccelProfiler::with_defaults();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_profiler_record() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        profiler.record("scale_image", Duration::from_millis(5), now, None);
        profiler.record("convert_color", Duration::from_millis(3), now, Some("yuv"));
        assert_eq!(profiler.entry_count(), 2);
    }

    #[test]
    fn test_profiler_disabled() {
        let mut profiler = AccelProfiler::new(100);
        profiler.set_enabled(false);
        profiler.record("op", Duration::from_millis(1), Instant::now(), None);
        assert_eq!(profiler.entry_count(), 0);
    }

    #[test]
    fn test_profiler_entries_for_frame() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        profiler.record("op1", Duration::from_millis(1), now, None);
        profiler.next_frame();
        profiler.record("op2", Duration::from_millis(2), now, None);
        profiler.record("op3", Duration::from_millis(3), now, None);

        let frame0 = profiler.entries_for_frame(0);
        assert_eq!(frame0.len(), 1);
        assert_eq!(frame0[0].operation, "op1");

        let frame1 = profiler.entries_for_frame(1);
        assert_eq!(frame1.len(), 2);
    }

    #[test]
    fn test_profiler_entries_for_operation() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        profiler.record("scale", Duration::from_millis(1), now, None);
        profiler.record("color", Duration::from_millis(2), now, None);
        profiler.record("scale", Duration::from_millis(3), now, None);

        let scales = profiler.entries_for_operation("scale");
        assert_eq!(scales.len(), 2);
    }

    #[test]
    fn test_profiler_frame_overlay() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        profiler.record("scale", Duration::from_millis(10), now, None);
        profiler.record("color", Duration::from_millis(5), now, Some("gpu"));

        let overlay = profiler.frame_overlay();
        assert!(overlay.contains("Frame 0"));
        assert!(overlay.contains("scale"));
        assert!(overlay.contains("color"));
        assert!(overlay.contains("[gpu]"));
        assert!(overlay.contains("2 ops"));
    }

    #[test]
    fn test_profiler_frame_overlay_empty() {
        let profiler = AccelProfiler::new(100);
        let overlay = profiler.frame_overlay();
        assert!(overlay.contains("no operations recorded"));
    }

    #[test]
    fn test_profiler_summary_report() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        profiler.record("scale", Duration::from_millis(10), now, None);
        profiler.record("scale", Duration::from_millis(20), now, None);
        profiler.record("color", Duration::from_millis(5), now, None);

        let report = profiler.summary_report();
        assert!(report.contains("Profiler Summary"));
        assert!(report.contains("scale"));
        assert!(report.contains("2 calls"));
        assert!(report.contains("color"));
    }

    #[test]
    fn test_profiler_summary_empty() {
        let profiler = AccelProfiler::new(100);
        let report = profiler.summary_report();
        assert_eq!(report, "No operations recorded");
    }

    #[test]
    fn test_profiler_percentile() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        for i in 1..=10 {
            profiler.record("op", Duration::from_millis(i), now, None);
        }

        let p50 = profiler
            .percentile_ms("op", 50.0)
            .expect("p50 should be valid");
        assert!(p50 >= 4.0 && p50 <= 6.0, "p50 = {p50}");

        let p90 = profiler
            .percentile_ms("op", 90.0)
            .expect("p90 should be valid");
        assert!(p90 >= 8.0 && p90 <= 10.0, "p90 = {p90}");

        assert!(profiler.percentile_ms("nonexistent", 50.0).is_none());
    }

    #[test]
    fn test_profiler_ring_buffer_eviction() {
        let mut profiler = AccelProfiler::new(5);
        let now = Instant::now();
        for i in 0..10 {
            profiler.record(&format!("op{}", i), Duration::from_millis(1), now, None);
        }
        assert_eq!(profiler.entry_count(), 5);
        // Oldest should be evicted
        assert_eq!(profiler.entries()[0].operation, "op5");
    }

    #[test]
    fn test_profiler_clear() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        profiler.record("op", Duration::from_millis(1), now, None);
        assert_eq!(profiler.entry_count(), 1);
        profiler.clear();
        assert_eq!(profiler.entry_count(), 0);
        assert!(profiler.operation_stats().is_empty());
    }

    #[test]
    fn test_profiler_operation_stats() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        profiler.record("scale", Duration::from_millis(10), now, None);
        profiler.record("scale", Duration::from_millis(20), now, None);

        let stats = profiler
            .get_operation_stats("scale")
            .expect("stats should be valid");
        assert_eq!(stats.invocation_count, 2);
        let avg = stats.average_ms().expect("avg should be valid");
        assert!((avg - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_profiler_next_frame() {
        let mut profiler = AccelProfiler::new(100);
        assert_eq!(profiler.current_frame(), 0);
        profiler.next_frame();
        assert_eq!(profiler.current_frame(), 1);
        profiler.next_frame();
        assert_eq!(profiler.current_frame(), 2);
    }

    #[test]
    fn test_profile_entry_duration_ms() {
        let entry = ProfileEntry {
            operation: "test".to_string(),
            duration: Duration::from_millis(42),
            started_at: Instant::now(),
            frame_index: Some(0),
            tag: None,
        };
        assert!((entry.duration_ms() - 42.0).abs() < 0.1);
    }

    #[test]
    fn test_profiler_tagged_entries() {
        let mut profiler = AccelProfiler::new(100);
        let now = Instant::now();
        profiler.record("op", Duration::from_millis(1), now, Some("gpu"));
        profiler.record("op", Duration::from_millis(2), now, Some("cpu"));
        profiler.record("op", Duration::from_millis(3), now, None);

        let entries = profiler.entries();
        assert_eq!(entries[0].tag.as_deref(), Some("gpu"));
        assert_eq!(entries[1].tag.as_deref(), Some("cpu"));
        assert!(entries[2].tag.is_none());
    }
}
