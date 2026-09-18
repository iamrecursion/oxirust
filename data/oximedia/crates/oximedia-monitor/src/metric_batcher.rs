//! Metric batching to reduce storage write frequency under high load.
//!
//! Instead of writing every metric sample immediately to the storage backend,
//! the `MetricBatcher` accumulates samples into batches that are flushed
//! either when the batch reaches a configurable size or when a time deadline
//! is hit.  This dramatically reduces I/O pressure on SQLite (or any other
//! store) during traffic spikes.
//!
//! # Example
//!
//! ```
//! use oximedia_monitor::metric_batcher::{MetricBatcher, BatcherConfig};
//!
//! let mut batcher = MetricBatcher::new(BatcherConfig::default());
//! let flushed = batcher.add("cpu_usage", 85.0);
//! // `flushed` is empty until the batch is full or deadline is reached.
//! assert!(flushed.is_empty());
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the metric batcher.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum number of entries per batch before an automatic flush.
    pub max_batch_size: usize,
    /// Maximum time a batch may be held before forced flush.
    pub max_batch_age: Duration,
    /// Maximum number of completed batches held in the output queue.
    pub max_pending_batches: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 500,
            max_batch_age: Duration::from_secs(5),
            max_pending_batches: 100,
        }
    }
}

impl BatcherConfig {
    /// Set the maximum batch size.
    #[must_use]
    pub fn with_max_batch_size(mut self, n: usize) -> Self {
        self.max_batch_size = n.max(1);
        self
    }

    /// Set the maximum batch age before forced flush.
    #[must_use]
    pub fn with_max_batch_age(mut self, d: Duration) -> Self {
        self.max_batch_age = d;
        self
    }

    /// Set the maximum pending batches.
    #[must_use]
    pub fn with_max_pending_batches(mut self, n: usize) -> Self {
        self.max_pending_batches = n.max(1);
        self
    }
}

// ---------------------------------------------------------------------------
// Batch entry
// ---------------------------------------------------------------------------

/// A single metric sample in a batch.
#[derive(Debug, Clone)]
pub struct BatchEntry {
    /// Metric name.
    pub metric: String,
    /// Metric value.
    pub value: f64,
    /// When the sample was received.
    pub received_at: Instant,
}

/// A completed batch of metric entries ready for storage.
#[derive(Debug, Clone)]
pub struct MetricBatch {
    /// All entries in this batch.
    pub entries: Vec<BatchEntry>,
    /// When this batch was created.
    pub created_at: Instant,
    /// When this batch was flushed.
    pub flushed_at: Instant,
}

impl MetricBatch {
    /// Number of entries in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Age of this batch since creation.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.flushed_at.duration_since(self.created_at)
    }
}

// ---------------------------------------------------------------------------
// Batcher statistics
// ---------------------------------------------------------------------------

/// Statistics for the metric batcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatcherStats {
    /// Total samples received.
    pub total_samples: u64,
    /// Total batches flushed.
    pub total_flushes: u64,
    /// Flushes triggered by reaching max batch size.
    pub size_flushes: u64,
    /// Flushes triggered by reaching max batch age.
    pub age_flushes: u64,
    /// Flushes triggered manually.
    pub manual_flushes: u64,
    /// Total batches dropped due to pending queue overflow.
    pub dropped_batches: u64,
}

// ---------------------------------------------------------------------------
// MetricBatcher
// ---------------------------------------------------------------------------

/// Batches incoming metric samples and flushes them at configurable intervals.
#[derive(Debug)]
pub struct MetricBatcher {
    config: BatcherConfig,
    current_batch: Vec<BatchEntry>,
    batch_start: Instant,
    pending: VecDeque<MetricBatch>,
    stats: BatcherStats,
}

impl MetricBatcher {
    /// Create a new metric batcher.
    #[must_use]
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            current_batch: Vec::new(),
            batch_start: Instant::now(),
            pending: VecDeque::new(),
            stats: BatcherStats::default(),
        }
    }

    /// Create a batcher with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(BatcherConfig::default())
    }

    /// Add a metric sample. Returns the flushed entries if a flush was triggered.
    ///
    /// A flush is triggered when:
    /// 1. The current batch reaches `max_batch_size`, or
    /// 2. The current batch has been open longer than `max_batch_age`.
    pub fn add(&mut self, metric: &str, value: f64) -> Vec<BatchEntry> {
        self.stats.total_samples += 1;

        self.current_batch.push(BatchEntry {
            metric: metric.to_string(),
            value,
            received_at: Instant::now(),
        });

        // Check if we should flush.
        if self.current_batch.len() >= self.config.max_batch_size {
            self.stats.size_flushes += 1;
            return self.flush_current();
        }

        if self.batch_start.elapsed() >= self.config.max_batch_age {
            self.stats.age_flushes += 1;
            return self.flush_current();
        }

        Vec::new()
    }

    /// Force-flush the current batch regardless of size or age.
    ///
    /// Returns the flushed entries, or an empty vec if nothing was pending.
    pub fn flush(&mut self) -> Vec<BatchEntry> {
        if self.current_batch.is_empty() {
            return Vec::new();
        }
        self.stats.manual_flushes += 1;
        self.flush_current()
    }

    /// Internal flush that moves the current batch into the pending queue
    /// and returns the entries.
    fn flush_current(&mut self) -> Vec<BatchEntry> {
        let now = Instant::now();
        let entries = std::mem::take(&mut self.current_batch);

        let batch = MetricBatch {
            entries: entries.clone(),
            created_at: self.batch_start,
            flushed_at: now,
        };

        self.stats.total_flushes += 1;

        // Enforce pending queue limit.
        if self.pending.len() >= self.config.max_pending_batches {
            self.pending.pop_front();
            self.stats.dropped_batches += 1;
        }
        self.pending.push_back(batch);

        // Start a new batch.
        self.batch_start = Instant::now();

        entries
    }

    /// Check if the current batch should be flushed due to age and flush if so.
    ///
    /// Call this periodically (e.g., from a timer) to ensure batches don't
    /// languish indefinitely during low-traffic periods.
    pub fn tick(&mut self) -> Vec<BatchEntry> {
        if !self.current_batch.is_empty() && self.batch_start.elapsed() >= self.config.max_batch_age
        {
            self.stats.age_flushes += 1;
            return self.flush_current();
        }
        Vec::new()
    }

    /// Drain completed batches from the pending queue.
    pub fn drain_pending(&mut self) -> Vec<MetricBatch> {
        self.pending.drain(..).collect()
    }

    /// Number of entries in the current (unflushed) batch.
    #[must_use]
    pub fn current_batch_size(&self) -> usize {
        self.current_batch.len()
    }

    /// Number of completed batches waiting in the pending queue.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get a snapshot of the batcher statistics.
    #[must_use]
    pub fn stats(&self) -> BatcherStats {
        self.stats
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &BatcherConfig {
        &self.config
    }

    /// Clear all state (current batch and pending queue).
    pub fn clear(&mut self) {
        self.current_batch.clear();
        self.pending.clear();
        self.batch_start = Instant::now();
    }
}

impl Default for MetricBatcher {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BatcherConfig --

    #[test]
    fn test_config_default() {
        let cfg = BatcherConfig::default();
        assert_eq!(cfg.max_batch_size, 500);
        assert_eq!(cfg.max_batch_age, Duration::from_secs(5));
        assert_eq!(cfg.max_pending_batches, 100);
    }

    #[test]
    fn test_config_builders() {
        let cfg = BatcherConfig::default()
            .with_max_batch_size(100)
            .with_max_batch_age(Duration::from_secs(10))
            .with_max_pending_batches(50);
        assert_eq!(cfg.max_batch_size, 100);
        assert_eq!(cfg.max_batch_age, Duration::from_secs(10));
        assert_eq!(cfg.max_pending_batches, 50);
    }

    #[test]
    fn test_config_min_batch_size() {
        let cfg = BatcherConfig::default().with_max_batch_size(0);
        assert_eq!(cfg.max_batch_size, 1);
    }

    // -- MetricBatcher basic --

    #[test]
    fn test_batcher_new() {
        let b = MetricBatcher::with_defaults();
        assert_eq!(b.current_batch_size(), 0);
        assert_eq!(b.pending_count(), 0);
        assert_eq!(b.stats().total_samples, 0);
    }

    #[test]
    fn test_add_below_limit_no_flush() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(10));
        let flushed = b.add("cpu", 50.0);
        assert!(flushed.is_empty());
        assert_eq!(b.current_batch_size(), 1);
        assert_eq!(b.stats().total_samples, 1);
    }

    #[test]
    fn test_add_triggers_size_flush() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(3));
        let _ = b.add("cpu", 1.0);
        let _ = b.add("cpu", 2.0);
        let flushed = b.add("cpu", 3.0); // 3rd entry triggers flush.
        assert_eq!(flushed.len(), 3);
        assert_eq!(b.current_batch_size(), 0);
        assert_eq!(b.pending_count(), 1);
        assert_eq!(b.stats().size_flushes, 1);
    }

    #[test]
    fn test_manual_flush() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(100));
        let _ = b.add("cpu", 1.0);
        let _ = b.add("mem", 2.0);
        let flushed = b.flush();
        assert_eq!(flushed.len(), 2);
        assert_eq!(b.current_batch_size(), 0);
        assert_eq!(b.stats().manual_flushes, 1);
    }

    #[test]
    fn test_manual_flush_empty_noop() {
        let mut b = MetricBatcher::with_defaults();
        let flushed = b.flush();
        assert!(flushed.is_empty());
        assert_eq!(b.stats().manual_flushes, 0);
    }

    #[test]
    fn test_tick_flushes_aged_batch() {
        let mut b = MetricBatcher::new(
            BatcherConfig::default()
                .with_max_batch_size(1000)
                .with_max_batch_age(Duration::from_millis(50)),
        );
        // Add an entry (batch_start is "now", so add won't trigger age flush).
        let added = b.add("cpu", 1.0);
        assert!(added.is_empty(), "add should not flush yet");
        assert_eq!(b.current_batch_size(), 1);

        // Now move batch_start into the past so tick sees it as aged.
        b.batch_start = Instant::now() - Duration::from_millis(100);
        let flushed = b.tick();
        assert_eq!(flushed.len(), 1);
        assert_eq!(b.stats().age_flushes, 1);
    }

    #[test]
    fn test_tick_no_flush_when_empty() {
        let mut b = MetricBatcher::new(
            BatcherConfig::default().with_max_batch_age(Duration::from_millis(0)),
        );
        let flushed = b.tick();
        assert!(flushed.is_empty());
    }

    // -- Pending queue --

    #[test]
    fn test_drain_pending() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(2));
        let _ = b.add("a", 1.0);
        let _ = b.add("b", 2.0); // triggers flush -> 1 pending batch
        let _ = b.add("c", 3.0);
        let _ = b.add("d", 4.0); // triggers flush -> 2 pending batches

        let batches = b.drain_pending();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
        assert_eq!(b.pending_count(), 0);
    }

    #[test]
    fn test_pending_queue_overflow_drops_oldest() {
        let mut b = MetricBatcher::new(
            BatcherConfig::default()
                .with_max_batch_size(1)
                .with_max_pending_batches(2),
        );
        let _ = b.add("a", 1.0); // flush -> pending[0]
        let _ = b.add("b", 2.0); // flush -> pending[1]
        let _ = b.add("c", 3.0); // flush -> pending drops [0], pushes [2]

        assert_eq!(b.pending_count(), 2);
        assert_eq!(b.stats().dropped_batches, 1);

        let batches = b.drain_pending();
        // The first batch ("a") was dropped; remaining are "b" and "c".
        assert_eq!(batches[0].entries[0].metric, "b");
        assert_eq!(batches[1].entries[0].metric, "c");
    }

    // -- MetricBatch --

    #[test]
    fn test_batch_len_and_empty() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(2));
        let _ = b.add("cpu", 1.0);
        let _ = b.add("cpu", 2.0);
        let batches = b.drain_pending();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);
        assert!(!batches[0].is_empty());
    }

    // -- Statistics --

    #[test]
    fn test_stats_tracking() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(5));
        for i in 0..12 {
            let _ = b.add("m", i as f64);
        }
        let _ = b.flush(); // manual flush of remaining 2 entries

        let stats = b.stats();
        assert_eq!(stats.total_samples, 12);
        assert_eq!(stats.size_flushes, 2); // 5+5 = 10 entries flushed by size
        assert_eq!(stats.manual_flushes, 1); // remaining 2
        assert_eq!(stats.total_flushes, 3);
    }

    // -- Clear --

    #[test]
    fn test_clear() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(100));
        let _ = b.add("cpu", 1.0);
        let _ = b.add("mem", 2.0);
        b.clear();
        assert_eq!(b.current_batch_size(), 0);
        assert_eq!(b.pending_count(), 0);
    }

    // -- Multiple metrics --

    #[test]
    fn test_mixed_metrics_batch() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(4));
        let _ = b.add("cpu", 50.0);
        let _ = b.add("memory", 70.0);
        let _ = b.add("disk", 30.0);
        let flushed = b.add("network", 100.0);
        assert_eq!(flushed.len(), 4);
        assert_eq!(flushed[0].metric, "cpu");
        assert_eq!(flushed[1].metric, "memory");
        assert_eq!(flushed[2].metric, "disk");
        assert_eq!(flushed[3].metric, "network");
    }

    // -- Continuous operation --

    #[test]
    fn test_multiple_flush_cycles() {
        let mut b = MetricBatcher::new(BatcherConfig::default().with_max_batch_size(2));
        for i in 0..10 {
            let _ = b.add("m", i as f64);
        }
        assert_eq!(b.pending_count(), 5); // 10 / 2 = 5 batches
        assert_eq!(b.current_batch_size(), 0); // all flushed exactly
    }
}
