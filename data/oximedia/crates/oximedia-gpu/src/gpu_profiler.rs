//! GPU profiling and timing utilities.
//!
//! Provides timestamp-based GPU profiling with named scopes for measuring
//! execution time of GPU operations, pipelines, and individual passes.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single GPU timestamp sample.
///
/// Records the start and optional end time of a GPU operation.
#[derive(Debug, Clone)]
pub struct GpuTimestamp {
    /// Human-readable label for this timestamp.
    pub label: String,
    /// Wall-clock time when the operation began.
    pub start: Instant,
    /// Wall-clock time when the operation ended, if completed.
    pub end: Option<Instant>,
}

impl GpuTimestamp {
    /// Create a new timestamp starting now.
    #[must_use]
    pub fn begin(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
            end: None,
        }
    }

    /// Mark this timestamp as finished.
    pub fn finish(&mut self) {
        self.end = Some(Instant::now());
    }

    /// Return the elapsed duration, or `None` if not yet finished.
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        self.end.map(|e| e.duration_since(self.start))
    }

    /// Return elapsed microseconds as `f64`, or `None` if not yet finished.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn elapsed_us(&self) -> Option<f64> {
        self.elapsed().map(|d| d.as_nanos() as f64 / 1_000.0)
    }
}

/// An RAII guard that automatically finishes a [`GpuTimestamp`] on drop.
///
/// Created via [`GpuProfiler::scope`].
pub struct GpuProfilerScope<'a> {
    profiler: &'a mut GpuProfiler,
    key: String,
}

impl<'a> GpuProfilerScope<'a> {
    fn new(profiler: &'a mut GpuProfiler, key: String) -> Self {
        Self { profiler, key }
    }
}

impl Drop for GpuProfilerScope<'_> {
    fn drop(&mut self) {
        self.profiler.end_scope(&self.key);
    }
}

/// Aggregate statistics for a named GPU scope.
#[derive(Debug, Clone, Default)]
pub struct ScopeStats {
    /// Total number of samples collected.
    pub count: u64,
    /// Accumulated duration of all samples.
    pub total: Duration,
    /// Minimum single-sample duration.
    pub min: Option<Duration>,
    /// Maximum single-sample duration.
    pub max: Option<Duration>,
}

impl ScopeStats {
    /// Record a new sample duration.
    pub fn record(&mut self, d: Duration) {
        self.count += 1;
        self.total += d;
        self.min = Some(self.min.map_or(d, |m| m.min(d)));
        self.max = Some(self.max.map_or(d, |m| m.max(d)));
    }

    /// Compute the mean duration across all samples.
    #[must_use]
    pub fn mean(&self) -> Option<Duration> {
        if self.count == 0 {
            None
        } else {
            Some(self.total / self.count as u32)
        }
    }
}

/// Central GPU profiler that owns all active and completed timestamps.
///
/// # Example
///
/// ```
/// use oximedia_gpu::gpu_profiler::GpuProfiler;
///
/// let mut profiler = GpuProfiler::new();
/// profiler.begin("tonemap");
/// // ... GPU work ...
/// profiler.end("tonemap");
/// let summary = profiler.summary();
/// assert!(summary.contains_key("tonemap"));
/// ```
#[derive(Debug, Default)]
pub struct GpuProfiler {
    /// Currently active (open) timestamps keyed by scope label.
    active: HashMap<String, GpuTimestamp>,
    /// Accumulated statistics per scope label.
    stats: HashMap<String, ScopeStats>,
}

impl GpuProfiler {
    /// Create a new, empty profiler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new profiling scope with the given label.
    ///
    /// If a scope with this label is already active it is overwritten.
    pub fn begin(&mut self, label: impl Into<String>) {
        let label = label.into();
        self.active
            .insert(label.clone(), GpuTimestamp::begin(label));
    }

    /// End an active scope identified by `label`.
    ///
    /// Records the elapsed time into cumulative stats. Does nothing if the
    /// label is not currently active.
    pub fn end(&mut self, label: &str) {
        if let Some(mut ts) = self.active.remove(label) {
            ts.finish();
            if let Some(d) = ts.elapsed() {
                self.stats.entry(label.to_owned()).or_default().record(d);
            }
        }
    }

    /// Internal helper used by `GpuProfilerScope::drop`.
    fn end_scope(&mut self, key: &str) {
        self.end(key);
    }

    /// Open a scope and return an RAII guard that calls `end` on drop.
    pub fn scope(&mut self, label: impl Into<String>) -> GpuProfilerScope<'_> {
        let key = label.into();
        self.begin(key.clone());
        GpuProfilerScope::new(self, key)
    }

    /// Return a snapshot of the accumulated statistics for every scope.
    #[must_use]
    pub fn summary(&self) -> &HashMap<String, ScopeStats> {
        &self.stats
    }

    /// Reset all statistics and discard any active scopes.
    pub fn reset(&mut self) {
        self.active.clear();
        self.stats.clear();
    }

    /// Return the number of distinct scope labels that have been recorded.
    #[must_use]
    pub fn scope_count(&self) -> usize {
        self.stats.len()
    }

    /// Return `true` if there are no recorded statistics.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn timestamp_begin_not_finished() {
        let ts = GpuTimestamp::begin("test");
        assert_eq!(ts.label, "test");
        assert!(ts.end.is_none());
        assert!(ts.elapsed().is_none());
    }

    #[test]
    fn timestamp_finish_elapsed() {
        let mut ts = GpuTimestamp::begin("op");
        thread::sleep(Duration::from_millis(1));
        ts.finish();
        let e = ts.elapsed().expect("should have elapsed");
        assert!(e >= Duration::from_millis(1));
    }

    #[test]
    fn timestamp_elapsed_us_some() {
        let mut ts = GpuTimestamp::begin("op");
        ts.finish();
        assert!(ts.elapsed_us().is_some());
    }

    #[test]
    fn timestamp_elapsed_us_none_when_unfinished() {
        let ts = GpuTimestamp::begin("op");
        assert!(ts.elapsed_us().is_none());
    }

    #[test]
    fn scope_stats_empty_mean_none() {
        let s = ScopeStats::default();
        assert!(s.mean().is_none());
    }

    #[test]
    fn scope_stats_records_single() {
        let mut s = ScopeStats::default();
        s.record(Duration::from_millis(10));
        assert_eq!(s.count, 1);
        assert_eq!(s.mean(), Some(Duration::from_millis(10)));
    }

    #[test]
    fn scope_stats_min_max() {
        let mut s = ScopeStats::default();
        s.record(Duration::from_millis(5));
        s.record(Duration::from_millis(15));
        assert_eq!(s.min, Some(Duration::from_millis(5)));
        assert_eq!(s.max, Some(Duration::from_millis(15)));
    }

    #[test]
    fn profiler_begin_end_records_stats() {
        let mut p = GpuProfiler::new();
        p.begin("pass");
        p.end("pass");
        assert!(p.summary().contains_key("pass"));
        assert_eq!(p.summary()["pass"].count, 1);
    }

    #[test]
    fn profiler_end_unknown_label_no_panic() {
        let mut p = GpuProfiler::new();
        p.end("nonexistent"); // should not panic
    }

    #[test]
    fn profiler_scope_raii() {
        let mut p = GpuProfiler::new();
        {
            let _scope = p.scope("render");
        }
        assert!(p.summary().contains_key("render"));
    }

    #[test]
    fn profiler_reset_clears_all() {
        let mut p = GpuProfiler::new();
        p.begin("x");
        p.end("x");
        p.reset();
        assert!(p.is_empty());
        assert_eq!(p.scope_count(), 0);
    }

    #[test]
    fn profiler_scope_count() {
        let mut p = GpuProfiler::new();
        p.begin("a");
        p.end("a");
        p.begin("b");
        p.end("b");
        assert_eq!(p.scope_count(), 2);
    }

    #[test]
    fn profiler_multiple_samples_accumulate() {
        let mut p = GpuProfiler::new();
        for _ in 0..3 {
            p.begin("pass");
            p.end("pass");
        }
        assert_eq!(p.summary()["pass"].count, 3);
    }

    #[test]
    fn profiler_is_empty_initially() {
        let p = GpuProfiler::new();
        assert!(p.is_empty());
    }

    #[test]
    fn profiler_default_equals_new() {
        let p: GpuProfiler = GpuProfiler::default();
        assert!(p.is_empty());
    }
}
