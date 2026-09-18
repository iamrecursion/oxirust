//! Automated bottleneck classification with CPU/IO/Memory/Lock heuristics.
//!
//! This module provides `BottleneckClassifier` (analysis-oriented, operating on
//! `ProfilingMetrics` instead of the name-based `Bottleneck` from
//! `bottleneck::detect`) together with `BottleneckReport` (primary + secondary
//! type and textual suggestions).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// BottleneckType
// ---------------------------------------------------------------------------

/// High-level classification of the dominant performance bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BottleneckType {
    /// The workload is limited by CPU computation time.
    CpuBound,
    /// The workload is limited by I/O latency or throughput.
    IoBound,
    /// The workload is limited by memory bandwidth or allocation rate.
    MemoryBound,
    /// The workload is dominated by lock contention / thread synchronisation.
    LockContention,
    /// Unable to determine the dominant bottleneck.
    Unknown,
}

impl BottleneckType {
    /// Returns a human-readable one-liner for this type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::CpuBound => "CPU-bound",
            Self::IoBound => "I/O-bound",
            Self::MemoryBound => "Memory-bound",
            Self::LockContention => "Lock contention",
            Self::Unknown => "Unknown",
        }
    }

    /// Returns an initial set of optimisation suggestions for this type.
    #[must_use]
    pub fn base_suggestions(self) -> &'static [&'static str] {
        match self {
            Self::CpuBound => &[
                "Profile hot loops and consider SIMD / vectorisation",
                "Parallelise independent computation across threads",
                "Reduce unnecessary branching and data-dependent control flow",
                "Move expensive constants out of hot loops",
            ],
            Self::IoBound => &[
                "Switch to asynchronous or non-blocking I/O",
                "Batch small reads/writes into larger operations",
                "Add an in-memory cache for frequently accessed data",
                "Prefetch data on a background thread",
            ],
            Self::MemoryBound => &[
                "Reduce heap allocation rate; use pooled or stack-allocated buffers",
                "Improve data locality (AoS → SoA, cache-friendly traversal order)",
                "Minimise large copying; prefer slice references or Arc",
                "Profile allocator hot-paths with a custom allocator",
            ],
            Self::LockContention => &[
                "Reduce critical-section size; keep locks as narrow as possible",
                "Replace coarse mutexes with fine-grained or per-entry locking",
                "Consider lock-free data structures (atomic operations, crossbeam)",
                "Use message-passing / channels instead of shared mutable state",
            ],
            Self::Unknown => &[
                "Run a sampling profiler to identify the dominant hot-path",
                "Collect CPU, memory and I/O counters simultaneously",
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// ProfilingMetrics
// ---------------------------------------------------------------------------

/// Aggregated profiling metrics used as input to the classifier.
///
/// All fields are optional; unset fields are excluded from classification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfilingMetrics {
    /// Total wall-clock duration of the measurement window (milliseconds).
    pub wall_time_ms: Option<f64>,
    /// Total CPU time consumed across all threads (milliseconds).
    pub cpu_time_ms: Option<f64>,
    /// Total time blocked on I/O (milliseconds).
    pub io_wait_ms: Option<f64>,
    /// Total time blocked on mutex/lock primitives (milliseconds).
    pub lock_wait_ms: Option<f64>,
    /// Heap bytes allocated during the window.
    pub allocated_bytes: Option<u64>,
    /// Number of heap allocation events during the window.
    pub allocation_count: Option<u64>,
    /// Peak resident-set size (bytes).
    pub peak_rss_bytes: Option<u64>,
    /// Total bytes read and written to storage.
    pub io_bytes: Option<u64>,
    /// L1/L2/L3 cache-miss count (if available via hardware counters).
    pub cache_miss_count: Option<u64>,
    /// Number of thread-context switches during the window.
    pub context_switches: Option<u64>,
    /// CPU utilisation as a fraction (0.0–1.0).
    pub cpu_utilisation: Option<f64>,
}

impl ProfilingMetrics {
    /// Creates an empty `ProfilingMetrics`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the CPU fraction of total wall time.
    ///
    /// `None` if wall_time or cpu_time is unknown.
    #[must_use]
    pub fn cpu_fraction(&self) -> Option<f64> {
        match (self.wall_time_ms, self.cpu_time_ms) {
            (Some(wall), Some(cpu)) if wall > 0.0 => Some(cpu / wall),
            _ => self.cpu_utilisation,
        }
    }

    /// Returns the I/O wait fraction of total wall time.
    #[must_use]
    pub fn io_fraction(&self) -> Option<f64> {
        match (self.wall_time_ms, self.io_wait_ms) {
            (Some(wall), Some(io)) if wall > 0.0 => Some(io / wall),
            _ => None,
        }
    }

    /// Returns the lock wait fraction of total wall time.
    #[must_use]
    pub fn lock_fraction(&self) -> Option<f64> {
        match (self.wall_time_ms, self.lock_wait_ms) {
            (Some(wall), Some(lock)) if wall > 0.0 => Some(lock / wall),
            _ => None,
        }
    }

    /// Returns allocations per millisecond of wall time.
    #[must_use]
    pub fn alloc_rate(&self) -> Option<f64> {
        match (self.wall_time_ms, self.allocation_count) {
            (Some(wall), Some(cnt)) if wall > 0.0 => Some(cnt as f64 / wall),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Classification thresholds
// ---------------------------------------------------------------------------

/// Threshold parameters for `BottleneckClassifier`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierThresholds {
    /// CPU fraction above which the workload is flagged as CPU-bound (default 0.70).
    pub cpu_bound_fraction: f64,
    /// I/O fraction above which the workload is flagged as I/O-bound (default 0.40).
    pub io_bound_fraction: f64,
    /// Lock fraction above which the workload is flagged as lock-contention (default 0.25).
    pub lock_contention_fraction: f64,
    /// Allocation rate (allocs/ms) above which memory-bound is flagged (default 1000.0).
    pub memory_alloc_rate_threshold: f64,
    /// CPU utilisation fraction above which CPU-bound is inferred when no cpu_time_ms
    /// is available (default 0.75).
    pub cpu_util_threshold: f64,
}

impl Default for ClassifierThresholds {
    fn default() -> Self {
        Self {
            cpu_bound_fraction: 0.70,
            io_bound_fraction: 0.40,
            lock_contention_fraction: 0.25,
            memory_alloc_rate_threshold: 1_000.0,
            cpu_util_threshold: 0.75,
        }
    }
}

// ---------------------------------------------------------------------------
// BottleneckReport
// ---------------------------------------------------------------------------

/// Result produced by `BottleneckClassifier::classify`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisBottleneckReport {
    /// The primary (dominant) bottleneck type.
    pub primary: BottleneckType,
    /// A secondary bottleneck, if a second class exceeds its threshold.
    pub secondary: Option<BottleneckType>,
    /// Ordered list of actionable optimisation suggestions.
    pub suggestions: Vec<String>,
    /// Brief one-line summary.
    pub summary: String,
    /// Scores (0.0–1.0) for each candidate type.
    pub scores: std::collections::HashMap<String, f64>,
}

// ---------------------------------------------------------------------------
// BottleneckClassifier
// ---------------------------------------------------------------------------

/// Analyses `ProfilingMetrics` and classifies the workload's dominant
/// bottleneck using configurable heuristics.
#[derive(Debug, Clone)]
pub struct BottleneckClassifier {
    thresholds: ClassifierThresholds,
}

impl BottleneckClassifier {
    /// Creates a classifier with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            thresholds: ClassifierThresholds::default(),
        }
    }

    /// Creates a classifier with custom thresholds.
    #[must_use]
    pub fn with_thresholds(thresholds: ClassifierThresholds) -> Self {
        Self { thresholds }
    }

    /// Classifies the bottleneck type from the given metrics.
    #[must_use]
    pub fn classify(&self, metrics: &ProfilingMetrics) -> AnalysisBottleneckReport {
        let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

        // --- Lock contention score ---
        if let Some(lock_frac) = metrics.lock_fraction() {
            let score = (lock_frac / self.thresholds.lock_contention_fraction).min(1.0);
            scores.insert("LockContention".to_owned(), score);
        }

        // --- I/O bound score ---
        if let Some(io_frac) = metrics.io_fraction() {
            let score = (io_frac / self.thresholds.io_bound_fraction).min(1.0);
            scores.insert("IoBound".to_owned(), score);
        }

        // --- Memory bound score (allocation rate) ---
        if let Some(rate) = metrics.alloc_rate() {
            let score = (rate / self.thresholds.memory_alloc_rate_threshold).min(1.0);
            scores.insert("MemoryBound".to_owned(), score);
        }
        // Also consider cache miss count as a memory signal
        if let (Some(cache_miss), Some(wall)) = (metrics.cache_miss_count, metrics.wall_time_ms) {
            if wall > 0.0 {
                // Normalise: >100 k misses/ms → full memory score
                let rate = cache_miss as f64 / wall;
                let score = (rate / 100_000.0).min(1.0);
                let existing = scores.get("MemoryBound").copied().unwrap_or(0.0);
                scores.insert("MemoryBound".to_owned(), existing.max(score));
            }
        }

        // --- CPU bound score ---
        if let Some(cpu_frac) = metrics.cpu_fraction() {
            let score = (cpu_frac / self.thresholds.cpu_bound_fraction).min(1.0);
            scores.insert("CpuBound".to_owned(), score);
        }

        // --- Determine primary and secondary ---
        let ranked = Self::rank_scores(&scores);

        let primary = ranked
            .first()
            .map(|(t, _)| *t)
            .unwrap_or(BottleneckType::Unknown);

        let secondary = ranked.get(1).and_then(|(t, score)| {
            // Only flag secondary if its score is at least 50 % of primary's
            if let Some((_, primary_score)) = ranked.first() {
                if *score >= primary_score * 0.5 && *score >= 0.3 {
                    return Some(*t);
                }
            }
            None
        });

        // --- Collect suggestions ---
        let mut suggestions: Vec<String> = primary
            .base_suggestions()
            .iter()
            .map(|s| s.to_string())
            .collect();
        if let Some(sec) = secondary {
            for s in sec.base_suggestions().iter().take(2) {
                suggestions.push(format!("[secondary] {}", s));
            }
        }

        let summary = format!(
            "Primary bottleneck: {}{}",
            primary.label(),
            secondary
                .map(|s| format!("; secondary: {}", s.label()))
                .unwrap_or_default(),
        );

        AnalysisBottleneckReport {
            primary,
            secondary,
            suggestions,
            summary,
            scores,
        }
    }

    /// Sorts the scores map, returning `(BottleneckType, score)` pairs in
    /// descending score order, filtering out types with score < 0.2.
    fn rank_scores(scores: &std::collections::HashMap<String, f64>) -> Vec<(BottleneckType, f64)> {
        let mut pairs: Vec<(BottleneckType, f64)> = scores
            .iter()
            .filter_map(|(k, &v)| {
                if v < 0.2 {
                    return None;
                }
                let t = match k.as_str() {
                    "CpuBound" => BottleneckType::CpuBound,
                    "IoBound" => BottleneckType::IoBound,
                    "MemoryBound" => BottleneckType::MemoryBound,
                    "LockContention" => BottleneckType::LockContention,
                    _ => BottleneckType::Unknown,
                };
                Some((t, v))
            })
            .collect();

        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }
}

impl Default for BottleneckClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_metrics(cpu_frac: f64, wall_ms: f64) -> ProfilingMetrics {
        ProfilingMetrics {
            wall_time_ms: Some(wall_ms),
            cpu_time_ms: Some(wall_ms * cpu_frac),
            ..Default::default()
        }
    }

    fn io_metrics(io_frac: f64, wall_ms: f64) -> ProfilingMetrics {
        ProfilingMetrics {
            wall_time_ms: Some(wall_ms),
            io_wait_ms: Some(wall_ms * io_frac),
            cpu_time_ms: Some(wall_ms * 0.1),
            ..Default::default()
        }
    }

    fn lock_metrics(lock_frac: f64, wall_ms: f64) -> ProfilingMetrics {
        ProfilingMetrics {
            wall_time_ms: Some(wall_ms),
            lock_wait_ms: Some(wall_ms * lock_frac),
            cpu_time_ms: Some(wall_ms * 0.1),
            ..Default::default()
        }
    }

    fn memory_metrics(alloc_rate: f64, wall_ms: f64) -> ProfilingMetrics {
        ProfilingMetrics {
            wall_time_ms: Some(wall_ms),
            allocation_count: Some((alloc_rate * wall_ms) as u64),
            cpu_time_ms: Some(wall_ms * 0.2),
            ..Default::default()
        }
    }

    #[test]
    fn test_cpu_bound_classification() {
        let metrics = cpu_metrics(0.90, 1000.0);
        let report = BottleneckClassifier::new().classify(&metrics);
        assert_eq!(report.primary, BottleneckType::CpuBound);
    }

    #[test]
    fn test_io_bound_classification() {
        let metrics = io_metrics(0.60, 1000.0);
        let report = BottleneckClassifier::new().classify(&metrics);
        assert_eq!(report.primary, BottleneckType::IoBound);
    }

    #[test]
    fn test_memory_bound_classification() {
        let metrics = memory_metrics(2000.0, 1000.0); // 2 000 allocs/ms
        let report = BottleneckClassifier::new().classify(&metrics);
        assert_eq!(report.primary, BottleneckType::MemoryBound);
    }

    #[test]
    fn test_lock_contention_classification() {
        let metrics = lock_metrics(0.50, 1000.0);
        let report = BottleneckClassifier::new().classify(&metrics);
        assert_eq!(report.primary, BottleneckType::LockContention);
    }

    #[test]
    fn test_unknown_when_no_data() {
        let metrics = ProfilingMetrics::default();
        let report = BottleneckClassifier::new().classify(&metrics);
        assert_eq!(report.primary, BottleneckType::Unknown);
    }

    #[test]
    fn test_secondary_bottleneck_detected() {
        let metrics = ProfilingMetrics {
            wall_time_ms: Some(1000.0),
            cpu_time_ms: Some(800.0), // 80 % CPU → CpuBound primary
            io_wait_ms: Some(350.0),  // 35 % I/O → IoBound secondary (within 50% of primary score)
            ..Default::default()
        };
        let report = BottleneckClassifier::new().classify(&metrics);
        assert_eq!(report.primary, BottleneckType::CpuBound);
        assert!(report.secondary.is_some(), "expected secondary");
    }

    #[test]
    fn test_suggestions_non_empty() {
        let metrics = cpu_metrics(0.95, 500.0);
        let report = BottleneckClassifier::new().classify(&metrics);
        assert!(!report.suggestions.is_empty());
    }

    #[test]
    fn test_summary_contains_label() {
        let metrics = io_metrics(0.70, 500.0);
        let report = BottleneckClassifier::new().classify(&metrics);
        assert!(
            report.summary.contains("I/O-bound"),
            "summary was: {}",
            report.summary
        );
    }

    #[test]
    fn test_scores_populated() {
        let metrics = cpu_metrics(0.85, 1000.0);
        let report = BottleneckClassifier::new().classify(&metrics);
        assert!(report.scores.contains_key("CpuBound"));
    }

    #[test]
    fn test_bottleneck_type_labels() {
        assert_eq!(BottleneckType::CpuBound.label(), "CPU-bound");
        assert_eq!(BottleneckType::IoBound.label(), "I/O-bound");
        assert_eq!(BottleneckType::MemoryBound.label(), "Memory-bound");
        assert_eq!(BottleneckType::LockContention.label(), "Lock contention");
        assert_eq!(BottleneckType::Unknown.label(), "Unknown");
    }
}
