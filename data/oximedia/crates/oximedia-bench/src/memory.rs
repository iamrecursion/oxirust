//! Memory usage benchmarking tools.
//!
//! This module provides structures for capturing, analysing, and scoring
//! memory allocation behaviour during media processing operations.

/// A point-in-time snapshot of memory usage.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemorySnapshot {
    /// Heap allocation in bytes.
    pub heap_bytes: u64,
    /// Stack usage in bytes (estimated).
    pub stack_bytes: u64,
    /// Monotonic timestamp in milliseconds since the trace started.
    pub timestamp_ms: u64,
}

/// A sequence of memory snapshots captured during a benchmark run.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct MemoryTrace {
    /// Ordered list of snapshots.
    pub snapshots: Vec<MemorySnapshot>,
}

impl MemoryTrace {
    /// Create an empty trace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a snapshot.
    pub fn push(&mut self, snapshot: MemorySnapshot) {
        self.snapshots.push(snapshot);
    }

    /// Peak heap usage across all snapshots.
    #[must_use]
    pub fn peak_heap(&self) -> u64 {
        self.snapshots
            .iter()
            .map(|s| s.heap_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Mean heap usage across all snapshots.
    #[must_use]
    pub fn avg_heap(&self) -> u64 {
        if self.snapshots.is_empty() {
            return 0;
        }
        let total: u64 = self.snapshots.iter().map(|s| s.heap_bytes).sum();
        total / self.snapshots.len() as u64
    }

    /// Linear growth rate of heap usage in bytes per millisecond.
    ///
    /// A positive value indicates growing memory; negative indicates freeing.
    /// Returns `0.0` if fewer than two snapshots exist or the time span is zero.
    #[must_use]
    pub fn growth_rate_bytes_per_ms(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        let first = &self.snapshots[0];
        let last = &self.snapshots[self.snapshots.len() - 1];
        let dt = (last.timestamp_ms as i64 - first.timestamp_ms as i64) as f64;
        if dt.abs() < f64::EPSILON {
            return 0.0;
        }
        let dh = last.heap_bytes as f64 - first.heap_bytes as f64;
        dh / dt
    }

    /// Variance of heap values.
    fn variance(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_heap() as f64;
        let sum_sq: f64 = self
            .snapshots
            .iter()
            .map(|s| {
                let d = s.heap_bytes as f64 - mean;
                d * d
            })
            .sum();
        sum_sq / self.snapshots.len() as f64
    }
}

/// Classification of an observed memory allocation pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AllocationPattern {
    /// Stable memory usage with low variance.
    Steady,
    /// High-variance, burst-style allocations.
    Spiky,
    /// Gradual, roughly linear increase.
    Linear,
    /// Consistent upward trend suggesting a memory leak.
    Leaking,
}

impl AllocationPattern {
    /// Detect the dominant allocation pattern from a memory trace.
    ///
    /// Heuristics:
    /// - **Leaking**: growth rate > 1 KB/ms (consistent growth).
    /// - **Spiky**: coefficient of variation > 0.5 (high relative variance).
    /// - **Linear**: moderate growth rate with low variance.
    /// - **Steady**: everything else.
    #[must_use]
    pub fn detect(trace: &MemoryTrace) -> Self {
        let rate = trace.growth_rate_bytes_per_ms();
        let mean = trace.avg_heap() as f64;
        let variance = trace.variance();

        // CV = std_dev / mean (relative variability)
        let cv = if mean.abs() > f64::EPSILON {
            variance.sqrt() / mean
        } else {
            0.0
        };

        if rate > 1024.0 {
            // > 1 KB/ms consistent growth → leaking
            Self::Leaking
        } else if cv > 0.5 {
            Self::Spiky
        } else if rate > 10.0 {
            // Moderate positive growth with low variance → linear
            Self::Linear
        } else {
            Self::Steady
        }
    }
}

/// Summary of memory performance for a single operation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryBenchmarkResult {
    /// Name of the benchmarked operation.
    pub operation: String,
    /// Average number of allocations per operation invocation.
    pub allocations_per_op: f32,
    /// Average bytes allocated per operation invocation.
    pub bytes_per_op: u64,
    /// Peak heap usage in megabytes.
    pub peak_mb: f32,
}

/// Memory efficiency score (0–100) with a letter grade.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryEfficiencyScore {
    /// Numeric score in the range [0.0, 100.0].
    pub score: f32,
    /// Letter grade: A, B, C, D, or F.
    pub grade: char,
}

impl MemoryEfficiencyScore {
    /// Compute a score and grade from a benchmark result.
    ///
    /// The score is derived from `bytes_per_op`:
    /// - ≤ 4 KB/op → A (90–100)
    /// - ≤ 64 KB/op → B (75–89)
    /// - ≤ 512 KB/op → C (60–74)
    /// - ≤ 4 MB/op → D (40–59)
    /// - > 4 MB/op → F (<40)
    #[must_use]
    pub fn compute(result: &MemoryBenchmarkResult) -> Self {
        let bpo = result.bytes_per_op;
        let (score, grade) = if bpo <= 4_096 {
            (95.0, 'A')
        } else if bpo <= 65_536 {
            (82.0, 'B')
        } else if bpo <= 524_288 {
            (67.0, 'C')
        } else if bpo <= 4_194_304 {
            (50.0, 'D')
        } else {
            (20.0, 'F')
        };
        Self { score, grade }
    }
}

/// Simulated cache efficiency metrics derived from access-pattern analysis.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CacheEfficiency {
    /// Estimated L1 cache hit percentage.
    pub l1_hits_pct: f32,
    /// Estimated L2 cache hit percentage.
    pub l2_hits_pct: f32,
    /// Estimated L3 cache hit percentage.
    pub l3_hits_pct: f32,
}

impl CacheEfficiency {
    /// Simulate cache efficiency based on the memory access stride.
    ///
    /// Smaller strides (sequential access) yield higher hit rates.
    /// `stride_bytes` is the typical distance between consecutive accesses.
    #[must_use]
    pub fn from_stride(stride_bytes: u32) -> Self {
        // Cache line is 64 bytes; sequential access maximises L1 hits.
        let l1 = if stride_bytes <= 64 {
            98.0
        } else if stride_bytes <= 256 {
            85.0
        } else if stride_bytes <= 4096 {
            60.0
        } else {
            30.0
        };

        let l2 = if stride_bytes <= 256 {
            98.0
        } else if stride_bytes <= 4096 {
            80.0
        } else {
            50.0
        };

        let l3 = if stride_bytes <= 4096 { 95.0 } else { 70.0 };

        Self {
            l1_hits_pct: l1,
            l2_hits_pct: l2,
            l3_hits_pct: l3,
        }
    }

    /// Overall cache hit rate (weighted average of L1, L2, L3).
    #[must_use]
    pub fn overall_hit_pct(&self) -> f32 {
        (self.l1_hits_pct * 0.6 + self.l2_hits_pct * 0.3 + self.l3_hits_pct * 0.1) / 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trace(values: &[(u64, u64)]) -> MemoryTrace {
        let mut trace = MemoryTrace::new();
        for &(ts, heap) in values {
            trace.push(MemorySnapshot {
                heap_bytes: heap,
                stack_bytes: 4096,
                timestamp_ms: ts,
            });
        }
        trace
    }

    #[test]
    fn test_peak_heap() {
        let trace = make_trace(&[(0, 1000), (10, 5000), (20, 3000)]);
        assert_eq!(trace.peak_heap(), 5000);
    }

    #[test]
    fn test_avg_heap() {
        let trace = make_trace(&[(0, 1000), (10, 3000), (20, 5000)]);
        assert_eq!(trace.avg_heap(), 3000);
    }

    #[test]
    fn test_empty_trace_peak() {
        let trace = MemoryTrace::new();
        assert_eq!(trace.peak_heap(), 0);
        assert_eq!(trace.avg_heap(), 0);
    }

    #[test]
    fn test_growth_rate_positive() {
        let trace = make_trace(&[(0, 0), (1000, 1_000_000)]);
        // 1 MB over 1000 ms = 1000 bytes/ms
        assert!((trace.growth_rate_bytes_per_ms() - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_growth_rate_zero_span() {
        let trace = make_trace(&[(0, 1000)]);
        assert_eq!(trace.growth_rate_bytes_per_ms(), 0.0);
    }

    #[test]
    fn test_pattern_leaking() {
        // > 1 KB/ms growth
        let trace = make_trace(&[(0, 0), (1000, 10_000_000)]);
        assert_eq!(
            AllocationPattern::detect(&trace),
            AllocationPattern::Leaking
        );
    }

    #[test]
    fn test_pattern_steady() {
        let trace = make_trace(&[(0, 100_000), (100, 100_100), (200, 100_050), (300, 100_000)]);
        assert_eq!(AllocationPattern::detect(&trace), AllocationPattern::Steady);
    }

    #[test]
    fn test_pattern_spiky() {
        // High variance (CV > 0.5) but first and last values are close so
        // the growth rate stays below the 1 KB/ms leaking threshold.
        let trace = make_trace(&[
            (0, 1_000_000),
            (100, 10_000_000),
            (200, 1_000_000),
            (300, 9_000_000),
            (400, 1_050_000), // end ≈ start → low growth rate
        ]);
        assert_eq!(AllocationPattern::detect(&trace), AllocationPattern::Spiky);
    }

    #[test]
    fn test_efficiency_score_grade_a() {
        let result = MemoryBenchmarkResult {
            operation: "test".to_string(),
            allocations_per_op: 1.0,
            bytes_per_op: 512,
            peak_mb: 1.0,
        };
        let score = MemoryEfficiencyScore::compute(&result);
        assert_eq!(score.grade, 'A');
        assert!(score.score >= 90.0);
    }

    #[test]
    fn test_efficiency_score_grade_f() {
        let result = MemoryBenchmarkResult {
            operation: "heavy".to_string(),
            allocations_per_op: 1000.0,
            bytes_per_op: 10_000_000,
            peak_mb: 500.0,
        };
        let score = MemoryEfficiencyScore::compute(&result);
        assert_eq!(score.grade, 'F');
    }

    #[test]
    fn test_cache_efficiency_sequential() {
        let ce = CacheEfficiency::from_stride(8); // 8-byte stride = sequential
        assert!(ce.l1_hits_pct >= 95.0);
    }

    #[test]
    fn test_cache_efficiency_random() {
        let ce = CacheEfficiency::from_stride(65536); // large stride = random
        assert!(ce.l1_hits_pct < 50.0);
    }

    #[test]
    fn test_cache_overall_hit_pct() {
        let ce = CacheEfficiency::from_stride(64);
        let overall = ce.overall_hit_pct();
        assert!(overall > 90.0);
    }
}
