//! Pipeline latency profiling.
//!
//! Record per-stage latency samples and compute statistics including
//! average, P99, and end-to-end path latency.

/// A single latency measurement for a named pipeline stage.
#[derive(Debug, Clone)]
pub struct LatencySample {
    /// Pipeline stage name.
    pub stage: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}

impl LatencySample {
    /// Create a new latency sample.
    pub fn new(stage: &str, start_ms: u64, end_ms: u64) -> Self {
        Self {
            stage: stage.to_string(),
            start_ms,
            end_ms,
        }
    }

    /// Return the duration of this sample in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Accumulates latency samples across pipeline stages.
#[derive(Debug, Default)]
pub struct LatencyProfiler {
    samples: Vec<LatencySample>,
}

impl LatencyProfiler {
    /// Create a new, empty latency profiler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new sample for `stage` spanning [`start`, `end`] (in ms).
    pub fn record(&mut self, stage: &str, start: u64, end: u64) {
        self.samples.push(LatencySample::new(stage, start, end));
    }

    /// Return all samples for the named `stage`.
    pub fn samples_for(&self, stage: &str) -> Vec<&LatencySample> {
        self.samples.iter().filter(|s| s.stage == stage).collect()
    }

    /// Return the average latency (ms) for `stage`, or `0.0` if no samples exist.
    pub fn avg_latency_ms(&self, stage: &str) -> f64 {
        let samples = self.samples_for(stage);
        if samples.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let total: f64 = samples.iter().map(|s| s.duration_ms() as f64).sum();
        #[allow(clippy::cast_precision_loss)]
        let count = samples.len() as f64;
        total / count
    }

    /// Return the 99th-percentile latency (ms) for `stage`, or `0` if no samples.
    pub fn p99_latency_ms(&self, stage: &str) -> u64 {
        let mut durations: Vec<u64> = self
            .samples_for(stage)
            .iter()
            .map(|s| s.duration_ms())
            .collect();
        if durations.is_empty() {
            return 0;
        }
        durations.sort_unstable();
        let idx = ((durations.len() as f64) * 0.99).ceil() as usize;
        let idx = idx.saturating_sub(1).min(durations.len() - 1);
        durations[idx]
    }

    /// Return the name of the stage with the highest average latency, if any.
    pub fn slowest_stage(&self) -> Option<&str> {
        // Collect unique stage names preserving insertion order.
        let mut seen: Vec<&str> = vec![];
        for s in &self.samples {
            if !seen.contains(&s.stage.as_str()) {
                seen.push(&s.stage);
            }
        }
        seen.into_iter().max_by(|a, b| {
            self.avg_latency_ms(a)
                .partial_cmp(&self.avg_latency_ms(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the total number of samples recorded.
    pub fn total_samples(&self) -> usize {
        self.samples.len()
    }
}

/// The total latency measured end-to-end across a sequence of pipeline stages.
#[derive(Debug, Clone)]
pub struct EndToEndLatency {
    /// Ordered list of stage names included in the path.
    pub path: Vec<String>,
    /// Total latency in milliseconds (sum of per-stage averages).
    pub total_ms: u64,
}

impl EndToEndLatency {
    /// Return the slice of stage names in the path.
    pub fn stages(&self) -> &[String] {
        &self.path
    }
}

/// Compute the end-to-end latency across `stages` by summing their average latencies.
pub fn compute_end_to_end(profiler: &LatencyProfiler, stages: &[&str]) -> EndToEndLatency {
    #[allow(clippy::cast_possible_truncation)]
    let total_ms: u64 = stages
        .iter()
        .map(|s| profiler.avg_latency_ms(s) as u64)
        .sum();
    EndToEndLatency {
        path: stages.iter().map(|s| s.to_string()).collect(),
        total_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_duration_normal() {
        let s = LatencySample::new("decode", 100, 150);
        assert_eq!(s.duration_ms(), 50);
    }

    #[test]
    fn test_sample_duration_zero_when_equal() {
        let s = LatencySample::new("decode", 200, 200);
        assert_eq!(s.duration_ms(), 0);
    }

    #[test]
    fn test_sample_duration_saturating() {
        let s = LatencySample::new("decode", 300, 100);
        assert_eq!(s.duration_ms(), 0);
    }

    #[test]
    fn test_record_and_samples_for() {
        let mut p = LatencyProfiler::new();
        p.record("decode", 0, 10);
        p.record("encode", 10, 25);
        p.record("decode", 30, 45);
        assert_eq!(p.samples_for("decode").len(), 2);
        assert_eq!(p.samples_for("encode").len(), 1);
    }

    #[test]
    fn test_avg_latency_ms_correct() {
        let mut p = LatencyProfiler::new();
        p.record("decode", 0, 10); // 10 ms
        p.record("decode", 0, 20); // 20 ms
        let avg = p.avg_latency_ms("decode");
        assert!((avg - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_latency_ms_no_samples() {
        let p = LatencyProfiler::new();
        assert_eq!(p.avg_latency_ms("missing"), 0.0);
    }

    #[test]
    fn test_p99_latency_single_sample() {
        let mut p = LatencyProfiler::new();
        p.record("encode", 0, 50);
        assert_eq!(p.p99_latency_ms("encode"), 50);
    }

    #[test]
    fn test_p99_latency_no_samples() {
        let p = LatencyProfiler::new();
        assert_eq!(p.p99_latency_ms("encode"), 0);
    }

    #[test]
    fn test_p99_latency_selects_high_percentile() {
        let mut p = LatencyProfiler::new();
        for i in 1u64..=100 {
            p.record("stage", 0, i);
        }
        // P99 of [1..100] should be 99 or 100
        let p99 = p.p99_latency_ms("stage");
        assert!(p99 >= 99);
    }

    #[test]
    fn test_slowest_stage_single() {
        let mut p = LatencyProfiler::new();
        p.record("fast", 0, 5);
        p.record("slow", 0, 100);
        assert_eq!(p.slowest_stage(), Some("slow"));
    }

    #[test]
    fn test_slowest_stage_none_when_empty() {
        let p = LatencyProfiler::new();
        assert!(p.slowest_stage().is_none());
    }

    #[test]
    fn test_compute_end_to_end_sum() {
        let mut p = LatencyProfiler::new();
        p.record("ingest", 0, 10);
        p.record("decode", 0, 20);
        p.record("encode", 0, 30);
        let e2e = compute_end_to_end(&p, &["ingest", "decode", "encode"]);
        assert_eq!(e2e.total_ms, 60);
        assert_eq!(e2e.stages().len(), 3);
    }

    #[test]
    fn test_compute_end_to_end_missing_stage_zero() {
        let mut p = LatencyProfiler::new();
        p.record("decode", 0, 20);
        let e2e = compute_end_to_end(&p, &["decode", "ghost"]);
        assert_eq!(e2e.total_ms, 20);
    }

    #[test]
    fn test_total_samples_count() {
        let mut p = LatencyProfiler::new();
        p.record("a", 0, 1);
        p.record("b", 0, 2);
        p.record("a", 0, 3);
        assert_eq!(p.total_samples(), 3);
    }
}
