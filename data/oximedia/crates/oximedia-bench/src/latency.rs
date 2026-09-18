//! Latency benchmarking tools.
//!
//! This module provides structures for measuring, summarising, and reporting
//! the latency distribution of media processing operations, together with
//! SLA compliance checks and latency-budget accounting.

/// A single latency sample recording the start and end of an operation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LatencySample {
    /// Operation start time in milliseconds (monotonic).
    pub start_ms: u64,
    /// Operation end time in milliseconds (monotonic).
    pub end_ms: u64,
}

impl LatencySample {
    /// Duration of the operation in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Statistical distribution over a set of latency samples.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct LatencyDistribution {
    /// Raw latency samples in milliseconds (may be unsorted).
    pub samples: Vec<u64>,
}

impl LatencyDistribution {
    /// Create a new distribution from a list of latency samples.
    #[must_use]
    pub fn from_samples(samples: Vec<u64>) -> Self {
        Self { samples }
    }

    /// Arithmetic mean latency in milliseconds.
    #[must_use]
    pub fn mean_ms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<u64>() as f64 / self.samples.len() as f64
    }

    /// 50th-percentile (median) latency.
    #[must_use]
    pub fn p50_ms(&self) -> u64 {
        self.percentile_u64(50.0)
    }

    /// 95th-percentile latency.
    #[must_use]
    pub fn p95_ms(&self) -> u64 {
        self.percentile_u64(95.0)
    }

    /// 99th-percentile latency.
    #[must_use]
    pub fn p99_ms(&self) -> u64 {
        self.percentile_u64(99.0)
    }

    /// 99.9th-percentile latency.
    #[must_use]
    pub fn p999_ms(&self) -> u64 {
        self.percentile_u64(99.9)
    }

    /// Standard deviation of latency in milliseconds.
    #[must_use]
    pub fn std_dev_ms(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_ms();
        let variance = self
            .samples
            .iter()
            .map(|&s| {
                let d = s as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / (self.samples.len() - 1) as f64;
        variance.sqrt()
    }

    /// Build a histogram with `buckets` equal-width bins.
    ///
    /// Returns a `Vec<(bucket_upper_ms, count)>` where `bucket_upper_ms` is the
    /// inclusive upper bound of the bucket in milliseconds.
    #[must_use]
    pub fn histogram(&self, buckets: u32) -> Vec<(u64, u32)> {
        if self.samples.is_empty() || buckets == 0 {
            return Vec::new();
        }

        let min = *self.samples.iter().min().unwrap_or(&0);
        let max = *self.samples.iter().max().unwrap_or(&0);

        if min == max {
            return vec![(max, self.samples.len() as u32)];
        }

        let range = max - min;
        let bucket_width = ((range as f64 / buckets as f64).ceil() as u64).max(1);

        let mut counts = vec![0u32; buckets as usize];
        for &s in &self.samples {
            let idx = ((s - min) / bucket_width) as usize;
            let idx = idx.min(buckets as usize - 1);
            counts[idx] += 1;
        }

        counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let upper = min + (i as u64 + 1) * bucket_width;
                (upper, count)
            })
            .collect()
    }

    // Internal: nth percentile (nearest-rank, ceiling method).
    fn percentile_u64(&self, p: f64) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let raw = p / 100.0 * sorted.len() as f64;
        let idx = if raw.fract() < f64::EPSILON {
            (raw as usize).saturating_sub(1)
        } else {
            raw.ceil() as usize - 1
        }
        .min(sorted.len() - 1);
        sorted[idx]
    }
}

/// SLA compliance report for a latency distribution.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SlaCompliance {
    /// Target latency threshold in milliseconds.
    pub target_ms: u64,
    /// Percentage of samples that meet the SLA target.
    pub compliance_pct: f32,
    /// Number of samples exceeding the target.
    pub violations: u32,
}

impl SlaCompliance {
    /// Compute SLA compliance for a distribution against a target.
    #[must_use]
    pub fn compute(dist: &LatencyDistribution, target_ms: u64) -> Self {
        if dist.samples.is_empty() {
            return Self {
                target_ms,
                compliance_pct: 100.0,
                violations: 0,
            };
        }
        let violations = dist.samples.iter().filter(|&&s| s > target_ms).count() as u32;
        let compliance_pct = (1.0 - violations as f32 / dist.samples.len() as f32) * 100.0;
        Self {
            target_ms,
            compliance_pct,
            violations,
        }
    }

    /// Whether the SLA compliance meets or exceeds a required threshold.
    #[must_use]
    pub fn meets_threshold(&self, required_pct: f32) -> bool {
        self.compliance_pct >= required_pct
    }
}

/// A latency budget that partitions a total time allowance among named components.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LatencyBudget {
    /// Total available latency in milliseconds.
    pub total_ms: u64,
    /// Named allocations as (component_name, allocated_ms).
    pub allocations: Vec<(String, u64)>,
}

impl LatencyBudget {
    /// Create a budget from a slice of `(name, ms)` pairs.
    #[must_use]
    pub fn from_components(components: &[(&str, u64)]) -> Self {
        let total_ms: u64 = components.iter().map(|(_, ms)| ms).sum();
        let allocations = components
            .iter()
            .map(|(name, ms)| ((*name).to_string(), *ms))
            .collect();
        Self {
            total_ms,
            allocations,
        }
    }

    /// Remaining headroom in milliseconds (may be negative if over-allocated).
    #[must_use]
    pub fn remaining_ms(&self) -> i64 {
        let used: u64 = self.allocations.iter().map(|(_, ms)| ms).sum();
        self.total_ms as i64 - used as i64
    }

    /// Returns `true` if the budget is not over-allocated.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.remaining_ms() >= 0
    }

    /// Return a budget that replaces the total with the given value,
    /// keeping the current allocations.
    #[must_use]
    pub fn with_total(mut self, total_ms: u64) -> Self {
        self.total_ms = total_ms;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(values: &[u64]) -> LatencyDistribution {
        LatencyDistribution::from_samples(values.to_vec())
    }

    #[test]
    fn test_sample_duration() {
        let s = LatencySample {
            start_ms: 100,
            end_ms: 150,
        };
        assert_eq!(s.duration_ms(), 50);
    }

    #[test]
    fn test_sample_duration_saturating() {
        let s = LatencySample {
            start_ms: 200,
            end_ms: 100, // end < start
        };
        assert_eq!(s.duration_ms(), 0);
    }

    #[test]
    fn test_mean_ms() {
        let d = dist(&[10, 20, 30, 40, 50]);
        assert!((d.mean_ms() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_p50_ms() {
        let d = dist(&[10, 20, 30, 40, 50]);
        assert_eq!(d.p50_ms(), 30);
    }

    #[test]
    fn test_p95_ms() {
        let mut samples: Vec<u64> = (1..=100).collect();
        samples.sort_unstable();
        let d = LatencyDistribution::from_samples(samples);
        assert_eq!(d.p95_ms(), 95);
    }

    #[test]
    fn test_p99_ms() {
        let samples: Vec<u64> = (1..=100).collect();
        let d = LatencyDistribution::from_samples(samples);
        assert_eq!(d.p99_ms(), 99);
    }

    #[test]
    fn test_p999_ms() {
        // 1000 samples (1..=1000); p999 = ceil(99.9/100 * 1000) - 1 = index 999 = value 1000
        let samples: Vec<u64> = (1..=1000).collect();
        let d = LatencyDistribution::from_samples(samples);
        assert_eq!(d.p999_ms(), 1000);
    }

    #[test]
    fn test_std_dev() {
        let d = dist(&[2, 4, 4, 4, 5, 5, 7, 9]);
        let sd = d.std_dev_ms();
        assert!((sd - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_empty_distribution() {
        let d = dist(&[]);
        assert_eq!(d.mean_ms(), 0.0);
        assert_eq!(d.p50_ms(), 0);
        assert_eq!(d.std_dev_ms(), 0.0);
    }

    #[test]
    fn test_histogram_basic() {
        let d = dist(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let h = d.histogram(5);
        assert_eq!(h.len(), 5);
        let total_count: u32 = h.iter().map(|(_, c)| c).sum();
        assert_eq!(total_count, 10);
    }

    #[test]
    fn test_sla_compliance_all_pass() {
        let d = dist(&[10, 20, 30]);
        let sla = SlaCompliance::compute(&d, 50);
        assert_eq!(sla.violations, 0);
        assert!((sla.compliance_pct - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_sla_compliance_partial() {
        let d = dist(&[10, 20, 100, 200]);
        let sla = SlaCompliance::compute(&d, 50);
        assert_eq!(sla.violations, 2);
        assert!((sla.compliance_pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_sla_meets_threshold() {
        let d = dist(&[10, 20, 30, 40, 50]);
        let sla = SlaCompliance::compute(&d, 100);
        assert!(sla.meets_threshold(95.0));
    }

    #[test]
    fn test_latency_budget_valid() {
        let budget =
            LatencyBudget::from_components(&[("network", 20), ("decode", 10), ("render", 5)]);
        assert_eq!(budget.total_ms, 35);
        assert!(budget.is_valid());
        assert_eq!(budget.remaining_ms(), 0);
    }

    #[test]
    fn test_latency_budget_with_total() {
        let budget = LatencyBudget::from_components(&[("decode", 30)]).with_total(50);
        assert_eq!(budget.remaining_ms(), 20);
        assert!(budget.is_valid());
    }

    #[test]
    fn test_latency_budget_over_allocated() {
        let budget = LatencyBudget::from_components(&[("a", 50), ("b", 60)]).with_total(100);
        assert!(!budget.is_valid());
        assert!(budget.remaining_ms() < 0);
    }
}
