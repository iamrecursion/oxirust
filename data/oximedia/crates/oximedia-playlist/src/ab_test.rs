//! A/B testing support for playlist ordering strategy evaluation.
//!
//! Provides deterministic, hash-bucket-based experiment assignment so that
//! the same user is always placed in the same variant across calls, enabling
//! accurate per-variant metric aggregation.

/// Compute a stable FNV-1a 64-bit hash of a string.
///
/// This is used in preference to `std::hash::DefaultHasher`, which is
/// explicitly documented as non-stable across Rust versions.
fn fnv1a_hash(s: &str) -> u64 {
    const FNV_PRIME: u64 = 1_099_511_628_211;
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    let mut hash = OFFSET_BASIS;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Strategy that determines how playlist items are ordered for a variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderingStrategy {
    /// The original, unmodified ordering.
    Default,
    /// Randomised shuffle.
    Shuffle,
    /// Order by popularity score (highest first).
    ByPopularity,
    /// Order by recency (most recently added first).
    ByRecency,
}

/// Accumulates impression, completion, and skip events for a single variant.
#[derive(Debug, Clone)]
pub struct MetricAccumulator {
    /// Total number of item impressions recorded.
    pub impressions: u64,
    /// Number of items played to completion.
    pub completions: u64,
    /// Number of items that were skipped.
    pub skips: u64,
}

impl MetricAccumulator {
    /// Create a zeroed accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            impressions: 0,
            completions: 0,
            skips: 0,
        }
    }

    /// Record one impression event.
    pub fn record_impression(&mut self) {
        self.impressions += 1;
    }

    /// Record one completion event.
    pub fn record_completion(&mut self) {
        self.completions += 1;
    }

    /// Record one skip event.
    pub fn record_skip(&mut self) {
        self.skips += 1;
    }

    /// Completion rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if no impressions have been recorded.
    #[must_use]
    pub fn completion_rate(&self) -> f32 {
        if self.impressions == 0 {
            return 0.0;
        }
        self.completions as f32 / self.impressions as f32
    }
}

impl Default for MetricAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// An A/B experiment that assigns users to ordering strategy variants.
///
/// Variant assignment is deterministic: the same `(user_id, experiment_id)`
/// pair always maps to the same variant, regardless of call order or time.
#[derive(Debug)]
pub struct AbExperiment {
    /// Stable experiment identifier.
    pub id: String,
    /// Ordered list of variant strategies.
    pub variants: Vec<OrderingStrategy>,
    /// Traffic fractions for each variant; must sum to 1.0 and have the same
    /// length as `variants`.
    pub traffic_split: Vec<f64>,
    /// Per-variant metric accumulators.
    pub metrics: Vec<MetricAccumulator>,
}

impl AbExperiment {
    /// Create a new experiment.
    ///
    /// # Panics
    ///
    /// Panics if `variants.len() != traffic_split.len()`.
    pub fn new(
        id: impl Into<String>,
        variants: Vec<OrderingStrategy>,
        traffic_split: Vec<f64>,
    ) -> Self {
        assert_eq!(
            variants.len(),
            traffic_split.len(),
            "variants and traffic_split must have the same length"
        );
        let n = variants.len();
        Self {
            id: id.into(),
            variants,
            traffic_split,
            metrics: vec![MetricAccumulator::new(); n],
        }
    }

    /// Deterministically assign a variant to `user_id` using a FNV-1a hash.
    ///
    /// The key hashed is `"{user_id}:{experiment_id}"`.  The resulting hash
    /// is reduced to a bucket index via cumulative traffic fractions.
    #[must_use]
    pub fn assign_variant(&self, user_id: &str) -> &OrderingStrategy {
        let key = format!("{}:{}", user_id, self.id);
        let hash = fnv1a_hash(&key);
        // Map hash to a value in [0.0, 1.0) using 10 000 buckets for
        // sufficient resolution without expensive floating-point modulo.
        let t = (hash % 10_000) as f64 / 10_000.0;
        let mut cumulative = 0.0_f64;
        for (i, &split) in self.traffic_split.iter().enumerate() {
            cumulative += split;
            if t < cumulative {
                return &self.variants[i];
            }
        }
        // Fallback to the last variant (handles floating-point rounding).
        &self.variants[self.variants.len() - 1]
    }

    /// Return the 0-based index of the variant assigned to `user_id`, or
    /// `None` if the variants list is empty.
    #[must_use]
    pub fn variant_index(&self, user_id: &str) -> Option<usize> {
        if self.variants.is_empty() {
            return None;
        }
        let key = format!("{}:{}", user_id, self.id);
        let hash = fnv1a_hash(&key);
        let t = (hash % 10_000) as f64 / 10_000.0;
        let mut cumulative = 0.0_f64;
        for (i, &split) in self.traffic_split.iter().enumerate() {
            cumulative += split;
            if t < cumulative {
                return Some(i);
            }
        }
        Some(self.variants.len() - 1)
    }

    /// Record an impression for the given variant index.
    pub fn record_impression(&mut self, variant_idx: usize) {
        if let Some(m) = self.metrics.get_mut(variant_idx) {
            m.record_impression();
        }
    }

    /// Record a completion for the given variant index.
    pub fn record_completion(&mut self, variant_idx: usize) {
        if let Some(m) = self.metrics.get_mut(variant_idx) {
            m.record_completion();
        }
    }

    /// Record a skip for the given variant index.
    pub fn record_skip(&mut self, variant_idx: usize) {
        if let Some(m) = self.metrics.get_mut(variant_idx) {
            m.record_skip();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_variant_experiment() -> AbExperiment {
        AbExperiment::new(
            "exp_ordering",
            vec![OrderingStrategy::Default, OrderingStrategy::ByPopularity],
            vec![0.5, 0.5],
        )
    }

    #[test]
    fn test_ab_assignment_is_deterministic() {
        let exp = two_variant_experiment();
        let first = exp.assign_variant("user123").clone();
        for _ in 0..99 {
            assert_eq!(exp.assign_variant("user123"), &first);
        }
    }

    #[test]
    fn test_ab_traffic_split_approximately_uniform() {
        let exp = two_variant_experiment();
        let mut counts = [0usize; 2];
        for i in 0..10_000usize {
            let user_id = format!("user_{i}");
            let variant = exp.assign_variant(&user_id);
            if *variant == OrderingStrategy::Default {
                counts[0] += 1;
            } else {
                counts[1] += 1;
            }
        }
        // Each variant should receive between 45 % and 55 % of assignments.
        for (i, &count) in counts.iter().enumerate() {
            let fraction = count as f64 / 10_000.0;
            assert!(
                (0.45..=0.55).contains(&fraction),
                "variant {i} got {fraction:.3} of assignments (expected 0.45–0.55)"
            );
        }
    }

    #[test]
    fn test_ab_metric_accumulator() {
        let mut acc = MetricAccumulator::new();
        acc.record_impression();
        acc.record_impression();
        acc.record_impression();
        acc.record_completion();
        acc.record_completion();
        assert_eq!(acc.impressions, 3);
        assert_eq!(acc.completions, 2);
        let rate = acc.completion_rate();
        assert!((rate - (2.0_f32 / 3.0)).abs() < 1e-5, "rate={rate}");
    }

    #[test]
    fn test_ab_metric_accumulator_zero_impressions() {
        let acc = MetricAccumulator::new();
        assert_eq!(acc.completion_rate(), 0.0);
    }

    #[test]
    fn test_ab_four_variant_experiment() {
        let exp = AbExperiment::new(
            "four_way",
            vec![
                OrderingStrategy::Default,
                OrderingStrategy::Shuffle,
                OrderingStrategy::ByPopularity,
                OrderingStrategy::ByRecency,
            ],
            vec![0.25, 0.25, 0.25, 0.25],
        );
        // Each variant should receive roughly 25 % of 10 000 assignments.
        let mut counts = [0usize; 4];
        for i in 0..10_000usize {
            let user_id = format!("u{i}");
            if let Some(idx) = exp.variant_index(&user_id) {
                counts[idx] += 1;
            }
        }
        for (i, &count) in counts.iter().enumerate() {
            let fraction = count as f64 / 10_000.0;
            assert!(
                (0.20..=0.30).contains(&fraction),
                "variant {i} got {fraction:.3} of assignments (expected 0.20–0.30)"
            );
        }
    }

    #[test]
    fn test_ab_record_impression_and_completion() {
        let mut exp = two_variant_experiment();
        exp.record_impression(0);
        exp.record_impression(0);
        exp.record_completion(0);
        assert_eq!(exp.metrics[0].impressions, 2);
        assert_eq!(exp.metrics[0].completions, 1);
        // Out-of-bounds index must not panic.
        exp.record_impression(99);
        exp.record_completion(99);
    }
}
