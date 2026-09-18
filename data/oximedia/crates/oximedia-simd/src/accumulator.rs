//! Running statistics accumulator for audio/video sample streams.
//!
//! [`SampleAccumulator`] computes mean and variance incrementally using
//! Welford's online algorithm, which is numerically stable and requires only
//! a single pass over the data.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A snapshot of statistics captured from a [`SampleAccumulator`].
///
/// # Examples
///
/// ```
/// use oximedia_simd::accumulator::SampleAccumulator;
/// let mut acc = SampleAccumulator::new();
/// for v in [1.0_f64, 2.0, 3.0, 4.0, 5.0] { acc.push(v); }
/// let s = acc.stats();
/// assert!((s.mean - 3.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccumulatorStats {
    /// Number of samples observed.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Population variance (0.0 when fewer than 2 samples).
    pub variance: f64,
    /// Minimum value observed.
    pub min: f64,
    /// Maximum value observed.
    pub max: f64,
}

impl AccumulatorStats {
    /// Returns the population standard deviation (sqrt of variance).
    #[inline]
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Returns the sample variance (Bessel-corrected), or `0.0` if `count < 2`.
    #[inline]
    #[must_use]
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.variance * self.count as f64 / (self.count - 1) as f64
    }

    /// Returns the range `max - min`, or `0.0` if no samples.
    #[inline]
    #[must_use]
    pub fn range(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max - self.min
        }
    }
}

/// An incremental accumulator that tracks mean and variance over a stream of
/// `f64` samples using Welford's online algorithm.
///
/// # Examples
///
/// ```
/// use oximedia_simd::accumulator::SampleAccumulator;
///
/// let mut acc = SampleAccumulator::new();
/// acc.push(10.0);
/// acc.push(20.0);
/// acc.push(30.0);
/// let s = acc.stats();
/// assert!((s.mean - 20.0).abs() < 1e-9);
/// assert_eq!(s.count, 3);
/// ```
#[derive(Debug, Clone)]
pub struct SampleAccumulator {
    count: u64,
    mean: f64,
    /// Running sum of squared deviations from the mean (M2 in Welford's alg.).
    m2: f64,
    min: f64,
    max: f64,
}

impl SampleAccumulator {
    /// Creates a new empty accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Resets the accumulator to its initial empty state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Incorporates a new sample value using Welford's online algorithm.
    ///
    /// This operation is O(1) and numerically stable.
    pub fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Pushes every element of a slice in order.
    pub fn push_slice(&mut self, values: &[f64]) {
        for &v in values {
            self.push(v);
        }
    }

    /// Returns the number of samples observed so far.
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Returns the current running mean, or `0.0` if no samples.
    #[inline]
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.mean
        }
    }

    /// Returns the current population variance, or `0.0` if fewer than 2 samples.
    #[inline]
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Returns the minimum value, or `f64::INFINITY` if no samples.
    #[inline]
    #[must_use]
    pub const fn min(&self) -> f64 {
        self.min
    }

    /// Returns the maximum value, or `f64::NEG_INFINITY` if no samples.
    #[inline]
    #[must_use]
    pub const fn max(&self) -> f64 {
        self.max
    }

    /// Returns a [`AccumulatorStats`] snapshot of the current state.
    #[must_use]
    pub fn stats(&self) -> AccumulatorStats {
        AccumulatorStats {
            count: self.count,
            mean: self.mean(),
            variance: self.variance(),
            min: if self.count == 0 { 0.0 } else { self.min },
            max: if self.count == 0 { 0.0 } else { self.max },
        }
    }

    /// Merges the statistics from `other` into `self`.
    ///
    /// Uses the parallel / Chan's algorithm for combining two Welford
    /// accumulators without needing the original samples.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean =
            (self.mean * self.count as f64 + other.mean * other.count as f64) / total as f64;
        let new_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.count = total;
        self.mean = new_mean;
        self.m2 = new_m2;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }
}

impl Default for SampleAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_accumulator_zero_count() {
        let acc = SampleAccumulator::new();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.mean(), 0.0);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn single_sample_mean() {
        let mut acc = SampleAccumulator::new();
        acc.push(42.0);
        assert!((acc.mean() - 42.0).abs() < 1e-10);
        assert_eq!(acc.count(), 1);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn two_samples_mean() {
        let mut acc = SampleAccumulator::new();
        acc.push(0.0);
        acc.push(10.0);
        assert!((acc.mean() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn known_variance() {
        // Values 1..=5 → mean=3, pop-variance=2
        let mut acc = SampleAccumulator::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            acc.push(v);
        }
        assert!((acc.mean() - 3.0).abs() < 1e-10);
        assert!((acc.variance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn std_dev_non_negative() {
        let mut acc = SampleAccumulator::new();
        acc.push_slice(&[10.0, 20.0, 30.0]);
        assert!(acc.stats().std_dev() >= 0.0);
    }

    #[test]
    fn min_max_tracking() {
        let mut acc = SampleAccumulator::new();
        acc.push_slice(&[5.0, 1.0, 9.0, 3.0]);
        assert_eq!(acc.min(), 1.0);
        assert_eq!(acc.max(), 9.0);
    }

    #[test]
    fn range_equals_max_minus_min() {
        let mut acc = SampleAccumulator::new();
        acc.push_slice(&[2.0, 8.0, 5.0]);
        let s = acc.stats();
        assert!((s.range() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn empty_stats_range_zero() {
        let acc = SampleAccumulator::new();
        assert_eq!(acc.stats().range(), 0.0);
    }

    #[test]
    fn sample_variance_bessel_correction() {
        let mut acc = SampleAccumulator::new();
        // values: 2, 4  — sample variance = ((2-3)^2 + (4-3)^2) / (2-1) = 2
        acc.push(2.0);
        acc.push(4.0);
        let s = acc.stats();
        assert!((s.sample_variance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn reset_clears_state() {
        let mut acc = SampleAccumulator::new();
        acc.push_slice(&[1.0, 2.0, 3.0]);
        acc.reset();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.mean(), 0.0);
    }

    #[test]
    fn merge_two_accumulators() {
        let mut a = SampleAccumulator::new();
        a.push_slice(&[1.0, 2.0, 3.0]);

        let mut b = SampleAccumulator::new();
        b.push_slice(&[4.0, 5.0, 6.0]);

        a.merge(&b);
        assert_eq!(a.count(), 6);
        assert!((a.mean() - 3.5).abs() < 1e-10);
    }

    #[test]
    fn merge_with_empty_other_is_noop() {
        let mut a = SampleAccumulator::new();
        a.push(7.0);
        let before_mean = a.mean();
        let empty = SampleAccumulator::new();
        a.merge(&empty);
        assert!((a.mean() - before_mean).abs() < 1e-10);
        assert_eq!(a.count(), 1);
    }

    #[test]
    fn merge_into_empty_self_copies() {
        let mut a = SampleAccumulator::new();
        let mut b = SampleAccumulator::new();
        b.push_slice(&[10.0, 20.0]);
        a.merge(&b);
        assert_eq!(a.count(), 2);
        assert!((a.mean() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn push_slice_matches_individual_pushes() {
        let mut a = SampleAccumulator::new();
        let mut b = SampleAccumulator::new();
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        a.push_slice(&data);
        for v in data {
            b.push(v);
        }
        assert!((a.mean() - b.mean()).abs() < 1e-10);
        assert!((a.variance() - b.variance()).abs() < 1e-10);
    }

    #[test]
    fn default_is_empty() {
        let acc = SampleAccumulator::default();
        assert_eq!(acc.count(), 0);
    }
}
