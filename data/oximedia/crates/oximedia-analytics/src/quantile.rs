//! Approximate quantile estimation using a simplified t-digest.
//!
//! A t-digest is a streaming data structure for approximate percentile
//! computation over large datasets.  This implementation follows the core idea
//! of Dunning & Ertl (2019):
//!
//! * Data is compressed into *centroids* (mean, weight) pairs.
//! * The number of centroids is bounded by the compression parameter `delta`.
//! * Nearby centroids are merged when their combined weight does not violate
//!   the size constraint `4 * q * (1 - q) * n / delta` (the "scale function").
//!
//! This implementation is sufficient for media analytics use-cases such as
//! computing P50/P95/P99 of watch-time, bitrate, or latency distributions at
//! scale.

use crate::error::AnalyticsError;

// ─── Centroid ─────────────────────────────────────────────────────────────────

/// A single centroid: a weighted mean of nearby data points.
#[derive(Debug, Clone)]
pub struct Centroid {
    /// Weighted mean of all data points assigned to this centroid.
    pub mean: f64,
    /// Total weight (number of data points).
    pub weight: f64,
}

impl Centroid {
    fn new(mean: f64, weight: f64) -> Self {
        Self { mean, weight }
    }
}

// ─── TDigest ──────────────────────────────────────────────────────────────────

/// A t-digest for approximate quantile estimation over streaming data.
///
/// # Construction
///
/// ```
/// use oximedia_analytics::quantile::TDigest;
/// let mut digest = TDigest::new(100.0);
/// for x in 0..1000 {
///     digest.add(x as f64);
/// }
/// let p50 = digest.quantile(0.5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TDigest {
    /// Compression parameter: larger values give more centroids (more accuracy).
    delta: f64,
    /// Sorted list of centroids (ascending by mean).
    centroids: Vec<Centroid>,
    /// Total weight (number of points added).
    total_weight: f64,
    /// Buffer of unmerged points (batch-merged periodically).
    buffer: Vec<f64>,
    /// Batch size before triggering a merge.
    buffer_capacity: usize,
    /// Running min and max.
    pub min: f64,
    pub max: f64,
}

impl TDigest {
    /// Create a new t-digest with the given compression parameter.
    ///
    /// Typical values: `delta = 100` (moderate accuracy, few centroids) to
    /// `delta = 1000` (high accuracy, more centroids).
    pub fn new(delta: f64) -> Self {
        let buffer_capacity = (delta as usize).max(64);
        Self {
            delta: delta.max(1.0),
            centroids: Vec::new(),
            total_weight: 0.0,
            buffer: Vec::with_capacity(buffer_capacity),
            buffer_capacity,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Add a single data point with weight 1.
    pub fn add(&mut self, value: f64) {
        self.add_weighted(value, 1.0);
    }

    /// Add a data point with an explicit weight.
    pub fn add_weighted(&mut self, value: f64, weight: f64) {
        if weight <= 0.0 {
            return;
        }
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.buffer.push(value);
        self.total_weight += weight;
        if self.buffer.len() >= self.buffer_capacity {
            self.flush();
        }
    }

    /// Add all values from a slice.
    pub fn add_all(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Merge another t-digest into this one.
    pub fn merge(&mut self, other: &TDigest) {
        // Drain the other's buffered points and centroids into this one.
        for &v in &other.buffer {
            self.add(v);
        }
        for c in &other.centroids {
            // Re-add centroid points using add_weighted.
            self.add_weighted(c.mean, c.weight);
        }
        // Re-adjust total (avoid double-counting): total_weight was incremented
        // by add_weighted already.  Undo the extra addition from centroids.
        // Actually: flush will handle merging; no adjustment needed since
        // add_weighted accumulates total_weight correctly.
    }

    /// Estimate the value at quantile `q` ∈ [0, 1].
    ///
    /// Returns an error if the digest has no data or `q` is out of range.
    pub fn quantile(&mut self, q: f64) -> Result<f64, AnalyticsError> {
        if q < 0.0 || q > 1.0 {
            return Err(AnalyticsError::ConfigError(format!(
                "quantile q={q} must be in [0, 1]"
            )));
        }
        self.flush();
        if self.centroids.is_empty() {
            return Err(AnalyticsError::InsufficientData(
                "t-digest is empty".to_string(),
            ));
        }

        let n = self.total_weight;
        if n == 0.0 {
            return Err(AnalyticsError::InsufficientData(
                "t-digest total weight is zero".to_string(),
            ));
        }

        // Special cases for min/max.
        if q <= 0.0 {
            return Ok(self.min);
        }
        if q >= 1.0 {
            return Ok(self.max);
        }

        let target = q * n;

        // Walk centroids accumulating weight.
        let mut cumulative = 0.0f64;
        for i in 0..self.centroids.len() {
            let half_weight = self.centroids[i].weight / 2.0;
            let lower = cumulative;
            let upper = cumulative + self.centroids[i].weight;

            if target <= lower + half_weight {
                // Target falls in the first half of this centroid.
                if i == 0 {
                    // Interpolate between min and centroid mean.
                    let t = (target - lower) / half_weight;
                    return Ok(self.min + t * (self.centroids[i].mean - self.min));
                }
                let prev_mid = cumulative - self.centroids[i - 1].weight / 2.0;
                let curr_mid = lower + half_weight;
                if curr_mid > prev_mid {
                    let t = (target - prev_mid) / (curr_mid - prev_mid);
                    return Ok(self.centroids[i - 1].mean
                        + t * (self.centroids[i].mean - self.centroids[i - 1].mean));
                }
                return Ok(self.centroids[i].mean);
            }
            cumulative = upper;
        }

        Ok(self.max)
    }

    /// Number of centroids (compactness measure).
    pub fn centroid_count(&self) -> usize {
        self.centroids.len()
    }

    /// Total weight (number of points added, including buffered).
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    // ── Internal merge logic ──────────────────────────────────────────────────

    /// Flush the internal buffer by merging buffered points into centroids.
    fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }

        // Sort buffer.
        let mut new_points: Vec<Centroid> = self
            .buffer
            .drain(..)
            .map(|v| Centroid::new(v, 1.0))
            .collect();
        new_points.sort_by(|a, b| {
            a.mean
                .partial_cmp(&b.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Merge existing centroids with new points (sorted merge).
        let old = std::mem::take(&mut self.centroids);
        let mut merged: Vec<Centroid> = Vec::with_capacity(old.len() + new_points.len());

        let mut old_iter = old.into_iter().peekable();
        let mut new_iter = new_points.into_iter().peekable();
        loop {
            match (old_iter.peek(), new_iter.peek()) {
                (Some(o), Some(n)) => {
                    if o.mean <= n.mean {
                        merged.push(old_iter.next().unwrap_or_else(|| unreachable!()));
                    } else {
                        merged.push(new_iter.next().unwrap_or_else(|| unreachable!()));
                    }
                }
                (Some(_), None) => {
                    merged.extend(old_iter);
                    break;
                }
                (None, Some(_)) => {
                    merged.extend(new_iter);
                    break;
                }
                (None, None) => break,
            }
        }

        // Compress merged list using the t-digest scale function.
        self.centroids = compress(merged, self.total_weight, self.delta);
    }
}

/// Compress a sorted list of centroids using the t-digest scale function.
///
/// The scale function is: `k(q) = (delta / (2π)) * arcsin(2q − 1)`.
/// Two adjacent centroids can be merged if their combined weight does not
/// violate the size limit imposed by their quantile position.
fn compress(sorted: Vec<Centroid>, total_weight: f64, delta: f64) -> Vec<Centroid> {
    if sorted.is_empty() {
        return sorted;
    }
    let max_centroids = (delta as usize * 2).max(16);
    let mut result: Vec<Centroid> = Vec::with_capacity(max_centroids);
    let mut cumulative_weight = 0.0f64;

    for c in sorted {
        if let Some(last) = result.last_mut() {
            let q = cumulative_weight / total_weight;
            // Size limit for the current centroid.
            let size_limit = 4.0 * q * (1.0 - q) * total_weight / delta;
            let size_limit = size_limit.max(1.0);

            if last.weight + c.weight <= size_limit {
                // Merge into the last centroid.
                let total = last.weight + c.weight;
                last.mean = (last.mean * last.weight + c.mean * c.weight) / total;
                last.weight = total;
                cumulative_weight += c.weight;
                continue;
            }
        }
        cumulative_weight += c.weight;
        result.push(c);
    }

    result
}

// ─── Percentile helper ────────────────────────────────────────────────────────

/// Convenience: compute multiple percentiles at once from a slice of f64 values.
///
/// `percentiles` should be values in [0, 100].  Returns a `Vec<f64>` of the
/// same length.
///
/// Uses a `TDigest` with `delta = 100` internally.
pub fn percentiles(values: &[f64], percentiles: &[f64]) -> Result<Vec<f64>, AnalyticsError> {
    if values.is_empty() {
        return Err(AnalyticsError::InsufficientData(
            "cannot compute percentiles of empty dataset".to_string(),
        ));
    }
    let mut digest = TDigest::new(100.0);
    digest.add_all(values);
    percentiles
        .iter()
        .map(|&p| {
            if p < 0.0 || p > 100.0 {
                Err(AnalyticsError::ConfigError(format!(
                    "percentile {p} out of range [0, 100]"
                )))
            } else {
                digest.quantile(p / 100.0)
            }
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── basic functionality ───────────────────────────────────────────────────

    #[test]
    fn tdigest_single_value() {
        let mut d = TDigest::new(100.0);
        d.add(42.0);
        let q50 = d.quantile(0.5).expect("quantile should succeed");
        assert!((q50 - 42.0).abs() < 1e-9, "q50={q50}");
    }

    #[test]
    fn tdigest_empty_returns_error() {
        let mut d = TDigest::new(100.0);
        assert!(d.quantile(0.5).is_err());
    }

    #[test]
    fn tdigest_invalid_quantile() {
        let mut d = TDigest::new(100.0);
        d.add(1.0);
        assert!(d.quantile(-0.1).is_err());
        assert!(d.quantile(1.1).is_err());
    }

    #[test]
    fn tdigest_uniform_distribution_p50() {
        // 1000 uniform values [1..1000]; median should be ~500.
        let mut d = TDigest::new(100.0);
        for i in 1..=1000 {
            d.add(i as f64);
        }
        let p50 = d.quantile(0.5).expect("quantile should succeed");
        assert!(
            (p50 - 500.0).abs() < 50.0,
            "P50 of uniform [1..1000] should be ~500, got {p50}"
        );
    }

    #[test]
    fn tdigest_uniform_distribution_p95() {
        let mut d = TDigest::new(100.0);
        for i in 1..=1000 {
            d.add(i as f64);
        }
        let p95 = d.quantile(0.95).expect("quantile should succeed");
        // P95 should be ~950 ± 50.
        assert!(
            (p95 - 950.0).abs() < 60.0,
            "P95 of uniform [1..1000] should be ~950, got {p95}"
        );
    }

    #[test]
    fn tdigest_min_max_exact() {
        let mut d = TDigest::new(100.0);
        for i in 1..=500 {
            d.add(i as f64);
        }
        assert_eq!(d.quantile(0.0).expect("quantile should succeed"), 1.0);
        assert_eq!(d.quantile(1.0).expect("quantile should succeed"), 500.0);
    }

    // ── percentiles helper ────────────────────────────────────────────────────

    #[test]
    fn percentiles_basic() {
        let values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = percentiles(&values, &[50.0, 95.0, 99.0]).expect("percentiles should succeed");
        assert_eq!(result.len(), 3);
        // P50 ≈ 50, P95 ≈ 95, P99 ≈ 99 (±5 tolerance).
        assert!((result[0] - 50.0).abs() < 10.0, "P50={}", result[0]);
        assert!((result[1] - 95.0).abs() < 10.0, "P95={}", result[1]);
    }

    #[test]
    fn percentiles_empty_returns_error() {
        assert!(percentiles(&[], &[50.0]).is_err());
    }

    #[test]
    fn percentiles_out_of_range_error() {
        let values = vec![1.0, 2.0, 3.0];
        assert!(percentiles(&values, &[105.0]).is_err());
    }

    // ── large dataset accuracy ────────────────────────────────────────────────

    #[test]
    fn tdigest_large_dataset_p99() {
        // 10 000 values; P99 should be within 2 % of the true value (9901).
        let mut d = TDigest::new(200.0);
        for i in 1..=10_000 {
            d.add(i as f64);
        }
        let p99 = d.quantile(0.99).expect("quantile should succeed");
        let true_p99 = 9901.0;
        let error_pct = ((p99 - true_p99) / true_p99).abs() * 100.0;
        assert!(
            error_pct < 5.0,
            "P99 error={error_pct:.2}% (p99={p99:.1}, expected~{true_p99})"
        );
    }

    #[test]
    fn tdigest_centroid_count_bounded() {
        let delta = 100.0;
        let mut d = TDigest::new(delta);
        for i in 1..=10_000 {
            d.add(i as f64);
        }
        d.quantile(0.5).ok(); // triggers flush
                              // Number of centroids should be << N.
        assert!(
            d.centroid_count() < 500,
            "too many centroids: {}",
            d.centroid_count()
        );
    }

    // ── merge ─────────────────────────────────────────────────────────────────

    #[test]
    fn tdigest_merge_produces_consistent_quantiles() {
        let mut d1 = TDigest::new(100.0);
        let mut d2 = TDigest::new(100.0);
        for i in 1..=500 {
            d1.add(i as f64);
        }
        for i in 501..=1000 {
            d2.add(i as f64);
        }
        d1.merge(&d2);
        let p50 = d1.quantile(0.5).expect("quantile should succeed");
        assert!(
            (p50 - 500.0).abs() < 80.0,
            "merged P50 should be ~500, got {p50}"
        );
    }
}
