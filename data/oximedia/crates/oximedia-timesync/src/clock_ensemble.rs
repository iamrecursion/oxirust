#![allow(dead_code)]
//! Clock ensemble for combining multiple clock sources.
//!
//! Implements weighted averaging and outlier rejection across multiple
//! independent clock sources to produce a robust composite time estimate.

use std::collections::HashMap;

/// Unique identifier for a clock source within the ensemble.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClockSourceId(String);

impl ClockSourceId {
    /// Creates a new clock source identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Quality metrics for a single clock source.
#[derive(Debug, Clone, Copy)]
pub struct ClockQuality {
    /// Estimated accuracy in nanoseconds (lower is better).
    pub accuracy_ns: f64,
    /// Stability metric (Allan deviation or similar, lower is better).
    pub stability: f64,
    /// Weight assigned to this clock (0.0 to 1.0).
    pub weight: f64,
    /// Whether this clock is currently reachable.
    pub reachable: bool,
}

impl Default for ClockQuality {
    fn default() -> Self {
        Self {
            accuracy_ns: 1_000_000.0, // 1ms default
            stability: 1.0,
            weight: 1.0,
            reachable: true,
        }
    }
}

impl ClockQuality {
    /// Creates a new quality metric with the given accuracy.
    #[must_use]
    pub fn with_accuracy_ns(accuracy_ns: f64) -> Self {
        Self {
            accuracy_ns,
            ..Default::default()
        }
    }

    /// Sets the stability metric.
    #[must_use]
    pub fn with_stability(mut self, stability: f64) -> Self {
        self.stability = stability;
        self
    }

    /// Sets the weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Marks the clock as unreachable.
    #[must_use]
    pub fn unreachable(mut self) -> Self {
        self.reachable = false;
        self.weight = 0.0;
        self
    }

    /// Returns a composite score (lower is better).
    #[must_use]
    pub fn score(&self) -> f64 {
        if !self.reachable {
            return f64::MAX;
        }
        self.accuracy_ns * self.stability
    }
}

/// A measurement from a single clock source.
#[derive(Debug, Clone, Copy)]
pub struct ClockMeasurement {
    /// Offset from the ensemble mean in nanoseconds.
    pub offset_ns: i64,
    /// Round-trip delay in nanoseconds.
    pub delay_ns: u64,
    /// Measurement timestamp (nanoseconds since epoch).
    pub timestamp_ns: u64,
}

impl ClockMeasurement {
    /// Creates a new clock measurement.
    #[must_use]
    pub fn new(offset_ns: i64, delay_ns: u64, timestamp_ns: u64) -> Self {
        Self {
            offset_ns,
            delay_ns,
            timestamp_ns,
        }
    }
}

/// Outlier rejection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierStrategy {
    /// No outlier rejection.
    None,
    /// Reject samples beyond N standard deviations.
    StdDeviation(u32),
    /// Reject samples beyond N times the median absolute deviation.
    MedianAbsoluteDeviation(u32),
}

/// Result of an ensemble computation.
#[derive(Debug, Clone, Copy)]
pub struct EnsembleResult {
    /// Weighted average offset in nanoseconds.
    pub offset_ns: f64,
    /// Estimated uncertainty in nanoseconds.
    pub uncertainty_ns: f64,
    /// Number of sources that contributed.
    pub contributing_sources: usize,
    /// Number of sources rejected as outliers.
    pub rejected_sources: usize,
}

/// A clock ensemble that combines multiple clock sources.
#[derive(Debug)]
pub struct ClockEnsemble {
    /// Clock sources with their quality metrics.
    sources: HashMap<ClockSourceId, ClockQuality>,
    /// Latest measurements per source.
    measurements: HashMap<ClockSourceId, ClockMeasurement>,
    /// Outlier rejection strategy.
    outlier_strategy: OutlierStrategy,
    /// History of ensemble results.
    history: Vec<EnsembleResult>,
    /// Maximum history length.
    max_history: usize,
}

impl ClockEnsemble {
    /// Creates a new clock ensemble.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            measurements: HashMap::new(),
            outlier_strategy: OutlierStrategy::StdDeviation(3),
            history: Vec::new(),
            max_history: 256,
        }
    }

    /// Sets the outlier rejection strategy.
    #[must_use]
    pub fn with_outlier_strategy(mut self, strategy: OutlierStrategy) -> Self {
        self.outlier_strategy = strategy;
        self
    }

    /// Registers a new clock source.
    pub fn add_source(&mut self, id: ClockSourceId, quality: ClockQuality) {
        self.sources.insert(id, quality);
    }

    /// Removes a clock source.
    pub fn remove_source(&mut self, id: &ClockSourceId) {
        self.sources.remove(id);
        self.measurements.remove(id);
    }

    /// Updates the quality metrics for a source.
    pub fn update_quality(&mut self, id: &ClockSourceId, quality: ClockQuality) {
        if let Some(q) = self.sources.get_mut(id) {
            *q = quality;
        }
    }

    /// Records a measurement from a clock source.
    pub fn record_measurement(&mut self, id: ClockSourceId, measurement: ClockMeasurement) {
        self.measurements.insert(id, measurement);
    }

    /// Returns the number of registered sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Returns the number of sources with recent measurements.
    #[must_use]
    pub fn active_source_count(&self) -> usize {
        self.sources
            .keys()
            .filter(|id| self.measurements.contains_key(id))
            .count()
    }

    /// Returns the quality metrics for a source.
    #[must_use]
    pub fn source_quality(&self, id: &ClockSourceId) -> Option<&ClockQuality> {
        self.sources.get(id)
    }

    /// Computes the ensemble time estimate from all active sources.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&mut self) -> Option<EnsembleResult> {
        let active: Vec<(&ClockSourceId, &ClockMeasurement)> = self
            .measurements
            .iter()
            .filter(|(id, _)| {
                self.sources
                    .get(*id)
                    .map_or(false, |q| q.reachable && q.weight > 0.0)
            })
            .collect();

        if active.is_empty() {
            return None;
        }

        // Collect offsets and weights
        let mut offset_weights: Vec<(f64, f64)> = active
            .iter()
            .map(|(id, m)| {
                let weight = self.sources.get(*id).map_or(1.0, |q| q.weight);
                (m.offset_ns as f64, weight)
            })
            .collect();

        // Apply outlier rejection
        let rejected = self.reject_outliers(&mut offset_weights);

        // Weighted average
        let total_weight: f64 = offset_weights.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return None;
        }

        let weighted_sum: f64 = offset_weights.iter().map(|(o, w)| o * w).sum();
        let mean_offset = weighted_sum / total_weight;

        // Compute uncertainty as weighted standard deviation
        let variance: f64 = offset_weights
            .iter()
            .map(|(o, w)| w * (o - mean_offset).powi(2))
            .sum::<f64>()
            / total_weight;
        let uncertainty = variance.sqrt();

        let result = EnsembleResult {
            offset_ns: mean_offset,
            uncertainty_ns: uncertainty,
            contributing_sources: offset_weights.len(),
            rejected_sources: rejected,
        };

        self.history.push(result);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        Some(result)
    }

    /// Rejects outliers based on the configured strategy.
    ///
    /// Returns the number of rejected entries.
    fn reject_outliers(&self, data: &mut Vec<(f64, f64)>) -> usize {
        match self.outlier_strategy {
            OutlierStrategy::None => 0,
            OutlierStrategy::StdDeviation(n) => self.reject_by_stddev(data, n),
            OutlierStrategy::MedianAbsoluteDeviation(n) => self.reject_by_mad(data, n),
        }
    }

    /// Rejects outliers beyond N standard deviations.
    ///
    /// Uses the median as the centre estimate instead of the mean so that
    /// extreme outliers do not drag the reference point and inflate the
    /// standard deviation, which would prevent their own rejection.
    #[allow(clippy::cast_precision_loss)]
    fn reject_by_stddev(&self, data: &mut Vec<(f64, f64)>, n: u32) -> usize {
        if data.len() < 3 {
            return 0;
        }
        let mut offsets: Vec<f64> = data.iter().map(|(o, _)| *o).collect();
        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = offsets[offsets.len() / 2];
        let variance =
            offsets.iter().map(|o| (o - median).powi(2)).sum::<f64>() / offsets.len() as f64;
        let std_dev = variance.sqrt();
        let threshold = std_dev * f64::from(n);

        let before = data.len();
        data.retain(|(o, _)| (o - median).abs() <= threshold);
        before - data.len()
    }

    /// Rejects outliers beyond N times the median absolute deviation.
    fn reject_by_mad(&self, data: &mut Vec<(f64, f64)>, n: u32) -> usize {
        if data.len() < 3 {
            return 0;
        }
        let mut offsets: Vec<f64> = data.iter().map(|(o, _)| *o).collect();
        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = offsets[offsets.len() / 2];

        let mut abs_devs: Vec<f64> = offsets.iter().map(|o| (o - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = abs_devs[abs_devs.len() / 2];
        let threshold = mad * f64::from(n);

        let before = data.len();
        data.retain(|(o, _)| (o - median).abs() <= threshold);
        before - data.len()
    }

    /// Returns the latest ensemble result.
    #[must_use]
    pub fn latest_result(&self) -> Option<&EnsembleResult> {
        self.history.last()
    }

    /// Returns the history of ensemble results.
    #[must_use]
    pub fn history(&self) -> &[EnsembleResult] {
        &self.history
    }

    /// Resets all measurements (keeps sources registered).
    pub fn reset_measurements(&mut self) {
        self.measurements.clear();
        self.history.clear();
    }
}

impl Default for ClockEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id(name: &str) -> ClockSourceId {
        ClockSourceId::new(name)
    }

    #[test]
    fn test_clock_source_id() {
        let id = make_id("ptp-master");
        assert_eq!(id.as_str(), "ptp-master");
    }

    #[test]
    fn test_clock_quality_default() {
        let q = ClockQuality::default();
        assert!(q.reachable);
        assert!((q.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clock_quality_unreachable() {
        let q = ClockQuality::default().unreachable();
        assert!(!q.reachable);
        assert!((q.weight - 0.0).abs() < f64::EPSILON);
        assert_eq!(q.score(), f64::MAX);
    }

    #[test]
    fn test_clock_quality_score() {
        let q = ClockQuality::with_accuracy_ns(100.0).with_stability(0.5);
        assert!((q.score() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clock_quality_weight_clamped() {
        let q = ClockQuality::default().with_weight(2.0);
        assert!((q.weight - 1.0).abs() < f64::EPSILON);
        let q2 = ClockQuality::default().with_weight(-0.5);
        assert!((q2.weight - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ensemble_add_remove_source() {
        let mut ens = ClockEnsemble::new();
        let id = make_id("s1");
        ens.add_source(id.clone(), ClockQuality::default());
        assert_eq!(ens.source_count(), 1);

        ens.remove_source(&id);
        assert_eq!(ens.source_count(), 0);
    }

    #[test]
    fn test_ensemble_compute_single_source() {
        let mut ens = ClockEnsemble::new().with_outlier_strategy(OutlierStrategy::None);
        let id = make_id("s1");
        ens.add_source(id.clone(), ClockQuality::default());
        ens.record_measurement(id, ClockMeasurement::new(500, 100, 1000));

        let result = ens.compute().expect("should succeed in test");
        assert!((result.offset_ns - 500.0).abs() < 0.01);
        assert_eq!(result.contributing_sources, 1);
    }

    #[test]
    fn test_ensemble_compute_weighted() {
        let mut ens = ClockEnsemble::new().with_outlier_strategy(OutlierStrategy::None);

        let id1 = make_id("s1");
        let id2 = make_id("s2");
        ens.add_source(id1.clone(), ClockQuality::default().with_weight(0.75));
        ens.add_source(id2.clone(), ClockQuality::default().with_weight(0.25));

        ens.record_measurement(id1, ClockMeasurement::new(100, 50, 1000));
        ens.record_measurement(id2, ClockMeasurement::new(200, 60, 1000));

        let result = ens.compute().expect("should succeed in test");
        // Weighted average: (100*0.75 + 200*0.25) / (0.75+0.25) = 125
        assert!((result.offset_ns - 125.0).abs() < 0.01);
    }

    #[test]
    fn test_ensemble_compute_no_active() {
        let mut ens = ClockEnsemble::new();
        assert!(ens.compute().is_none());
    }

    #[test]
    fn test_ensemble_compute_unreachable_excluded() {
        let mut ens = ClockEnsemble::new().with_outlier_strategy(OutlierStrategy::None);
        let id1 = make_id("good");
        let id2 = make_id("bad");
        ens.add_source(id1.clone(), ClockQuality::default());
        ens.add_source(id2.clone(), ClockQuality::default().unreachable());

        ens.record_measurement(id1, ClockMeasurement::new(300, 50, 1000));
        ens.record_measurement(id2, ClockMeasurement::new(9999, 50, 1000));

        let result = ens.compute().expect("should succeed in test");
        assert!((result.offset_ns - 300.0).abs() < 0.01);
        assert_eq!(result.contributing_sources, 1);
    }

    #[test]
    fn test_ensemble_history() {
        let mut ens = ClockEnsemble::new().with_outlier_strategy(OutlierStrategy::None);
        let id = make_id("s1");
        ens.add_source(id.clone(), ClockQuality::default());
        ens.record_measurement(id.clone(), ClockMeasurement::new(100, 50, 1000));
        let _ = ens.compute();
        ens.record_measurement(id, ClockMeasurement::new(200, 50, 2000));
        let _ = ens.compute();

        assert_eq!(ens.history().len(), 2);
        let latest = ens.latest_result().expect("should succeed in test");
        assert!((latest.offset_ns - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_ensemble_reset_measurements() {
        let mut ens = ClockEnsemble::new();
        let id = make_id("s1");
        ens.add_source(id.clone(), ClockQuality::default());
        ens.record_measurement(id, ClockMeasurement::new(100, 50, 1000));
        assert_eq!(ens.active_source_count(), 1);

        ens.reset_measurements();
        assert_eq!(ens.active_source_count(), 0);
        assert_eq!(ens.source_count(), 1); // source still registered
    }

    #[test]
    fn test_outlier_rejection_stddev() {
        let mut ens = ClockEnsemble::new().with_outlier_strategy(OutlierStrategy::StdDeviation(2));

        // Add 5 sources: 4 clustered, 1 outlier
        for i in 0..4 {
            let id = make_id(&format!("s{i}"));
            ens.add_source(id.clone(), ClockQuality::default());
            ens.record_measurement(id, ClockMeasurement::new(100 + i, 50, 1000));
        }
        let outlier_id = make_id("outlier");
        ens.add_source(outlier_id.clone(), ClockQuality::default());
        ens.record_measurement(outlier_id, ClockMeasurement::new(999_999, 50, 1000));

        let result = ens.compute().expect("should succeed in test");
        // Outlier should be rejected, result should be near 100
        assert!(result.offset_ns < 200.0);
    }
}

// ---------------------------------------------------------------------------
// Bayesian clock ensemble
// ---------------------------------------------------------------------------

/// Gaussian prior / posterior state for a single Bayesian clock estimate.
///
/// The Bayesian estimator models each observation as a Gaussian with mean μ
/// and variance σ² drawn from the source's `accuracy_ns`.  The posterior is
/// obtained by combining the prior with each likelihood in closed form
/// (conjugate Gaussian update):
///
/// ```text
/// σ_post² = 1 / (1/σ_prior² + 1/σ_like²)
/// μ_post   = σ_post² × (μ_prior/σ_prior² + μ_like/σ_like²)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GaussianState {
    /// Mean offset estimate (nanoseconds).
    pub mean_ns: f64,
    /// Variance of the estimate (nanoseconds²).
    pub variance_ns2: f64,
}

impl GaussianState {
    /// Creates a state with the given mean and variance.
    #[must_use]
    pub fn new(mean_ns: f64, variance_ns2: f64) -> Self {
        Self {
            mean_ns,
            variance_ns2: variance_ns2.max(f64::MIN_POSITIVE),
        }
    }

    /// Returns the standard deviation (nanoseconds).
    #[must_use]
    pub fn std_dev_ns(&self) -> f64 {
        self.variance_ns2.sqrt()
    }

    /// Performs a Bayesian (conjugate Gaussian) update given a likelihood with
    /// the specified mean and variance.
    ///
    /// Returns the posterior [`GaussianState`].
    #[must_use]
    pub fn update(&self, likelihood_mean_ns: f64, likelihood_variance_ns2: f64) -> Self {
        let like_var = likelihood_variance_ns2.max(f64::MIN_POSITIVE);
        let prior_precision = 1.0 / self.variance_ns2;
        let like_precision = 1.0 / like_var;
        let post_precision = prior_precision + like_precision;
        let post_variance = 1.0 / post_precision;
        let post_mean =
            post_variance * (prior_precision * self.mean_ns + like_precision * likelihood_mean_ns);
        Self {
            mean_ns: post_mean,
            variance_ns2: post_variance,
        }
    }
}

/// Result from a [`BayesianEnsemble`] computation.
#[derive(Debug, Clone, Copy)]
pub struct BayesianEnsembleResult {
    /// Posterior mean offset (nanoseconds).
    pub mean_ns: f64,
    /// Posterior standard deviation (nanoseconds).
    pub std_dev_ns: f64,
    /// Number of sources fused into this estimate.
    pub fused_sources: usize,
}

/// Multi-source Bayesian clock ensemble.
///
/// Maintains a running Gaussian posterior and fuses each new observation via
/// conjugate Gaussian update.  This is equivalent to optimal minimum-variance
/// linear estimation (Kalman filter measurement step) when noise is Gaussian.
///
/// # Usage
/// ```rust,ignore
/// let mut ensemble = BayesianEnsemble::new(0.0, 1e12);  // diffuse prior
/// ensemble.fuse(100.0, 500.0_f64.powi(2));   // PTP, σ=500ns
/// ensemble.fuse(200_000.0, 5e6_f64.powi(2)); // NTP, σ=5ms
/// let result = ensemble.result();
/// ```
#[derive(Debug)]
pub struct BayesianEnsemble {
    /// Current posterior state.
    state: GaussianState,
    /// Number of observations fused so far.
    fused_count: usize,
}

impl BayesianEnsemble {
    /// Creates a new ensemble with a Gaussian prior specified by
    /// `prior_mean_ns` and `prior_variance_ns2`.
    ///
    /// Use a large `prior_variance_ns2` (e.g. `1e18`) for a diffuse
    /// (uninformative) prior.
    #[must_use]
    pub fn new(prior_mean_ns: f64, prior_variance_ns2: f64) -> Self {
        Self {
            state: GaussianState::new(prior_mean_ns, prior_variance_ns2),
            fused_count: 0,
        }
    }

    /// Creates a new ensemble with a diffuse prior centred at zero.
    #[must_use]
    pub fn with_diffuse_prior() -> Self {
        Self::new(0.0, 1e18)
    }

    /// Fuses a new Gaussian observation into the posterior.
    ///
    /// `measurement_ns`       — measured offset in nanoseconds.
    /// `measurement_var_ns2`  — measurement variance in nanoseconds² (= σ²).
    ///
    /// Use `accuracy_ns.powi(2)` when you have a one-sigma accuracy figure.
    pub fn fuse(&mut self, measurement_ns: f64, measurement_var_ns2: f64) {
        self.state = self.state.update(measurement_ns, measurement_var_ns2);
        self.fused_count += 1;
    }

    /// Returns the current posterior estimate.
    #[must_use]
    pub fn result(&self) -> BayesianEnsembleResult {
        BayesianEnsembleResult {
            mean_ns: self.state.mean_ns,
            std_dev_ns: self.state.std_dev_ns(),
            fused_sources: self.fused_count,
        }
    }

    /// Returns the current posterior [`GaussianState`].
    #[must_use]
    pub fn state(&self) -> GaussianState {
        self.state
    }

    /// Resets the posterior to a new prior.
    pub fn reset(&mut self, prior_mean_ns: f64, prior_variance_ns2: f64) {
        self.state = GaussianState::new(prior_mean_ns, prior_variance_ns2);
        self.fused_count = 0;
    }

    /// Convenience: fuse all active measurements from a [`ClockEnsemble`]
    /// using Bayesian estimation.
    ///
    /// Each source's `accuracy_ns` is used as the Gaussian σ.  Unreachable
    /// or zero-accuracy sources are skipped.
    ///
    /// Returns the final [`BayesianEnsembleResult`], or `None` if no sources
    /// were fused.
    #[must_use]
    pub fn from_clock_ensemble(
        ensemble: &ClockEnsemble,
        prior_mean_ns: f64,
        prior_variance_ns2: f64,
    ) -> Option<BayesianEnsembleResult> {
        let mut bay = BayesianEnsemble::new(prior_mean_ns, prior_variance_ns2);

        for (id, measurement) in &ensemble.measurements {
            if let Some(quality) = ensemble.sources.get(id) {
                if !quality.reachable || quality.accuracy_ns <= 0.0 {
                    continue;
                }
                let var = quality.accuracy_ns * quality.accuracy_ns;
                bay.fuse(measurement.offset_ns as f64, var);
            }
        }

        if bay.fused_count == 0 {
            return None;
        }
        Some(bay.result())
    }
}

#[cfg(test)]
mod bayesian_tests {
    use super::*;

    #[test]
    fn test_gaussian_state_update_single_measurement() {
        // With a very diffuse prior, the posterior should be close to the
        // measurement.
        let prior = GaussianState::new(0.0, 1e18);
        let posterior = prior.update(500.0, 100.0_f64.powi(2));
        // posterior mean ≈ 500 ns (measurement dominates diffuse prior).
        assert!(
            (posterior.mean_ns - 500.0).abs() < 1.0,
            "posterior mean should be near measurement: got {}",
            posterior.mean_ns
        );
    }

    #[test]
    fn test_gaussian_state_update_two_equal_measurements() {
        // Two equal measurements with equal variance and a diffuse (uninformative)
        // prior → posterior mean should converge to the measurement value.
        // A finite prior at mean=0 would bias the result toward 0; use a
        // very large prior variance so the prior is essentially non-informative.
        let prior = GaussianState::new(0.0, 1e18);
        let m = 300.0_f64;
        let var = 200.0_f64.powi(2);
        let post1 = prior.update(m, var);
        let post2 = post1.update(m, var);
        assert!(
            (post2.mean_ns - m).abs() < 1.0,
            "two equal measurements → posterior mean ≈ measurement: {}",
            post2.mean_ns
        );
        // Variance should shrink with each update.
        assert!(
            post2.variance_ns2 < prior.variance_ns2,
            "variance must shrink after updates"
        );
    }

    #[test]
    fn test_bayesian_ensemble_diffuse_prior() {
        let mut bay = BayesianEnsemble::with_diffuse_prior();
        // Fuse a very precise measurement.
        bay.fuse(100.0, 50.0_f64.powi(2));
        let r = bay.result();
        assert!((r.mean_ns - 100.0).abs() < 5.0, "mean should be near 100ns");
        assert_eq!(r.fused_sources, 1);
    }

    #[test]
    fn test_bayesian_ensemble_two_sources_weighted_by_precision() {
        // Source A: accurate (σ = 100 ns)
        // Source B: noisy   (σ = 10_000 ns)
        // Posterior should be much closer to A's measurement.
        let mut bay = BayesianEnsemble::with_diffuse_prior();
        bay.fuse(100.0, 100.0_f64.powi(2)); // A
        bay.fuse(200_000.0, 10_000.0_f64.powi(2)); // B (NTP-like)
        let r = bay.result();
        // Result must be closer to 100 than to 200_000.
        assert!(
            (r.mean_ns - 100.0).abs() < (r.mean_ns - 200_000.0).abs(),
            "precise source A should dominate: mean = {}",
            r.mean_ns
        );
        assert_eq!(r.fused_sources, 2);
    }

    #[test]
    fn test_bayesian_ensemble_reset() {
        let mut bay = BayesianEnsemble::with_diffuse_prior();
        bay.fuse(500.0, 100.0_f64.powi(2));
        bay.reset(0.0, 1e18);
        let r = bay.result();
        assert_eq!(r.fused_sources, 0);
        assert!((r.mean_ns - 0.0).abs() < 1.0);
    }

    #[test]
    fn test_from_clock_ensemble_no_sources() {
        let mut ens = ClockEnsemble::new();
        let id = ClockSourceId::new("s1");
        ens.add_source(id.clone(), ClockQuality::default().unreachable());
        ens.record_measurement(id, ClockMeasurement::new(100, 50, 1000));

        let result = BayesianEnsemble::from_clock_ensemble(&ens, 0.0, 1e18);
        assert!(
            result.is_none(),
            "all sources unreachable → no Bayesian result"
        );
    }

    #[test]
    fn test_from_clock_ensemble_single_source() {
        let mut ens = ClockEnsemble::new();
        let id = ClockSourceId::new("ptp");
        let quality = ClockQuality::with_accuracy_ns(500.0);
        ens.add_source(id.clone(), quality);
        ens.record_measurement(id, ClockMeasurement::new(1000, 50, 1000));

        let result = BayesianEnsemble::from_clock_ensemble(&ens, 0.0, 1e18)
            .expect("should produce Bayesian result");
        assert_eq!(result.fused_sources, 1);
        assert!(
            (result.mean_ns - 1000.0).abs() < 10.0,
            "mean should be near 1000ns: got {}",
            result.mean_ns
        );
    }

    #[test]
    fn test_gaussian_std_dev() {
        let state = GaussianState::new(0.0, 10_000.0);
        let sd = state.std_dev_ns();
        assert!((sd - 100.0).abs() < 0.01, "sqrt(10000) = 100");
    }
}
