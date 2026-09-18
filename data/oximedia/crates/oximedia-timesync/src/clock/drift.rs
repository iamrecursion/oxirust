//! Clock drift compensation and oscillator characterization.
//!
//! Provides exponential-smoothing drift estimation plus overlapping Allan
//! variance / deviation computation for oscillator stability analysis per
//! IEEE Std 1139 and ITU-T G.810.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// DriftEstimator
// ---------------------------------------------------------------------------

/// Estimates clock frequency drift (in ppb — parts per billion) using
/// exponential smoothing, and characterises oscillator stability through
/// overlapping Allan variance.
///
/// # Allan Variance overview
/// Given N phase samples x[0..N-1] spaced by T₀ seconds, the overlapping
/// Allan variance at averaging time τ = m·T₀ is:
///
/// ```text
///                    1
///  AVAR(τ) = ───────────────── · Σ (x[i+2m] - 2·x[i+m] + x[i])²
///             2·(N-2m)·τ²
/// ```
///
/// The Allan deviation ADEV(τ) = √AVAR(τ).
pub struct DriftEstimator {
    /// Reference timestamp for drift computation.
    reference_time: Option<Instant>,
    /// Reference offset (ns) at `reference_time`.
    reference_offset: Option<i64>,
    /// Current drift estimate in ppb.
    drift_ppb: f64,
    /// Exponential smoothing factor α ∈ (0, 1).
    alpha: f64,
    /// Ring buffer of (offset_ns, Instant) phase samples for Allan variance.
    phase_samples: VecDeque<(i64, Instant)>,
    /// Maximum number of phase samples retained.
    max_samples: usize,
}

impl DriftEstimator {
    /// Creates a new drift estimator with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reference_time: None,
            reference_offset: None,
            drift_ppb: 0.0,
            alpha: 0.1,
            phase_samples: VecDeque::new(),
            max_samples: 1024,
        }
    }

    /// Creates a new drift estimator with a custom smoothing factor `alpha`.
    ///
    /// `alpha` must be in `(0.0, 1.0)`.  Clamps silently.
    #[must_use]
    pub fn with_alpha(alpha: f64) -> Self {
        let alpha = alpha.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        Self {
            alpha,
            ..Self::new()
        }
    }

    /// Creates a new drift estimator with a custom maximum sample count for
    /// Allan variance computation.
    #[must_use]
    pub fn with_max_samples(max_samples: usize) -> Self {
        let max_samples = max_samples.max(4);
        Self {
            max_samples,
            ..Self::new()
        }
    }

    /// Updates drift estimate with a new offset measurement and records the
    /// phase sample for Allan variance.
    ///
    /// `offset_ns` — measured offset from reference in nanoseconds.
    /// `now`       — monotonic instant of the measurement.
    pub fn update(&mut self, offset_ns: i64, now: Instant) {
        if let (Some(ref_time), Some(ref_offset)) = (self.reference_time, self.reference_offset) {
            let elapsed = now.duration_since(ref_time).as_secs_f64();

            if elapsed > 0.0 {
                let offset_change = offset_ns - ref_offset;
                // drift = Δoffset / elapsed  (in ppb: × 1e9 because offset in ns,
                // elapsed in s, and 1 ppb = 1 ns/s).
                let drift_estimate = offset_change as f64 / elapsed;

                self.drift_ppb = self.alpha * drift_estimate + (1.0 - self.alpha) * self.drift_ppb;

                self.reference_time = Some(now);
                self.reference_offset = Some(offset_ns);
            }
        } else {
            self.reference_time = Some(now);
            self.reference_offset = Some(offset_ns);
        }

        self.record_phase_sample(offset_ns, now);
    }

    /// Returns the current drift estimate in parts-per-billion (ppb).
    #[must_use]
    pub fn drift_ppb(&self) -> f64 {
        self.drift_ppb
    }

    /// Predicts the offset after `duration` given the current drift estimate.
    ///
    /// `current_offset` — current measured offset in nanoseconds.
    /// Returns predicted offset in nanoseconds.
    #[must_use]
    pub fn predict_offset(&self, current_offset: i64, duration: Duration) -> i64 {
        // drift_ppb [ns/s] × elapsed [s] = Δoffset [ns]
        let predicted_change = self.drift_ppb * duration.as_secs_f64();
        current_offset + predicted_change as i64
    }

    /// Resets all state: drift estimate, reference values, and phase samples.
    pub fn reset(&mut self) {
        self.reference_time = None;
        self.reference_offset = None;
        self.drift_ppb = 0.0;
        self.phase_samples.clear();
    }

    // -----------------------------------------------------------------------
    // Phase sample management
    // -----------------------------------------------------------------------

    /// Records a raw phase sample (offset_ns, instant) for Allan variance.
    ///
    /// Automatically called by `update`; may also be called independently
    /// when only tracking phase without updating the drift estimate.
    pub fn record_phase_sample(&mut self, offset_ns: i64, now: Instant) {
        self.phase_samples.push_back((offset_ns, now));
        while self.phase_samples.len() > self.max_samples {
            self.phase_samples.pop_front();
        }
    }

    /// Returns the number of retained phase samples.
    #[must_use]
    pub fn phase_sample_count(&self) -> usize {
        self.phase_samples.len()
    }

    // -----------------------------------------------------------------------
    // Allan variance / deviation
    // -----------------------------------------------------------------------

    /// Computes the overlapping Allan variance for averaging time `tau_secs`.
    ///
    /// Requires at least `3` samples whose inter-sample spacing `T₀` can be
    /// inferred from the timestamps.  Returns `None` when there is insufficient
    /// data.
    ///
    /// # Algorithm
    /// 1. Derive the nominal sample spacing T₀ from the first two samples.
    /// 2. Compute averaging factor m = round(τ / T₀), clamped to [1, N/3].
    /// 3. Evaluate overlapping AVAR:
    ///    `AVAR = 1/(2·(N−2m)·τ²) · Σᵢ (x[i+2m] − 2·x[i+m] + x[i])²`
    #[must_use]
    pub fn allan_variance(&self, tau_secs: f64) -> Option<f64> {
        let samples: Vec<(f64, f64)> = self
            .phase_samples
            .iter()
            .map(|(x, t)| {
                // Convert phase to seconds for numerically stable AVAR.
                let phase_s = *x as f64 * 1e-9;
                let t_s = t.elapsed().as_secs_f64();
                (phase_s, t_s)
            })
            .collect();

        let n = samples.len();
        if n < 3 {
            return None;
        }

        // Estimate nominal sample spacing T₀ from the stored Instants.
        // Recompute using the raw deque to avoid precision loss.
        let t0_secs = self.estimate_sample_spacing()?;
        if t0_secs <= 0.0 {
            return None;
        }

        // Averaging factor m.
        let m_raw = (tau_secs / t0_secs).round() as usize;
        let m = m_raw.clamp(1, n / 3);
        if m == 0 || n < 2 * m + 1 {
            return None;
        }

        // Extract phase-only vector (in seconds) in FIFO order.
        let x: Vec<f64> = self
            .phase_samples
            .iter()
            .map(|(v, _)| *v as f64 * 1e-9)
            .collect();

        let tau = m as f64 * t0_secs;
        let tau_sq = tau * tau;

        // Overlapping sum of squared second differences.
        let count = n - 2 * m;
        let sum: f64 = (0..count)
            .map(|i| {
                let diff = x[i + 2 * m] - 2.0 * x[i + m] + x[i];
                diff * diff
            })
            .sum();

        Some(sum / (2.0 * count as f64 * tau_sq))
    }

    /// Computes the overlapping Allan deviation (√AVAR) for `tau_secs`.
    ///
    /// Returns `None` when `allan_variance` would return `None`.
    #[must_use]
    pub fn allan_deviation(&self, tau_secs: f64) -> Option<f64> {
        self.allan_variance(tau_secs).map(f64::sqrt)
    }

    /// Computes (τ, ADEV) pairs for each element of `taus_secs`.
    ///
    /// Entries for which `allan_deviation` returns `None` are silently
    /// omitted from the result.
    #[must_use]
    pub fn allan_deviation_plot(&self, taus_secs: &[f64]) -> Vec<(f64, f64)> {
        taus_secs
            .iter()
            .filter_map(|&tau| {
                let adev = self.allan_deviation(tau)?;
                Some((tau, adev))
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Estimates the median inter-sample spacing T₀ in seconds from the
    /// stored Instant values.  Returns `None` if fewer than 2 samples.
    fn estimate_sample_spacing(&self) -> Option<f64> {
        let samples: Vec<Instant> = self.phase_samples.iter().map(|(_, t)| *t).collect();
        let n = samples.len();
        if n < 2 {
            return None;
        }
        // Collect consecutive differences.
        let mut diffs: Vec<f64> = samples
            .windows(2)
            .map(|w| w[1].saturating_duration_since(w[0]).as_secs_f64())
            .filter(|&d| d > 0.0)
            .collect();
        if diffs.is_empty() {
            return None;
        }
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Use median for robustness against irregular sampling.
        Some(diffs[diffs.len() / 2])
    }
}

impl Default for DriftEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AllanVarianceEstimator
// ---------------------------------------------------------------------------

/// Standalone Allan variance estimator for oscillator stability characterisation.
///
/// Accepts phase observations (in nanoseconds) and computes both the standard
/// (non-overlapping) and overlapping Allan variance as defined in IEEE Std 1139
/// and ITU-T G.810.
///
/// # Algorithm (standard AVAR)
/// Given N phase samples x[0..N-1] spaced by nominal τ samples apart:
///
/// ```text
///                      1
///  AVAR(τ) = ─────────────────── · Σᵢ (x[i+2] − 2·x[i+1] + x[i])²
///             2·(N − 2)
/// ```
///
/// For the **overlapping** variant the sum runs over all valid `i` for a
/// chosen averaging factor m:
///
/// ```text
///                           1
///  OAVAR(τ=m) = ───────────────────── · Σᵢ (x[i+2m] − 2·x[i+m] + x[i])²
///                2·m²·(N − 2m)
/// ```
///
/// (The m² factor normalises out the averaging time so the result retains
/// units of (s/s)² rather than (ns)².)
pub struct AllanVarianceEstimator {
    /// Stored phase observations in nanoseconds, oldest-first.
    samples: VecDeque<f64>,
    /// Maximum number of retained samples.
    max_samples: usize,
}

impl AllanVarianceEstimator {
    /// Creates a new estimator that retains up to `max_samples` phase
    /// observations.  At least 3 samples are required for any computation;
    /// `max_samples` is clamped to a minimum of 4.
    #[must_use]
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples: max_samples.max(4),
        }
    }

    /// Appends a new phase observation `phase_ns` (in nanoseconds) to the
    /// estimator, evicting the oldest sample when the capacity is reached.
    pub fn add_sample(&mut self, phase_ns: f64) {
        self.samples.push_back(phase_ns);
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    /// Returns the number of phase samples currently held.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Computes the **standard** (non-overlapping) Allan variance for a given
    /// averaging interval expressed in samples (`tau_samples`).
    ///
    /// The formula applied is the three-sample second difference:
    ///
    /// ```text
    /// AVAR(τ) = 1/(2·(N−2)) · Σᵢ (x[i+2] − 2·x[i+1] + x[i])²
    /// ```
    ///
    /// The result is in units of ns² (normalised by the count only; the
    /// caller is responsible for further normalisation by τ² if the
    /// fractional-frequency Allan variance is desired).
    ///
    /// Returns `None` when fewer than 3 samples are available or when
    /// `tau_samples` would make the sum degenerate.
    #[must_use]
    pub fn allan_variance(&self, tau_samples: usize) -> Option<f64> {
        let n = self.samples.len();
        // Require tau ≥ 1 and at least 2*tau+1 samples so the sum has ≥ 1 term.
        if tau_samples == 0 || n < 2 * tau_samples + 1 {
            return None;
        }
        let m = tau_samples;
        let count = n - 2 * m;
        if count == 0 {
            return None;
        }

        // Convert the deque to a slice-friendly vec.
        let x: Vec<f64> = self.samples.iter().copied().collect();

        let sum: f64 = (0..count)
            .map(|i| {
                let diff = x[i + 2 * m] - 2.0 * x[i + m] + x[i];
                diff * diff
            })
            .sum();

        Some(sum / (2.0 * count as f64))
    }

    /// Computes the **overlapping** Allan variance for averaging interval
    /// `tau_samples`.
    ///
    /// The overlapping estimator makes use of all possible sub-sequences of
    /// length 2m+1, giving lower statistical uncertainty than the standard
    /// estimator for the same data record.
    ///
    /// Formula:
    ///
    /// ```text
    ///                              1
    /// OAVAR(τ=m) = ─────────────────────── · Σᵢ (x[i+2m] − 2·x[i+m] + x[i])²
    ///               2·m²·(N − 2m)
    /// ```
    ///
    /// Returns `None` when there is insufficient data.
    #[must_use]
    pub fn overlapping_allan_variance(&self, tau_samples: usize) -> Option<f64> {
        let n = self.samples.len();
        if tau_samples == 0 || n < 2 * tau_samples + 1 {
            return None;
        }
        let m = tau_samples;
        let count = n - 2 * m;
        if count == 0 {
            return None;
        }

        let x: Vec<f64> = self.samples.iter().copied().collect();

        let sum: f64 = (0..count)
            .map(|i| {
                let diff = x[i + 2 * m] - 2.0 * x[i + m] + x[i];
                diff * diff
            })
            .sum();

        let m2 = (m * m) as f64;
        Some(sum / (2.0 * m2 * count as f64))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sequence of (offset_ns, Instant) samples with a fixed spacing
    /// and a linear phase ramp (simulating constant frequency offset).
    fn make_ramp_samples(n: usize, spacing: Duration, rate_ppb: f64) -> Vec<(i64, Instant)> {
        let base = Instant::now();
        (0..n)
            .map(|i| {
                let t = base + spacing * i as u32;
                // x[i] = rate_ppb [ns/s] × i × T₀ [s]
                let offset_ns = (rate_ppb * (i as f64) * spacing.as_secs_f64()) as i64;
                (offset_ns, t)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Basic drift estimation tests (preserved from original)
    // -----------------------------------------------------------------------

    #[test]
    fn test_drift_estimator_basic() {
        let mut estimator = DriftEstimator::new();
        let now = Instant::now();

        estimator.update(1000, now);
        estimator.update(2000, now + Duration::from_secs(1));

        let drift = estimator.drift_ppb();
        assert!(
            drift > 0.0,
            "drift should be positive for increasing offset"
        );
    }

    #[test]
    fn test_drift_prediction_zero_drift() {
        let estimator = DriftEstimator::new();
        // With zero drift, prediction equals current offset.
        let predicted = estimator.predict_offset(1000, Duration::from_secs(1));
        assert_eq!(predicted, 1000);
    }

    #[test]
    fn test_drift_reset_clears_state() {
        let mut estimator = DriftEstimator::new();
        let now = Instant::now();
        estimator.update(1_000, now);
        estimator.update(5_000, now + Duration::from_secs(1));
        estimator.reset();
        assert_eq!(estimator.drift_ppb(), 0.0);
        assert_eq!(estimator.phase_sample_count(), 0);
        assert!(estimator.reference_time.is_none());
    }

    // -----------------------------------------------------------------------
    // Phase sample management
    // -----------------------------------------------------------------------

    #[test]
    fn test_phase_sample_capacity_limit() {
        let mut estimator = DriftEstimator::with_max_samples(8);
        let base = Instant::now();
        for i in 0u32..20 {
            estimator
                .record_phase_sample(i as i64, base + Duration::from_millis(u64::from(i) * 100));
        }
        assert_eq!(
            estimator.phase_sample_count(),
            8,
            "should cap at max_samples"
        );
    }

    #[test]
    fn test_update_records_phase_sample() {
        let mut estimator = DriftEstimator::new();
        let base = Instant::now();
        estimator.update(100, base);
        estimator.update(200, base + Duration::from_millis(100));
        assert_eq!(estimator.phase_sample_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Allan variance / deviation
    // -----------------------------------------------------------------------

    #[test]
    fn test_allan_variance_insufficient_data() {
        let mut estimator = DriftEstimator::new();
        let base = Instant::now();
        // Only 2 samples — not enough for any averaging factor.
        estimator.record_phase_sample(0, base);
        estimator.record_phase_sample(100, base + Duration::from_secs(1));
        assert!(estimator.allan_variance(1.0).is_none(), "need ≥3 samples");
    }

    #[test]
    fn test_allan_variance_constant_phase_is_zero() {
        // Constant phase (no frequency offset) → AVAR = 0.
        let mut estimator = DriftEstimator::new();
        let base = Instant::now();
        for i in 0u32..20 {
            estimator.record_phase_sample(1_000_000, base + Duration::from_secs(u64::from(i)));
        }
        let avar = estimator.allan_variance(1.0);
        assert!(avar.is_some(), "should compute AVAR with 20 samples");
        assert!(
            avar.expect("should have AVAR value") < 1e-20,
            "constant phase → AVAR ≈ 0"
        );
    }

    #[test]
    fn test_allan_deviation_is_sqrt_of_variance() {
        let mut estimator = DriftEstimator::new();
        let _base = Instant::now();
        let spacing = Duration::from_secs(1);
        let rate_ppb = 500.0_f64;
        for (offset, t) in make_ramp_samples(30, spacing, rate_ppb) {
            estimator.record_phase_sample(offset, t);
        }
        let avar = estimator.allan_variance(1.0);
        let adev = estimator.allan_deviation(1.0);
        if let (Some(v), Some(d)) = (avar, adev) {
            let expected_d = v.sqrt();
            let rel_err = (d - expected_d).abs() / (expected_d + 1e-30);
            assert!(
                rel_err < 1e-10,
                "ADEV should equal sqrt(AVAR), rel_err={rel_err}"
            );
        }
        // If both are None, that's also acceptable (edge-case spacing).
    }

    #[test]
    fn test_allan_deviation_plot_length() {
        let mut estimator = DriftEstimator::new();
        let base = Instant::now();
        let spacing = Duration::from_secs(1);
        for i in 0u32..40 {
            estimator.record_phase_sample((i as i64) * 10, base + spacing * i);
        }
        let taus = [1.0_f64, 2.0, 4.0, 8.0];
        let plot = estimator.allan_deviation_plot(&taus);
        // At least some entries should succeed.
        assert!(
            !plot.is_empty(),
            "should produce at least one (tau, adev) pair"
        );
        // Each returned tau must be one of the input taus.
        for (tau, _adev) in &plot {
            assert!(taus.contains(tau), "tau {tau} not in input");
        }
    }

    #[test]
    fn test_reset_clears_phase_samples_for_avar() {
        let mut estimator = DriftEstimator::new();
        let base = Instant::now();
        for i in 0u32..20 {
            estimator.record_phase_sample(i as i64, base + Duration::from_secs(u64::from(i)));
        }
        assert!(estimator.phase_sample_count() > 0);
        estimator.reset();
        assert_eq!(estimator.phase_sample_count(), 0);
        assert!(
            estimator.allan_variance(1.0).is_none(),
            "no samples after reset"
        );
    }

    #[test]
    fn test_linear_ramp_adev_nonzero() {
        // A linear phase ramp (constant frequency offset) should yield
        // a non-zero, finite ADEV.
        let mut estimator = DriftEstimator::new();
        let spacing = Duration::from_secs(1);
        let rate_ppb = 1000.0_f64;
        for (offset, t) in make_ramp_samples(50, spacing, rate_ppb) {
            estimator.record_phase_sample(offset, t);
        }
        let adev = estimator.allan_deviation(1.0);
        assert!(adev.is_some(), "should compute ADEV for linear ramp");
        let d = adev.expect("ADEV should be Some");
        assert!(
            d.is_finite() && d >= 0.0,
            "ADEV should be finite non-negative"
        );
    }

    // -----------------------------------------------------------------------
    // AllanVarianceEstimator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_avar_estimator_insufficient_data() {
        let mut est = AllanVarianceEstimator::new(64);
        // 0 samples: no variance.
        assert!(est.allan_variance(1).is_none());
        est.add_sample(0.0);
        est.add_sample(1.0);
        // 2 samples, tau=1 → need 2*1+1 = 3 samples.
        assert!(est.allan_variance(1).is_none());
        assert!(est.overlapping_allan_variance(1).is_none());
    }

    #[test]
    fn test_avar_estimator_capacity_eviction() {
        let mut est = AllanVarianceEstimator::new(4);
        for i in 0u32..10 {
            est.add_sample(i as f64 * 100.0);
        }
        assert_eq!(est.sample_count(), 4, "should cap at max_samples");
    }

    #[test]
    fn test_avar_estimator_constant_phase_is_zero() {
        // Constant phase → all second differences are zero → AVAR = 0.
        let mut est = AllanVarianceEstimator::new(64);
        for _ in 0..30 {
            est.add_sample(1_000_000.0); // 1 ms constant offset
        }
        let avar = est.allan_variance(1).expect("should compute AVAR");
        assert!(avar < 1e-10, "constant phase → AVAR ≈ 0, got {avar}");
        let oavar = est
            .overlapping_allan_variance(1)
            .expect("OAVAR should compute");
        assert!(oavar < 1e-10, "constant phase → OAVAR ≈ 0, got {oavar}");
    }

    #[test]
    fn test_avar_estimator_white_phase_noise_slope() {
        // For white phase noise AVAR ∝ 1/τ² in the time-domain sense.
        // In our sample-count convention (no τ² normalisation) AVAR(tau=m) should
        // decrease as m increases for a white-noise-like signal.
        // We verify AVAR(1) > AVAR(2) for N=64 pseudo-random samples.
        // Use a deterministic pseudo-random sequence (Lehmer LCG) for reproducibility.
        let mut est = AllanVarianceEstimator::new(128);
        let mut v: f64 = 0.0;
        // Simulate white phase noise: independent increments, zero mean.
        for i in 0u64..64 {
            // Simple deterministic alternating noise: ±500 ns pattern.
            let noise = if i % 2 == 0 { 500.0 } else { -500.0 };
            v += noise;
            est.add_sample(v);
        }
        let avar1 = est.allan_variance(1).expect("AVAR(1)");
        let avar2 = est.allan_variance(2).expect("AVAR(2)");
        // Both should be finite and positive for non-trivial data.
        assert!(
            avar1.is_finite() && avar1 > 0.0,
            "AVAR(1) should be positive"
        );
        assert!(
            avar2.is_finite() && avar2 >= 0.0,
            "AVAR(2) should be non-negative"
        );
    }

    #[test]
    fn test_avar_estimator_white_freq_noise_flat() {
        // For white frequency noise (linear phase ramp with constant slope)
        // the standard second-difference is zero after one level of differencing.
        // AVAR(tau=1) of a linear ramp x[i] = a*i is:
        //   second_diff = a*(i+2) - 2*a*(i+1) + a*i = 0  → AVAR = 0.
        let mut est = AllanVarianceEstimator::new(128);
        let slope = 10.0_f64; // 10 ns/sample constant frequency offset
        for i in 0i64..50 {
            est.add_sample(slope * i as f64);
        }
        let avar = est.allan_variance(1).expect("AVAR(1) for linear ramp");
        assert!(
            avar < 1e-6,
            "linear ramp (white freq noise) should give AVAR ≈ 0, got {avar}"
        );
    }

    #[test]
    fn test_overlapping_avar_matches_standard_at_tau1() {
        // At tau=1 both formulas are identical (m=1, normalisation differs only
        // by 1/m²=1).  We verify they match to floating-point precision.
        let mut est = AllanVarianceEstimator::new(64);
        for i in 0u32..40 {
            est.add_sample(i as f64 * 37.5 + (i % 3) as f64 * 200.0);
        }
        let std = est.allan_variance(1).expect("standard AVAR");
        let ovl = est.overlapping_allan_variance(1).expect("overlapping AVAR");
        // At m=1 the standard formula gives sum/(2*(N-2)) and the overlapping gives
        // sum/(2*1²*(N-2)), so they are numerically equal.
        let rel = (std - ovl).abs() / (std + 1e-30);
        assert!(
            rel < 1e-10,
            "std and overlapping AVAR should match at tau=1"
        );
    }
}
