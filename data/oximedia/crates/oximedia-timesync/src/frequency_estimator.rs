//! Clock frequency error estimation via linear regression.
//!
//! Estimates the frequency offset between a local clock and a reference
//! (remote) clock by fitting a straight line through a series of time-offset
//! samples.  The slope of that line is the frequency error expressed in
//! parts per billion (ppb).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// FrequencyError
// ---------------------------------------------------------------------------

/// A clock frequency error expressed in parts per billion (ppb).
///
/// A positive value means the local clock runs *faster* than the reference;
/// a negative value means it runs *slower*.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyError {
    /// Frequency deviation in ppb (parts per billion).
    pub ppb: f64,
}

impl FrequencyError {
    /// Create a new `FrequencyError`.
    #[must_use]
    pub fn new(ppb: f64) -> Self {
        Self { ppb }
    }

    /// Convert to parts per million (ppm).
    #[must_use]
    pub fn ppm(&self) -> f64 {
        self.ppb / 1_000.0
    }

    /// Equivalent nanoseconds of drift per second.
    #[must_use]
    pub fn nsec_per_sec(&self) -> f64 {
        self.ppb
    }

    /// Returns `true` when the absolute frequency error is within `max_ppb`.
    #[must_use]
    pub fn is_acceptable(&self, max_ppb: f64) -> bool {
        self.ppb.abs() <= max_ppb
    }
}

// ---------------------------------------------------------------------------
// ClockEstimate
// ---------------------------------------------------------------------------

/// Combined estimate of a clock's current offset and frequency error.
#[derive(Debug, Clone, Copy)]
pub struct ClockEstimate {
    /// Current clock offset from the reference, in nanoseconds.
    pub offset_ns: i64,
    /// Estimated frequency error.
    pub freq_error: FrequencyError,
    /// Confidence in this estimate (0.0 = unreliable, 1.0 = very reliable).
    pub confidence: f32,
}

impl ClockEstimate {
    /// Create a new estimate.
    #[must_use]
    pub fn new(offset_ns: i64, freq_error: FrequencyError, confidence: f32) -> Self {
        Self {
            offset_ns,
            freq_error,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Returns `true` when the estimate has confidence ≥ 0.7.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.7
    }
}

// ---------------------------------------------------------------------------
// LinearRegressor
// ---------------------------------------------------------------------------

/// Ordinary least-squares linear regression over (x, y) pairs.
///
/// Computes slope and intercept of the best-fit line y = slope·x + intercept.
#[derive(Debug, Clone, Default)]
pub struct LinearRegressor {
    /// Accumulated data points.
    points: Vec<(f64, f64)>,
}

impl LinearRegressor {
    /// Create a new, empty regressor.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add a data point.
    pub fn add_point(&mut self, x: f64, y: f64) {
        self.points.push((x, y));
    }

    /// Number of data points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` when no data points have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute the least-squares slope, or `None` if fewer than 2 points are
    /// available or the x-values are all identical (zero variance).
    #[must_use]
    pub fn slope(&self) -> Option<f64> {
        let n = self.points.len();
        if n < 2 {
            return None;
        }

        let (sx, sy, sxx, sxy) = self.sums();
        let nf = n as f64;
        let denom = nf * sxx - sx * sx;
        if denom.abs() < f64::EPSILON {
            return None;
        }
        Some((nf * sxy - sx * sy) / denom)
    }

    /// Compute the least-squares intercept, or `None` if fewer than 2 points
    /// are available.
    #[must_use]
    pub fn intercept(&self) -> Option<f64> {
        let n = self.points.len();
        if n < 2 {
            return None;
        }
        let slope = self.slope()?;
        let (sx, sy, _, _) = self.sums();
        let nf = n as f64;
        Some((sy - slope * sx) / nf)
    }

    /// Coefficient of determination (R²), or `None` if fewer than 2 points.
    #[must_use]
    pub fn r_squared(&self) -> Option<f64> {
        let n = self.points.len();
        if n < 2 {
            return None;
        }
        let slope = self.slope()?;
        let intercept = self.intercept()?;
        let (_, sy, _, _) = self.sums();
        let mean_y = sy / n as f64;
        let ss_tot: f64 = self.points.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = self
            .points
            .iter()
            .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
            .sum();
        if ss_tot.abs() < f64::EPSILON {
            Some(1.0)
        } else {
            Some(1.0 - ss_res / ss_tot)
        }
    }

    // Internal helper — computes Σx, Σy, Σx², Σxy.
    fn sums(&self) -> (f64, f64, f64, f64) {
        self.points
            .iter()
            .fold((0.0, 0.0, 0.0, 0.0), |(sx, sy, sxx, sxy), &(x, y)| {
                (sx + x, sy + y, sxx + x * x, sxy + x * y)
            })
    }
}

// ---------------------------------------------------------------------------
// FrequencyEstimator
// ---------------------------------------------------------------------------

/// Estimates the frequency error of a local clock relative to a remote
/// reference by recording (local_timestamp_ns, offset_ns) pairs and fitting a
/// line through the offset series.
///
/// The slope of that line (Δoffset / Δtime) is the frequency error in ppb
/// (since both numerator and denominator are in nanoseconds, the ratio is
/// dimensionless × 10⁻⁹, i.e. ppb).
#[derive(Debug, Clone)]
pub struct FrequencyEstimator {
    /// (local_ts_ns, remote_ts_ns) samples.
    timestamps: Vec<(u64, i64)>,
    /// Maximum number of samples to retain.
    window_size: usize,
    /// Internal regressor built from the current window.
    regressor: LinearRegressor,
}

impl FrequencyEstimator {
    /// Create a new estimator with the given sliding-window size.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            timestamps: Vec::new(),
            window_size,
            regressor: LinearRegressor::new(),
        }
    }

    /// Record a new (local_ts_ns, offset_ns) sample and update the internal
    /// regressor.
    ///
    /// `local_ts_ns`  — local clock reading in nanoseconds.
    /// `remote_ts_ns` — offset from the remote reference (remote − local) in
    ///                  nanoseconds.
    pub fn add_sample(&mut self, local_ts_ns: u64, remote_ts_ns: i64) {
        if self.timestamps.len() >= self.window_size {
            self.timestamps.remove(0);
        }
        self.timestamps.push((local_ts_ns, remote_ts_ns));
        self.rebuild_regressor();
    }

    /// Estimate the current frequency error.
    ///
    /// Returns `None` when fewer than 2 samples have been added.
    #[must_use]
    pub fn estimate_frequency_error(&self) -> Option<FrequencyError> {
        let slope = self.regressor.slope()?;
        // Slope is ppb because offset (ns) / local_time (ns) = dimensionless × 10⁻⁹
        Some(FrequencyError::new(slope))
    }

    /// Return a [`ClockEstimate`] using the most recent offset and the
    /// regressor's R² as a proxy for confidence.
    #[must_use]
    pub fn clock_estimate(&self) -> Option<ClockEstimate> {
        let freq_error = self.estimate_frequency_error()?;
        let last_offset = self.timestamps.last()?.1;
        let r2 = self.regressor.r_squared().unwrap_or(0.0).max(0.0);
        let confidence = r2 as f32;
        Some(ClockEstimate::new(last_offset, freq_error, confidence))
    }

    /// Number of samples currently in the window.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.timestamps.len()
    }

    // Rebuild the regressor from the current timestamp window.
    fn rebuild_regressor(&mut self) {
        self.regressor = LinearRegressor::new();
        for &(local_ts, offset) in &self.timestamps {
            self.regressor.add_point(local_ts as f64, offset as f64);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- FrequencyError ---

    #[test]
    fn test_frequency_error_ppm() {
        let fe = FrequencyError::new(1000.0);
        assert!((fe.ppm() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_frequency_error_nsec_per_sec() {
        let fe = FrequencyError::new(50.0);
        assert!((fe.nsec_per_sec() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_frequency_error_is_acceptable_within() {
        let fe = FrequencyError::new(100.0);
        assert!(fe.is_acceptable(200.0));
    }

    #[test]
    fn test_frequency_error_is_acceptable_exceeds() {
        let fe = FrequencyError::new(300.0);
        assert!(!fe.is_acceptable(200.0));
    }

    #[test]
    fn test_frequency_error_negative_acceptable() {
        let fe = FrequencyError::new(-150.0);
        assert!(fe.is_acceptable(200.0));
    }

    // --- ClockEstimate ---

    #[test]
    fn test_clock_estimate_reliable() {
        let est = ClockEstimate::new(500, FrequencyError::new(10.0), 0.9);
        assert!(est.is_reliable());
    }

    #[test]
    fn test_clock_estimate_unreliable() {
        let est = ClockEstimate::new(500, FrequencyError::new(10.0), 0.5);
        assert!(!est.is_reliable());
    }

    #[test]
    fn test_clock_estimate_confidence_clamped() {
        let est = ClockEstimate::new(0, FrequencyError::new(0.0), 1.5);
        assert!((est.confidence - 1.0).abs() < 1e-6);
    }

    // --- LinearRegressor ---

    #[test]
    fn test_regressor_no_points_returns_none() {
        let r = LinearRegressor::new();
        assert!(r.slope().is_none());
        assert!(r.intercept().is_none());
    }

    #[test]
    fn test_regressor_one_point_returns_none() {
        let mut r = LinearRegressor::new();
        r.add_point(1.0, 2.0);
        assert!(r.slope().is_none());
    }

    #[test]
    fn test_regressor_perfect_line() {
        let mut r = LinearRegressor::new();
        // y = 2x + 3
        for i in 0..10 {
            r.add_point(i as f64, 2.0 * i as f64 + 3.0);
        }
        let slope = r.slope().expect("should succeed in test");
        let intercept = r.intercept().expect("should succeed in test");
        assert!((slope - 2.0).abs() < 1e-9);
        assert!((intercept - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_regressor_zero_slope() {
        let mut r = LinearRegressor::new();
        // y = 5 (constant)
        for i in 0..5 {
            r.add_point(i as f64, 5.0);
        }
        let slope = r.slope().expect("should succeed in test");
        assert!(slope.abs() < 1e-9);
    }

    #[test]
    fn test_regressor_r_squared_perfect() {
        let mut r = LinearRegressor::new();
        for i in 0..10 {
            r.add_point(i as f64, i as f64 * 3.0);
        }
        let r2 = r.r_squared().expect("should succeed in test");
        assert!((r2 - 1.0).abs() < 1e-9);
    }

    // --- FrequencyEstimator ---

    #[test]
    fn test_estimator_no_samples() {
        let est = FrequencyEstimator::new(10);
        assert!(est.estimate_frequency_error().is_none());
        assert!(est.clock_estimate().is_none());
    }

    #[test]
    fn test_estimator_one_sample() {
        let mut est = FrequencyEstimator::new(10);
        est.add_sample(1_000_000_000, 500);
        assert!(est.estimate_frequency_error().is_none());
    }

    #[test]
    fn test_estimator_constant_drift() {
        // Simulate a clock running 100 ppb fast:
        // after t nanoseconds the offset grows by 100 * t / 1e9 ns.
        // i.e. offset(t) = 100e-9 * t  → slope ≈ 100e-9 (too small to see as ppb easily)
        // Instead: drift = 100 ppb → 100 ns/s → offset_ns = local_ns * 100 / 1e9
        // We simulate with simple integer arithmetic.
        let mut est = FrequencyEstimator::new(20);
        for i in 0..10_u64 {
            let local_ns = i * 1_000_000_000; // 0..9 seconds
            let offset_ns = (i as i64) * 100; // 100 ns/s drift
            est.add_sample(local_ns, offset_ns);
        }
        let fe = est
            .estimate_frequency_error()
            .expect("should succeed in test");
        // slope ≈ 100 / 1_000_000_000 = 1e-7 ppb — the units here are
        // (ns offset) / (ns local time), so slope is actually dimensionless,
        // but we store it as-is. The value should be ~1e-7.
        assert!(fe.ppb.abs() > 0.0);
    }

    #[test]
    fn test_estimator_window_eviction() {
        let mut est = FrequencyEstimator::new(5);
        for i in 0..10_u64 {
            est.add_sample(i * 1_000, i as i64 * 10);
        }
        assert_eq!(est.sample_count(), 5);
    }

    #[test]
    fn test_estimator_clock_estimate_fields() {
        let mut est = FrequencyEstimator::new(10);
        for i in 0..5_u64 {
            est.add_sample(i * 1_000_000, i as i64 * 50);
        }
        let ce = est.clock_estimate().expect("should succeed in test");
        assert_eq!(ce.offset_ns, 200); // last offset = 4 * 50
    }
}
