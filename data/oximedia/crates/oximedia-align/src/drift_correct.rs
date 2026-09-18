//! Clock drift correction using PLL-based techniques.
//!
//! Provides phase-locked loop (PLL) based clock drift correction,
//! drift rate estimation, and sample-level adjustment for A/V synchronization.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

/// PLL (Phase-Locked Loop) state for drift correction
#[derive(Debug, Clone)]
pub struct PllState {
    /// Current phase error (in samples)
    pub phase_error: f64,
    /// Current frequency error (fractional)
    pub freq_error: f64,
    /// Loop filter state
    pub filter_state: f64,
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Current correction (samples per input sample)
    pub correction: f64,
}

impl PllState {
    /// Create a new PLL state with given gains
    #[must_use]
    pub fn new(kp: f64, ki: f64) -> Self {
        Self {
            phase_error: 0.0,
            freq_error: 0.0,
            filter_state: 0.0,
            kp,
            ki,
            correction: 0.0,
        }
    }

    /// Update PLL with a new phase measurement and return correction
    pub fn update(&mut self, measured_phase: f64, expected_phase: f64) -> f64 {
        self.phase_error = measured_phase - expected_phase;
        self.filter_state += self.ki * self.phase_error;
        self.correction = self.kp * self.phase_error + self.filter_state;
        self.freq_error = self.correction;
        self.correction
    }

    /// Reset PLL state
    pub fn reset(&mut self) {
        self.phase_error = 0.0;
        self.freq_error = 0.0;
        self.filter_state = 0.0;
        self.correction = 0.0;
    }

    /// Returns true if PLL is locked (error below threshold)
    #[must_use]
    pub fn is_locked(&self, threshold: f64) -> bool {
        self.phase_error.abs() < threshold && self.freq_error.abs() < threshold * 0.01
    }
}

impl Default for PllState {
    fn default() -> Self {
        Self::new(0.1, 0.001)
    }
}

/// Drift rate estimator using linear regression over a window of samples
#[derive(Debug, Clone)]
pub struct DriftRateEstimator {
    /// Window of (time, offset) measurements
    measurements: Vec<(f64, f64)>,
    /// Maximum number of measurements to keep
    max_window: usize,
    /// Current estimated drift rate (parts per million)
    estimated_ppm: f64,
}

impl DriftRateEstimator {
    /// Create a new drift rate estimator
    #[must_use]
    pub fn new(max_window: usize) -> Self {
        Self {
            measurements: Vec::new(),
            max_window,
            estimated_ppm: 0.0,
        }
    }

    /// Add a new measurement (time in seconds, offset in samples)
    pub fn add_measurement(&mut self, time: f64, offset: f64) {
        self.measurements.push((time, offset));
        if self.measurements.len() > self.max_window {
            self.measurements.remove(0);
        }
        if self.measurements.len() >= 2 {
            self.estimated_ppm = self.compute_drift_ppm();
        }
    }

    /// Compute drift rate in PPM using linear regression
    fn compute_drift_ppm(&self) -> f64 {
        let n = self.measurements.len() as f64;
        let sum_t: f64 = self.measurements.iter().map(|(t, _)| t).sum();
        let sum_o: f64 = self.measurements.iter().map(|(_, o)| o).sum();
        let sum_t2: f64 = self.measurements.iter().map(|(t, _)| t * t).sum();
        let sum_to: f64 = self.measurements.iter().map(|(t, o)| t * o).sum();

        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < f64::EPSILON {
            return 0.0;
        }
        let slope = (n * sum_to - sum_t * sum_o) / denom;
        // Convert slope (samples/second) to PPM assuming 48000 Hz
        slope / 48.0 // slope / (sample_rate / 1_000_000)
    }

    /// Get current drift rate estimate in PPM
    #[must_use]
    pub fn drift_ppm(&self) -> f64 {
        self.estimated_ppm
    }

    /// Get drift rate in samples per second (at given sample rate)
    #[must_use]
    pub fn drift_samples_per_second(&self, sample_rate: u32) -> f64 {
        self.estimated_ppm * f64::from(sample_rate) / 1_000_000.0
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.estimated_ppm = 0.0;
    }

    /// Number of measurements held
    #[must_use]
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Returns true if no measurements
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }
}

/// Sample adjustment result after drift correction
#[derive(Debug, Clone, Copy)]
pub struct SampleAdjustment {
    /// Number of samples to insert (positive) or drop (negative)
    pub delta: i64,
    /// Fractional part of the adjustment
    pub fractional: f64,
    /// Cumulative drift in samples
    pub cumulative_drift: f64,
}

impl SampleAdjustment {
    /// Create a new sample adjustment
    #[must_use]
    pub fn new(delta: i64, fractional: f64, cumulative_drift: f64) -> Self {
        Self {
            delta,
            fractional,
            cumulative_drift,
        }
    }

    /// Returns true if any adjustment is needed
    #[must_use]
    pub fn needs_adjustment(&self) -> bool {
        self.delta != 0
    }
}

/// Drift corrector combining PLL and sample adjustment
#[derive(Debug, Clone)]
pub struct DriftCorrector {
    /// PLL state
    pll: PllState,
    /// Drift rate estimator
    estimator: DriftRateEstimator,
    /// Accumulated fractional sample error
    accumulator: f64,
    /// Total samples processed
    total_samples: u64,
    /// Sample rate
    sample_rate: u32,
}

impl DriftCorrector {
    /// Create a new drift corrector
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            pll: PllState::new(0.05, 0.0005),
            estimator: DriftRateEstimator::new(100),
            accumulator: 0.0,
            total_samples: 0,
            sample_rate,
        }
    }

    /// Process a block of samples and return the required adjustment
    pub fn process_block(&mut self, block_size: usize, measured_offset: f64) -> SampleAdjustment {
        let time = self.total_samples as f64 / f64::from(self.sample_rate);
        self.estimator.add_measurement(time, measured_offset);

        let drift_rate = self.estimator.drift_samples_per_second(self.sample_rate);
        let block_drift = drift_rate * block_size as f64 / f64::from(self.sample_rate);

        self.accumulator += block_drift;
        let delta = self.accumulator.trunc() as i64;
        self.accumulator -= delta as f64;

        self.total_samples += block_size as u64;

        SampleAdjustment::new(delta, self.accumulator, measured_offset + block_drift)
    }

    /// Update PLL with external reference
    pub fn update_pll(&mut self, measured: f64, expected: f64) -> f64 {
        self.pll.update(measured, expected)
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.pll.reset();
        self.estimator.clear();
        self.accumulator = 0.0;
        self.total_samples = 0;
    }

    /// Get current drift estimate in PPM
    #[must_use]
    pub fn drift_ppm(&self) -> f64 {
        self.estimator.drift_ppm()
    }

    /// Check if PLL is locked
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.pll.is_locked(1.0)
    }
}

/// Genlock drift estimator for multi-camera recording synchronisation.
///
/// In multi-camera setups, cameras nominally run at the same frame rate but
/// their internal clocks drift relative to each other.  Genlock drift
/// estimation measures this drift over long durations (hours) by periodically
/// comparing synchronisation markers (e.g. audio cross-correlation offsets,
/// timecode differences, or flash events).
///
/// The estimator fits a linear model to the observed offset-vs-time data,
/// yielding a drift rate in PPM (parts per million) plus a constant offset.
/// It also provides an uncertainty estimate based on the residuals of the fit.
///
/// # Usage
///
/// 1. Create with `GenlockDriftEstimator::new(camera_count)`.
/// 2. Feed periodic offset measurements via `add_observation()`.
/// 3. Query the drift model with `drift_ppm()`, `predicted_offset()`, etc.
#[derive(Debug, Clone)]
pub struct GenlockDriftEstimator {
    /// Number of cameras in the multi-camera rig.
    camera_count: usize,
    /// Per-camera-pair observations: (time_seconds, offset_samples).
    observations: Vec<Vec<(f64, f64)>>,
    /// Maximum observations to keep per pair.
    max_observations: usize,
}

/// Result of a genlock drift analysis for a single camera pair.
#[derive(Debug, Clone)]
pub struct GenlockDriftResult {
    /// Camera index A.
    pub camera_a: usize,
    /// Camera index B.
    pub camera_b: usize,
    /// Estimated drift rate in parts per million.
    pub drift_ppm: f64,
    /// Constant offset at t=0 (in samples).
    pub offset_at_zero: f64,
    /// R-squared goodness of fit (1.0 = perfect linear drift).
    pub r_squared: f64,
    /// Root mean square residual (in samples).
    pub rms_residual: f64,
    /// Number of observations used.
    pub num_observations: usize,
}

impl GenlockDriftEstimator {
    /// Create a new genlock drift estimator for `camera_count` cameras.
    ///
    /// This creates storage for all `C(camera_count, 2)` unique pairs.
    #[must_use]
    pub fn new(camera_count: usize, max_observations: usize) -> Self {
        let num_pairs = if camera_count >= 2 {
            camera_count * (camera_count - 1) / 2
        } else {
            0
        };
        Self {
            camera_count,
            observations: vec![Vec::new(); num_pairs],
            max_observations,
        }
    }

    /// Add an observation of the offset between camera `cam_a` and `cam_b`
    /// at time `time_seconds`. The offset is in samples.
    ///
    /// Returns `false` if the camera indices are invalid or equal.
    pub fn add_observation(
        &mut self,
        cam_a: usize,
        cam_b: usize,
        time_seconds: f64,
        offset_samples: f64,
    ) -> bool {
        if let Some(pair_idx) = self.pair_index(cam_a, cam_b) {
            let obs = &mut self.observations[pair_idx];
            obs.push((time_seconds, offset_samples));
            if obs.len() > self.max_observations {
                obs.remove(0);
            }
            true
        } else {
            false
        }
    }

    /// Compute the drift estimate for a specific camera pair.
    ///
    /// Returns `None` if the pair is invalid or has fewer than 2 observations.
    pub fn estimate_pair(&self, cam_a: usize, cam_b: usize) -> Option<GenlockDriftResult> {
        let pair_idx = self.pair_index(cam_a, cam_b)?;
        let obs = &self.observations[pair_idx];
        if obs.len() < 2 {
            return None;
        }

        let (slope, intercept, r_sq, rms) = linear_regression_with_stats(obs);

        // Convert slope (samples/second) to PPM.
        // If sample_rate is unknown, we express PPM as slope * 1e6 / nominal_rate.
        // For a general-purpose estimator, report raw slope and let the caller
        // convert. Here we assume 48000 Hz as a common default.
        let drift_ppm = slope / 48.0; // slope / (48000 / 1_000_000)

        Some(GenlockDriftResult {
            camera_a: cam_a.min(cam_b),
            camera_b: cam_a.max(cam_b),
            drift_ppm,
            offset_at_zero: intercept,
            r_squared: r_sq,
            rms_residual: rms,
            num_observations: obs.len(),
        })
    }

    /// Compute drift estimates for all camera pairs that have sufficient data.
    pub fn estimate_all(&self) -> Vec<GenlockDriftResult> {
        let mut results = Vec::new();
        for a in 0..self.camera_count {
            for b in (a + 1)..self.camera_count {
                if let Some(result) = self.estimate_pair(a, b) {
                    results.push(result);
                }
            }
        }
        results
    }

    /// Predict the offset at a future time for a camera pair.
    ///
    /// Returns `None` if the pair has insufficient data for a drift model.
    pub fn predicted_offset(&self, cam_a: usize, cam_b: usize, time_seconds: f64) -> Option<f64> {
        let pair_idx = self.pair_index(cam_a, cam_b)?;
        let obs = &self.observations[pair_idx];
        if obs.len() < 2 {
            return None;
        }
        let (slope, intercept, _, _) = linear_regression_with_stats(obs);
        Some(slope * time_seconds + intercept)
    }

    /// Number of observations for a camera pair.
    pub fn observation_count(&self, cam_a: usize, cam_b: usize) -> usize {
        self.pair_index(cam_a, cam_b)
            .map(|i| self.observations[i].len())
            .unwrap_or(0)
    }

    /// Number of cameras.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.camera_count
    }

    /// Clear all observations.
    pub fn clear(&mut self) {
        for obs in &mut self.observations {
            obs.clear();
        }
    }

    // -- Internal helpers --

    /// Map (cam_a, cam_b) to a pair index. Returns None if invalid.
    fn pair_index(&self, cam_a: usize, cam_b: usize) -> Option<usize> {
        if cam_a >= self.camera_count || cam_b >= self.camera_count || cam_a == cam_b {
            return None;
        }
        let (lo, hi) = if cam_a < cam_b {
            (cam_a, cam_b)
        } else {
            (cam_b, cam_a)
        };
        // Triangular index: sum of (camera_count - 1) + (camera_count - 2) + ... for rows < lo
        // plus (hi - lo - 1).
        let idx = lo * (2 * self.camera_count - lo - 1) / 2 + (hi - lo - 1);
        if idx < self.observations.len() {
            Some(idx)
        } else {
            None
        }
    }
}

/// Fit y = slope * x + intercept via least squares, and return
/// (slope, intercept, r_squared, rms_residual).
fn linear_regression_with_stats(data: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let n = data.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = data.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        let mean_y = sum_y / n;
        return (0.0, mean_y, 0.0, 0.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // R-squared
    let mean_y = sum_y / n;
    let ss_tot: f64 = data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = data
        .iter()
        .map(|(x, y)| {
            let pred = slope * x + intercept;
            (y - pred).powi(2)
        })
        .sum();

    let r_sq = if ss_tot > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    let rms = (ss_res / n).sqrt();

    (slope, intercept, r_sq, rms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pll_state_creation() {
        let pll = PllState::new(0.1, 0.001);
        assert_eq!(pll.kp, 0.1);
        assert_eq!(pll.ki, 0.001);
        assert_eq!(pll.phase_error, 0.0);
        assert_eq!(pll.correction, 0.0);
    }

    #[test]
    fn test_pll_default() {
        let pll = PllState::default();
        assert_eq!(pll.kp, 0.1);
        assert_eq!(pll.ki, 0.001);
    }

    #[test]
    fn test_pll_update_convergence() {
        let mut pll = PllState::new(0.5, 0.01);
        let mut phase = 10.0_f64;
        for _ in 0..50 {
            let correction = pll.update(phase, 0.0);
            phase -= correction;
        }
        assert!(phase.abs() < 1.0, "PLL should converge: {phase}");
    }

    #[test]
    fn test_pll_reset() {
        let mut pll = PllState::new(0.1, 0.001);
        pll.update(5.0, 0.0);
        assert_ne!(pll.phase_error, 0.0);
        pll.reset();
        assert_eq!(pll.phase_error, 0.0);
        assert_eq!(pll.filter_state, 0.0);
        assert_eq!(pll.correction, 0.0);
    }

    #[test]
    fn test_pll_lock_detection() {
        let pll = PllState::new(0.1, 0.001);
        // Fresh PLL with zero errors should be locked
        assert!(pll.is_locked(1.0));
    }

    #[test]
    fn test_drift_rate_estimator_creation() {
        let est = DriftRateEstimator::new(50);
        assert_eq!(est.max_window, 50);
        assert_eq!(est.drift_ppm(), 0.0);
        assert!(est.is_empty());
    }

    #[test]
    fn test_drift_rate_single_measurement() {
        let mut est = DriftRateEstimator::new(100);
        est.add_measurement(1.0, 5.0);
        assert_eq!(est.len(), 1);
        // Need at least 2 for regression
        assert_eq!(est.drift_ppm(), 0.0);
    }

    #[test]
    fn test_drift_rate_two_measurements() {
        let mut est = DriftRateEstimator::new(100);
        est.add_measurement(0.0, 0.0);
        est.add_measurement(1.0, 48.0); // 48 samples/sec drift
        assert_ne!(est.drift_ppm(), 0.0);
    }

    #[test]
    fn test_drift_rate_window_limit() {
        let mut est = DriftRateEstimator::new(5);
        for i in 0..10 {
            est.add_measurement(i as f64, i as f64 * 2.0);
        }
        assert_eq!(est.len(), 5);
    }

    #[test]
    fn test_drift_rate_clear() {
        let mut est = DriftRateEstimator::new(100);
        est.add_measurement(0.0, 0.0);
        est.add_measurement(1.0, 10.0);
        est.clear();
        assert!(est.is_empty());
        assert_eq!(est.drift_ppm(), 0.0);
    }

    #[test]
    fn test_sample_adjustment_creation() {
        let adj = SampleAdjustment::new(2, 0.3, 15.7);
        assert_eq!(adj.delta, 2);
        assert!((adj.fractional - 0.3).abs() < f64::EPSILON);
        assert!(adj.needs_adjustment());
    }

    #[test]
    fn test_sample_adjustment_no_adjustment() {
        let adj = SampleAdjustment::new(0, 0.5, 0.5);
        assert!(!adj.needs_adjustment());
    }

    #[test]
    fn test_drift_corrector_creation() {
        let dc = DriftCorrector::new(48000);
        assert_eq!(dc.sample_rate, 48000);
        assert_eq!(dc.total_samples, 0);
    }

    #[test]
    fn test_drift_corrector_process_block() {
        let mut dc = DriftCorrector::new(48000);
        // Add some measurements first
        dc.estimator.add_measurement(0.0, 0.0);
        dc.estimator.add_measurement(1.0, 48.0);
        let adj = dc.process_block(480, 48.0);
        // Should produce some adjustment given drift
        assert_eq!(adj.delta.abs() <= 1, true); // small block, small delta
    }

    #[test]
    fn test_drift_corrector_reset() {
        let mut dc = DriftCorrector::new(48000);
        dc.estimator.add_measurement(0.0, 0.0);
        dc.estimator.add_measurement(1.0, 100.0);
        dc.total_samples = 1000;
        dc.reset();
        assert_eq!(dc.total_samples, 0);
        assert_eq!(dc.accumulator, 0.0);
        assert!(dc.estimator.is_empty());
    }

    #[test]
    fn test_drift_corrector_pll_update() {
        let mut dc = DriftCorrector::new(48000);
        let correction = dc.update_pll(10.0, 0.0);
        assert!(correction.abs() > 0.0);
    }

    // ── GenlockDriftEstimator ────────────────────────────────────────────────

    #[test]
    fn test_genlock_creation() {
        let est = GenlockDriftEstimator::new(4, 1000);
        assert_eq!(est.camera_count(), 4);
        // 4 cameras => 6 pairs
        assert_eq!(est.observations.len(), 6);
    }

    #[test]
    fn test_genlock_add_observation() {
        let mut est = GenlockDriftEstimator::new(3, 100);
        assert!(est.add_observation(0, 1, 0.0, 0.0));
        assert!(est.add_observation(0, 1, 1.0, 48.0));
        assert_eq!(est.observation_count(0, 1), 2);
        // Symmetry: (1, 0) should give same count
        assert_eq!(est.observation_count(1, 0), 2);
    }

    #[test]
    fn test_genlock_invalid_pair() {
        let mut est = GenlockDriftEstimator::new(2, 100);
        assert!(!est.add_observation(0, 0, 0.0, 0.0)); // same camera
        assert!(!est.add_observation(0, 5, 0.0, 0.0)); // out of bounds
    }

    #[test]
    fn test_genlock_linear_drift() {
        let mut est = GenlockDriftEstimator::new(2, 1000);
        // Feed perfectly linear drift: 10 samples/second
        for i in 0..100 {
            let t = i as f64;
            est.add_observation(0, 1, t, t * 10.0);
        }
        let result = est.estimate_pair(0, 1).expect("should have enough data");
        // slope = 10 samples/s => PPM = 10 / 48 ≈ 0.2083
        assert!(
            (result.drift_ppm - 10.0 / 48.0).abs() < 0.01,
            "drift_ppm={}",
            result.drift_ppm
        );
        assert!(result.r_squared > 0.999, "r²={}", result.r_squared);
        assert!(result.rms_residual < 0.01, "rms={}", result.rms_residual);
    }

    #[test]
    fn test_genlock_predicted_offset() {
        let mut est = GenlockDriftEstimator::new(2, 1000);
        est.add_observation(0, 1, 0.0, 0.0);
        est.add_observation(0, 1, 10.0, 100.0);
        // slope = 10, intercept = 0
        let pred = est.predicted_offset(0, 1, 20.0).expect("should predict");
        assert!((pred - 200.0).abs() < 1.0, "pred={pred}");
    }

    #[test]
    fn test_genlock_insufficient_data() {
        let mut est = GenlockDriftEstimator::new(2, 100);
        est.add_observation(0, 1, 0.0, 0.0);
        assert!(est.estimate_pair(0, 1).is_none());
        assert!(est.predicted_offset(0, 1, 5.0).is_none());
    }

    #[test]
    fn test_genlock_estimate_all() {
        let mut est = GenlockDriftEstimator::new(3, 100);
        // Only add data for pair (0,1)
        est.add_observation(0, 1, 0.0, 0.0);
        est.add_observation(0, 1, 1.0, 5.0);
        let results = est.estimate_all();
        assert_eq!(results.len(), 1); // only one pair has data
        assert_eq!(results[0].camera_a, 0);
        assert_eq!(results[0].camera_b, 1);
    }

    #[test]
    fn test_genlock_clear() {
        let mut est = GenlockDriftEstimator::new(2, 100);
        est.add_observation(0, 1, 0.0, 0.0);
        est.add_observation(0, 1, 1.0, 10.0);
        assert_eq!(est.observation_count(0, 1), 2);
        est.clear();
        assert_eq!(est.observation_count(0, 1), 0);
    }

    #[test]
    fn test_genlock_max_observations() {
        let mut est = GenlockDriftEstimator::new(2, 5);
        for i in 0..10 {
            est.add_observation(0, 1, i as f64, i as f64 * 2.0);
        }
        assert_eq!(est.observation_count(0, 1), 5);
    }

    // ── linear_regression_with_stats ─────────────────────────────────────────

    #[test]
    fn test_linear_regression_perfect_line() {
        let data: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 3.0 * i as f64 + 2.0)).collect();
        let (slope, intercept, r_sq, rms) = linear_regression_with_stats(&data);
        assert!((slope - 3.0).abs() < 1e-10);
        assert!((intercept - 2.0).abs() < 1e-10);
        assert!((r_sq - 1.0).abs() < 1e-10);
        assert!(rms < 1e-10);
    }

    #[test]
    fn test_linear_regression_constant() {
        let data: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 7.0)).collect();
        let (slope, intercept, _, _) = linear_regression_with_stats(&data);
        assert!(slope.abs() < 1e-10);
        assert!((intercept - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_regression_single_point() {
        let data = vec![(1.0, 2.0)];
        let (slope, _, _, _) = linear_regression_with_stats(&data);
        assert_eq!(slope, 0.0);
    }
}
