//! Offset correction algorithms: PI controller, EWMA, and outlier rejection.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Recommended clock frequency adjustment in parts-per-billion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrequencyAdjustment {
    /// Adjustment in ppb (positive = speed up, negative = slow down).
    pub ppb: f64,
}

impl FrequencyAdjustment {
    /// Create a new frequency adjustment.
    #[must_use]
    pub fn new(ppb: f64) -> Self {
        Self { ppb }
    }

    /// Zero adjustment.
    #[must_use]
    pub fn zero() -> Self {
        Self { ppb: 0.0 }
    }
}

/// Configuration for the PI controller.
#[derive(Debug, Clone)]
pub struct PiConfig {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
    /// Maximum integral windup in nanoseconds.
    pub integral_limit: f64,
    /// Maximum output adjustment in ppb.
    pub output_limit: f64,
}

impl Default for PiConfig {
    fn default() -> Self {
        Self {
            kp: 0.7,
            ki: 0.3,
            integral_limit: 200_000.0,
            output_limit: 500_000.0,
        }
    }
}

/// Proportional-Integral clock-offset controller.
#[derive(Debug)]
pub struct PiController {
    config: PiConfig,
    integral: f64,
    last_error: f64,
}

impl PiController {
    /// Create a controller with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(PiConfig::default())
    }

    /// Create a controller with custom configuration.
    #[must_use]
    pub fn with_config(config: PiConfig) -> Self {
        Self {
            config,
            integral: 0.0,
            last_error: 0.0,
        }
    }

    /// Update the controller with a new offset measurement (nanoseconds).
    ///
    /// Returns the recommended frequency adjustment in ppb.
    #[must_use]
    pub fn update(&mut self, offset_ns: f64) -> FrequencyAdjustment {
        self.integral = (self.integral + offset_ns)
            .clamp(-self.config.integral_limit, self.config.integral_limit);
        self.last_error = offset_ns;

        let output = self.config.kp * offset_ns + self.config.ki * self.integral;
        let clamped = output.clamp(-self.config.output_limit, self.config.output_limit);
        FrequencyAdjustment::new(clamped)
    }

    /// Reset controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.last_error = 0.0;
    }

    /// Current integral accumulator.
    #[must_use]
    pub fn integral(&self) -> f64 {
        self.integral
    }

    /// Last error applied.
    #[must_use]
    pub fn last_error(&self) -> f64 {
        self.last_error
    }
}

impl Default for PiController {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponentially-Weighted Moving Average for offset smoothing.
#[derive(Debug)]
pub struct EwmaFilter {
    alpha: f64,
    estimate: Option<f64>,
}

impl EwmaFilter {
    /// Create a new EWMA filter.
    ///
    /// `alpha` ∈ (0, 1]: weight given to the new sample.
    /// Higher alpha = less smoothing.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-6, 1.0),
            estimate: None,
        }
    }

    /// Update with a new offset sample.
    pub fn update(&mut self, sample: f64) -> f64 {
        let new_est = match self.estimate {
            None => sample,
            Some(prev) => prev + self.alpha * (sample - prev),
        };
        self.estimate = Some(new_est);
        new_est
    }

    /// Current EWMA estimate (None if no samples received).
    #[must_use]
    pub fn estimate(&self) -> Option<f64> {
        self.estimate
    }

    /// Reset the filter.
    pub fn reset(&mut self) {
        self.estimate = None;
    }
}

/// Parameters for the outlier rejection filter.
#[derive(Debug, Clone)]
pub struct OutlierConfig {
    /// Number of standard deviations beyond which a sample is rejected.
    pub sigma_threshold: f64,
    /// Minimum number of samples required before rejection starts.
    pub warmup_samples: usize,
}

impl Default for OutlierConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 3.0,
            warmup_samples: 8,
        }
    }
}

/// Rejects outlier offset measurements using a rolling mean/variance.
#[derive(Debug)]
pub struct OutlierFilter {
    config: OutlierConfig,
    samples: Vec<f64>,
    max_history: usize,
}

impl OutlierFilter {
    /// Create a new outlier filter.
    #[must_use]
    pub fn new(config: OutlierConfig, max_history: usize) -> Self {
        Self {
            config,
            samples: Vec::new(),
            max_history: max_history.max(4),
        }
    }

    /// Test whether a new sample is an outlier.
    ///
    /// Returns `true` if the sample should be discarded.
    #[must_use]
    pub fn is_outlier(&self, sample: f64) -> bool {
        if self.samples.len() < self.config.warmup_samples {
            return false;
        }
        let mean = self.mean();
        let std = self.std_dev();
        if std < 1.0 {
            return false;
        }
        (sample - mean).abs() > self.config.sigma_threshold * std
    }

    /// Feed a sample into the history window (regardless of outlier status).
    pub fn push(&mut self, sample: f64) {
        self.samples.push(sample);
        if self.samples.len() > self.max_history {
            self.samples.remove(0);
        }
    }

    /// Mean of the current sample window.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    /// Population standard deviation of the window.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let variance =
            self.samples.iter().map(|s| (s - m).powi(2)).sum::<f64>() / self.samples.len() as f64;
        variance.sqrt()
    }

    /// Number of samples currently stored.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clear all stored samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_adjustment_zero() {
        let a = FrequencyAdjustment::zero();
        assert!((a.ppb - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frequency_adjustment_new() {
        let a = FrequencyAdjustment::new(1234.5);
        assert!((a.ppb - 1234.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pi_controller_default() {
        let c = PiController::new();
        assert!((c.config.kp - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pi_controller_zero_offset_zero_output() {
        let mut c = PiController::new();
        let adj = c.update(0.0);
        assert!((adj.ppb - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pi_controller_positive_offset_positive_output() {
        let mut c = PiController::new();
        let adj = c.update(1000.0);
        assert!(adj.ppb > 0.0);
    }

    #[test]
    fn test_pi_controller_negative_offset_negative_output() {
        let mut c = PiController::new();
        let adj = c.update(-1000.0);
        assert!(adj.ppb < 0.0);
    }

    #[test]
    fn test_pi_controller_output_clamped() {
        let mut c = PiController::new();
        // Extreme offset should be clamped to output_limit
        let adj = c.update(1_000_000_000.0);
        assert!(adj.ppb.abs() <= c.config.output_limit);
    }

    #[test]
    fn test_pi_controller_reset_clears_integral() {
        let mut c = PiController::new();
        let _ = c.update(5000.0);
        c.reset();
        assert!((c.integral() - 0.0).abs() < f64::EPSILON);
        assert!((c.last_error() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ewma_filter_first_sample() {
        let mut f = EwmaFilter::new(0.2);
        let v = f.update(100.0);
        assert!((v - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ewma_filter_smooths() {
        let mut f = EwmaFilter::new(0.2);
        f.update(0.0);
        let v = f.update(100.0);
        // Should be 0 + 0.2 * (100 - 0) = 20
        assert!((v - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewma_filter_reset() {
        let mut f = EwmaFilter::new(0.5);
        f.update(50.0);
        f.reset();
        assert!(f.estimate().is_none());
    }

    #[test]
    fn test_outlier_filter_warmup() {
        let f = OutlierFilter::new(OutlierConfig::default(), 20);
        // Below warmup_samples, nothing is an outlier
        assert!(!f.is_outlier(1_000_000.0));
    }

    #[test]
    fn test_outlier_filter_push_and_count() {
        let mut f = OutlierFilter::new(OutlierConfig::default(), 20);
        for i in 0..5 {
            f.push(i as f64 * 10.0);
        }
        assert_eq!(f.sample_count(), 5);
    }

    #[test]
    fn test_outlier_filter_detects_outlier() {
        let mut f = OutlierFilter::new(
            OutlierConfig {
                warmup_samples: 4,
                ..OutlierConfig::default()
            },
            20,
        );
        // Use samples with variance > 1 so the std guard doesn't short-circuit.
        for v in [
            95.0_f64, 100.0, 105.0, 98.0, 102.0, 99.0, 101.0, 97.0, 103.0, 100.0,
        ] {
            f.push(v);
        }
        // A value many sigma away from ~100 should be detected as an outlier.
        assert!(f.is_outlier(100_000.0));
    }

    #[test]
    fn test_outlier_filter_mean() {
        let mut f = OutlierFilter::new(OutlierConfig::default(), 20);
        f.push(10.0);
        f.push(20.0);
        f.push(30.0);
        assert!((f.mean() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_outlier_filter_reset() {
        let mut f = OutlierFilter::new(OutlierConfig::default(), 20);
        f.push(50.0);
        f.reset();
        assert_eq!(f.sample_count(), 0);
    }
}
