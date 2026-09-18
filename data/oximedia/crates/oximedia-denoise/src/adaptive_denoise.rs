//! Adaptive denoising — threshold computation, online noise estimation and quality metrics.
#![allow(dead_code)]

use std::collections::VecDeque;

// ── Quality enum ──────────────────────────────────────────────────────────────

/// Perceived denoising quality level derived from SNR improvement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DenoiseQuality {
    /// Less than 3 dB SNR improvement.
    Poor,
    /// 3–8 dB SNR improvement.
    Fair,
    /// 8–15 dB SNR improvement.
    Good,
    /// > 15 dB SNR improvement.
    Excellent,
}

impl DenoiseQuality {
    /// Classify a quality level from the SNR improvement in dB.
    #[must_use]
    pub fn from_snr_improvement(snr_db: f32) -> Self {
        if snr_db >= 15.0 {
            DenoiseQuality::Excellent
        } else if snr_db >= 8.0 {
            DenoiseQuality::Good
        } else if snr_db >= 3.0 {
            DenoiseQuality::Fair
        } else {
            DenoiseQuality::Poor
        }
    }

    /// Expected SNR improvement in dB for this quality level (midpoint).
    #[must_use]
    pub fn snr_improvement_db(self) -> f32 {
        match self {
            DenoiseQuality::Poor => 1.5,
            DenoiseQuality::Fair => 5.5,
            DenoiseQuality::Good => 11.5,
            DenoiseQuality::Excellent => 20.0,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            DenoiseQuality::Poor => "Poor",
            DenoiseQuality::Fair => "Fair",
            DenoiseQuality::Good => "Good",
            DenoiseQuality::Excellent => "Excellent",
        }
    }
}

// ── Adaptive threshold ────────────────────────────────────────────────────────

/// Computes an adaptive denoising threshold based on local signal level.
///
/// The threshold scales linearly between a minimum (for low-signal regions)
/// and a maximum (for high-signal regions), ensuring noise-floor-level
/// pixels are aggressively filtered while bright content is preserved.
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    /// Threshold applied when `signal_level` is at its minimum (0.0).
    pub min_threshold: f32,
    /// Threshold applied when `signal_level` is at its maximum (1.0).
    pub max_threshold: f32,
    /// Sensitivity curve exponent (1.0 = linear, <1 = concave, >1 = convex).
    pub gamma: f32,
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold.
    #[must_use]
    pub fn new(min_threshold: f32, max_threshold: f32, gamma: f32) -> Self {
        Self {
            min_threshold,
            max_threshold,
            gamma,
        }
    }

    /// Compute the threshold for a normalised signal level in `[0.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_threshold(&self, signal_level: f32) -> f32 {
        let level = signal_level.clamp(0.0, 1.0);
        let t = level.powf(self.gamma);
        self.min_threshold + t * (self.max_threshold - self.min_threshold)
    }

    /// Whether the threshold at the given level would suppress the signal
    /// (i.e. the computed threshold is larger than the signal).
    #[must_use]
    pub fn would_suppress(&self, signal_level: f32) -> bool {
        self.compute_threshold(signal_level) > signal_level
    }
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::new(0.05, 0.3, 1.0)
    }
}

// ── Noise estimator ───────────────────────────────────────────────────────────

/// Online noise estimator using a sliding-window variance approach.
///
/// Processes one sample at a time and maintains an exponentially weighted
/// moving estimate of the noise standard deviation.
#[derive(Debug, Clone)]
pub struct NoiseEstimator {
    /// Smoothing factor α for EMA (0 < α ≤ 1).
    pub alpha: f32,
    /// Window of recent samples used for local variance.
    window: VecDeque<f32>,
    /// Maximum window length.
    window_size: usize,
    /// Current noise estimate (std-dev).
    estimate: f32,
    /// EMA of the squared sample for Welford-style variance.
    ema_sq: f32,
    /// EMA of the sample mean.
    ema_mean: f32,
}

impl NoiseEstimator {
    /// Create a new estimator with the given window size and smoothing factor.
    #[must_use]
    pub fn new(window_size: usize, alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(1e-6, 1.0),
            window: VecDeque::with_capacity(window_size),
            window_size,
            estimate: 0.0,
            ema_sq: 0.0,
            ema_mean: 0.0,
        }
    }

    /// Update the estimator with a new sample.
    pub fn update(&mut self, sample: f32) {
        // Sliding window
        self.window.push_back(sample);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }

        // Exponential moving average of mean and mean-square
        self.ema_mean = self.alpha * sample + (1.0 - self.alpha) * self.ema_mean;
        self.ema_sq = self.alpha * sample * sample + (1.0 - self.alpha) * self.ema_sq;

        // Variance = E[X²] - E[X]²
        let variance = (self.ema_sq - self.ema_mean * self.ema_mean).max(0.0);
        self.estimate = variance.sqrt();
    }

    /// Return the current noise estimate (standard deviation).
    #[must_use]
    pub fn estimate(&self) -> f32 {
        self.estimate
    }

    /// Return the number of samples currently in the window.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Return `true` when the estimator has seen at least `window_size` samples.
    #[must_use]
    pub fn is_warm(&self) -> bool {
        self.window.len() >= self.window_size
    }

    /// Reset the estimator state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.estimate = 0.0;
        self.ema_sq = 0.0;
        self.ema_mean = 0.0;
    }
}

impl Default for NoiseEstimator {
    fn default() -> Self {
        Self::new(64, 0.05)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // DenoiseQuality ──────────────────────────────────────────────────────────

    #[test]
    fn test_quality_from_snr_poor() {
        assert_eq!(
            DenoiseQuality::from_snr_improvement(1.0),
            DenoiseQuality::Poor
        );
    }

    #[test]
    fn test_quality_from_snr_fair() {
        assert_eq!(
            DenoiseQuality::from_snr_improvement(5.0),
            DenoiseQuality::Fair
        );
    }

    #[test]
    fn test_quality_from_snr_good() {
        assert_eq!(
            DenoiseQuality::from_snr_improvement(10.0),
            DenoiseQuality::Good
        );
    }

    #[test]
    fn test_quality_from_snr_excellent() {
        assert_eq!(
            DenoiseQuality::from_snr_improvement(20.0),
            DenoiseQuality::Excellent
        );
    }

    #[test]
    fn test_quality_snr_improvement_db_ordering() {
        assert!(
            DenoiseQuality::Poor.snr_improvement_db() < DenoiseQuality::Fair.snr_improvement_db()
        );
        assert!(
            DenoiseQuality::Fair.snr_improvement_db() < DenoiseQuality::Good.snr_improvement_db()
        );
        assert!(
            DenoiseQuality::Good.snr_improvement_db()
                < DenoiseQuality::Excellent.snr_improvement_db()
        );
    }

    #[test]
    fn test_quality_label() {
        assert_eq!(DenoiseQuality::Poor.label(), "Poor");
        assert_eq!(DenoiseQuality::Excellent.label(), "Excellent");
    }

    #[test]
    fn test_quality_ord() {
        assert!(DenoiseQuality::Poor < DenoiseQuality::Excellent);
    }

    // AdaptiveThreshold ───────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_threshold_at_zero() {
        let t = AdaptiveThreshold::new(0.1, 0.9, 1.0);
        let th = t.compute_threshold(0.0);
        assert!((th - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_threshold_at_one() {
        let t = AdaptiveThreshold::new(0.1, 0.9, 1.0);
        let th = t.compute_threshold(1.0);
        assert!((th - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_threshold_midpoint_linear() {
        let t = AdaptiveThreshold::new(0.0, 1.0, 1.0);
        let th = t.compute_threshold(0.5);
        assert!((th - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_threshold_clamping() {
        let t = AdaptiveThreshold::new(0.1, 0.5, 1.0);
        assert!((t.compute_threshold(-0.5) - 0.1).abs() < 1e-5);
        assert!((t.compute_threshold(2.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_threshold_would_suppress() {
        let t = AdaptiveThreshold::new(0.5, 0.9, 1.0);
        // signal_level=0.1 → threshold ≈ 0.54 > 0.1 → suppress
        assert!(t.would_suppress(0.1));
        // signal_level=1.0 → threshold=0.9 < 1.0 → no suppress
        assert!(!t.would_suppress(1.0));
    }

    // NoiseEstimator ──────────────────────────────────────────────────────────

    #[test]
    fn test_noise_estimator_starts_zero() {
        let e = NoiseEstimator::new(16, 0.1);
        assert!((e.estimate() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_noise_estimator_constant_signal_zero_noise() {
        let mut e = NoiseEstimator::new(16, 0.2);
        for _ in 0..200 {
            e.update(1.0);
        }
        // Constant signal should converge to ~zero noise
        assert!(e.estimate() < 0.01);
    }

    #[test]
    fn test_noise_estimator_warm_after_window() {
        let mut e = NoiseEstimator::new(8, 0.1);
        assert!(!e.is_warm());
        for _ in 0..8 {
            e.update(0.0);
        }
        assert!(e.is_warm());
    }

    #[test]
    fn test_noise_estimator_reset() {
        let mut e = NoiseEstimator::new(16, 0.1);
        for _ in 0..20 {
            e.update(1.0);
        }
        e.reset();
        assert_eq!(e.window_len(), 0);
        assert!((e.estimate() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_noise_estimator_noisy_signal_nonzero() {
        let mut e = NoiseEstimator::new(32, 0.1);
        for i in 0..200 {
            // Alternating +1/-1 noise around 0
            let v = if i % 2 == 0 { 1.0 } else { -1.0 };
            e.update(v);
        }
        // Should detect significant noise
        assert!(e.estimate() > 0.1);
    }
}
