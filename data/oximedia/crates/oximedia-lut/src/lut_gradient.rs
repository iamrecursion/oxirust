#![allow(dead_code)]
//! LUT gradient analysis and smooth transition evaluation.
//!
//! Analyzes the smoothness of color transforms by computing
//! discrete gradients across LUT entries, detecting discontinuities,
//! and measuring perceptual uniformity of transitions.

/// Represents a gradient sample along one axis of a 1D LUT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientSample {
    /// Index position in the LUT.
    pub index: usize,
    /// The gradient (first derivative) value at this position.
    pub gradient: f64,
}

/// Summary statistics of a gradient analysis.
#[derive(Clone, Debug)]
pub struct GradientStats {
    /// Minimum gradient value observed.
    pub min_gradient: f64,
    /// Maximum gradient value observed.
    pub max_gradient: f64,
    /// Mean gradient value.
    pub mean_gradient: f64,
    /// Standard deviation of gradient values.
    pub std_deviation: f64,
    /// Number of samples analyzed.
    pub sample_count: usize,
}

impl GradientStats {
    /// Returns the range (max - min) of gradient values.
    #[must_use]
    pub fn range(&self) -> f64 {
        self.max_gradient - self.min_gradient
    }

    /// Returns the coefficient of variation (`std_dev` / mean) if mean is nonzero.
    #[must_use]
    pub fn coefficient_of_variation(&self) -> Option<f64> {
        if self.mean_gradient.abs() < f64::EPSILON {
            None
        } else {
            Some(self.std_deviation / self.mean_gradient.abs())
        }
    }
}

/// Smoothness classification of a LUT transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothnessLevel {
    /// Very smooth transitions (low gradient variance).
    Smooth,
    /// Acceptable transitions for professional use.
    Acceptable,
    /// Noticeable steps or banding may occur.
    Rough,
    /// Severe discontinuities detected.
    Discontinuous,
}

/// Compute the discrete gradient (first differences) of a 1D LUT channel.
#[must_use]
pub fn compute_gradient_1d(data: &[f64]) -> Vec<GradientSample> {
    if data.len() < 2 {
        return Vec::new();
    }
    data.windows(2)
        .enumerate()
        .map(|(i, w)| GradientSample {
            index: i,
            gradient: w[1] - w[0],
        })
        .collect()
}

/// Compute gradient statistics from a set of gradient samples.
pub fn compute_gradient_stats(samples: &[GradientSample]) -> Option<GradientStats> {
    if samples.is_empty() {
        return None;
    }
    let n = samples.len() as f64;
    let min_g = samples.iter().map(|s| s.gradient).fold(f64::MAX, f64::min);
    let max_g = samples.iter().map(|s| s.gradient).fold(f64::MIN, f64::max);
    let sum: f64 = samples.iter().map(|s| s.gradient).sum();
    let mean = sum / n;
    let variance: f64 = samples
        .iter()
        .map(|s| (s.gradient - mean).powi(2))
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();

    Some(GradientStats {
        min_gradient: min_g,
        max_gradient: max_g,
        mean_gradient: mean,
        std_deviation: std_dev,
        sample_count: samples.len(),
    })
}

/// Detect discontinuities where the gradient changes abruptly.
///
/// `threshold` is the minimum ratio of gradient change to mean that
/// is considered a discontinuity.
#[must_use]
pub fn detect_discontinuities(samples: &[GradientSample], threshold: f64) -> Vec<GradientSample> {
    if samples.len() < 2 {
        return Vec::new();
    }
    let stats = match compute_gradient_stats(samples) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let mean_abs = stats.mean_gradient.abs().max(1e-12);

    let mut discontinuities = Vec::new();
    for w in samples.windows(2) {
        let change = (w[1].gradient - w[0].gradient).abs();
        if change / mean_abs > threshold {
            discontinuities.push(w[1]);
        }
    }
    discontinuities
}

/// Classify the smoothness of a 1D LUT channel based on gradient analysis.
#[must_use]
pub fn classify_smoothness(data: &[f64]) -> SmoothnessLevel {
    let samples = compute_gradient_1d(data);
    if samples.is_empty() {
        return SmoothnessLevel::Smooth;
    }
    let stats = match compute_gradient_stats(&samples) {
        Some(s) => s,
        None => return SmoothnessLevel::Smooth,
    };

    let cv = stats.coefficient_of_variation().unwrap_or(0.0).abs();

    if cv < 0.05 {
        SmoothnessLevel::Smooth
    } else if cv < 0.2 {
        SmoothnessLevel::Acceptable
    } else if cv < 0.5 {
        SmoothnessLevel::Rough
    } else {
        SmoothnessLevel::Discontinuous
    }
}

/// Compute the second derivative (gradient of the gradient) of a 1D channel.
#[must_use]
pub fn compute_second_derivative(data: &[f64]) -> Vec<f64> {
    if data.len() < 3 {
        return Vec::new();
    }
    data.windows(3).map(|w| w[2] - 2.0 * w[1] + w[0]).collect()
}

/// Find the maximum absolute second derivative (inflection strength).
pub fn max_inflection_strength(data: &[f64]) -> f64 {
    let d2 = compute_second_derivative(data);
    d2.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

/// Measure the overall smoothness score of a 1D LUT channel.
///
/// Returns a value in `[0.0, 1.0]` where 1.0 is perfectly smooth (linear).
#[must_use]
pub fn smoothness_score(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 1.0;
    }
    let d2 = compute_second_derivative(data);
    if d2.is_empty() {
        return 1.0;
    }
    let n = d2.len() as f64;
    let rms = (d2.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    // Map RMS of second derivative to a 0-1 score
    // lower rms = smoother
    let score = 1.0 / (1.0 + rms * 100.0);
    score.clamp(0.0, 1.0)
}

/// Generate a smooth linear ramp from 0 to 1 with the given number of steps.
#[must_use]
pub fn generate_linear_ramp(steps: usize) -> Vec<f64> {
    if steps == 0 {
        return Vec::new();
    }
    if steps == 1 {
        return vec![0.5];
    }
    let divisor = (steps - 1) as f64;
    (0..steps).map(|i| i as f64 / divisor).collect()
}

/// Generate a gamma curve ramp.
#[must_use]
pub fn generate_gamma_ramp(steps: usize, gamma: f64) -> Vec<f64> {
    if steps == 0 {
        return Vec::new();
    }
    if steps == 1 {
        return vec![0.5_f64.powf(gamma)];
    }
    let divisor = (steps - 1) as f64;
    (0..steps)
        .map(|i| (i as f64 / divisor).powf(gamma))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_gradient_1d_basic() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let grad = compute_gradient_1d(&data);
        assert_eq!(grad.len(), 4);
        for g in &grad {
            assert!((g.gradient - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_gradient_1d_empty() {
        assert!(compute_gradient_1d(&[]).is_empty());
        assert!(compute_gradient_1d(&[1.0]).is_empty());
    }

    #[test]
    fn test_gradient_stats_linear() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let grad = compute_gradient_1d(&data);
        let stats = compute_gradient_stats(&grad).expect("should succeed in test");
        assert!((stats.mean_gradient - 0.25).abs() < 1e-10);
        assert!(stats.std_deviation < 1e-10);
        assert_eq!(stats.sample_count, 4);
    }

    #[test]
    fn test_gradient_stats_range() {
        let data = vec![0.0, 0.1, 0.5, 0.6, 1.0];
        let grad = compute_gradient_1d(&data);
        let stats = compute_gradient_stats(&grad).expect("should succeed in test");
        assert!(stats.range() > 0.0);
    }

    #[test]
    fn test_gradient_stats_empty() {
        assert!(compute_gradient_stats(&[]).is_none());
    }

    #[test]
    fn test_coefficient_of_variation() {
        let stats = GradientStats {
            min_gradient: 0.0,
            max_gradient: 1.0,
            mean_gradient: 0.5,
            std_deviation: 0.1,
            sample_count: 10,
        };
        let cv = stats
            .coefficient_of_variation()
            .expect("should succeed in test");
        assert!((cv - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_coefficient_of_variation_zero_mean() {
        let stats = GradientStats {
            min_gradient: -1.0,
            max_gradient: 1.0,
            mean_gradient: 0.0,
            std_deviation: 0.5,
            sample_count: 10,
        };
        assert!(stats.coefficient_of_variation().is_none());
    }

    #[test]
    fn test_classify_smoothness_linear() {
        let data = generate_linear_ramp(256);
        assert_eq!(classify_smoothness(&data), SmoothnessLevel::Smooth);
    }

    #[test]
    fn test_classify_smoothness_step() {
        // step function should be discontinuous
        let mut data = vec![0.0; 128];
        data.extend(vec![1.0; 128]);
        let level = classify_smoothness(&data);
        assert!(level == SmoothnessLevel::Discontinuous || level == SmoothnessLevel::Rough);
    }

    #[test]
    fn test_detect_discontinuities_linear() {
        let data = generate_linear_ramp(100);
        let grad = compute_gradient_1d(&data);
        let disc = detect_discontinuities(&grad, 2.0);
        assert!(disc.is_empty());
    }

    #[test]
    fn test_second_derivative_linear() {
        let data = generate_linear_ramp(10);
        let d2 = compute_second_derivative(&data);
        for v in &d2 {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_max_inflection_strength_linear() {
        let data = generate_linear_ramp(50);
        let mis = max_inflection_strength(&data);
        assert!(mis < 1e-10);
    }

    #[test]
    fn test_smoothness_score_linear() {
        let data = generate_linear_ramp(256);
        let score = smoothness_score(&data);
        assert!(score > 0.95);
    }

    #[test]
    fn test_smoothness_score_short() {
        assert!((smoothness_score(&[0.5]) - 1.0).abs() < f64::EPSILON);
        assert!((smoothness_score(&[0.0, 1.0]) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_linear_ramp() {
        let ramp = generate_linear_ramp(5);
        assert_eq!(ramp.len(), 5);
        assert!((ramp[0] - 0.0).abs() < f64::EPSILON);
        assert!((ramp[4] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_gamma_ramp() {
        let ramp = generate_gamma_ramp(5, 2.2);
        assert_eq!(ramp.len(), 5);
        assert!((ramp[0] - 0.0).abs() < 1e-10);
        assert!((ramp[4] - 1.0).abs() < 1e-10);
        // Middle value should be < 0.5 for gamma > 1
        assert!(ramp[2] < 0.5);
    }

    #[test]
    fn test_generate_linear_ramp_empty() {
        assert!(generate_linear_ramp(0).is_empty());
    }

    #[test]
    fn test_generate_gamma_ramp_single() {
        let ramp = generate_gamma_ramp(1, 1.0);
        assert_eq!(ramp.len(), 1);
        assert!((ramp[0] - 0.5).abs() < 1e-10);
    }
}
