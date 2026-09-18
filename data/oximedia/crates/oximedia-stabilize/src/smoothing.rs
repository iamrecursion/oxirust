//! Trajectory smoothing algorithms: Gaussian smoothing, exponential moving
//! average, and a scalar Kalman filter approximation.
//!
//! All algorithms operate on a sequence of scalar values that represent one
//! component of a motion trajectory (e.g. horizontal translation, rotation
//! angle).  Multi-dimensional trajectories can be smoothed component-by-component.

#![allow(dead_code)]

use std::f64::consts::PI;

/// Result of running a smoother over a signal.
#[derive(Debug, Clone)]
pub struct SmoothedSignal {
    /// Smoothed values (same length as the input).
    pub values: Vec<f64>,
    /// The algorithm used.
    pub algorithm: SmoothingAlgorithm,
}

/// Smoothing algorithm identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothingAlgorithm {
    /// Gaussian (convolution-based).
    Gaussian,
    /// Exponential moving average.
    Ema,
    /// Scalar Kalman filter approximation.
    Kalman,
    /// Box (uniform) filter.
    Box,
}

/// Applies a 1-D Gaussian smoothing filter to `signal`.
///
/// # Arguments
/// * `signal` – Input values.
/// * `sigma` – Standard deviation of the Gaussian in samples (must be > 0).
///
/// Uses reflect-padding at the boundaries.
#[must_use]
pub fn gaussian_smooth(signal: &[f64], sigma: f64) -> SmoothedSignal {
    if signal.is_empty() || sigma <= 0.0 {
        return SmoothedSignal {
            values: signal.to_vec(),
            algorithm: SmoothingAlgorithm::Gaussian,
        };
    }

    // Build kernel (truncate at 3σ)
    let half_w = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = (0..=2 * half_w)
        .map(|i| {
            let x = i as f64 - half_w as f64;
            (-0.5 * (x / sigma).powi(2)).exp()
        })
        .collect();
    let kernel_sum: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|&k| k / kernel_sum).collect();

    let n = signal.len();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            kernel
                .iter()
                .enumerate()
                .map(|(ki, &k)| {
                    let offset = ki as isize - half_w as isize;
                    let idx = (i as isize + offset).max(0).min(n as isize - 1) as usize;
                    k * signal[idx]
                })
                .sum()
        })
        .collect();

    SmoothedSignal {
        values,
        algorithm: SmoothingAlgorithm::Gaussian,
    }
}

/// Applies an exponential moving average (EMA) filter.
///
/// # Arguments
/// * `signal` – Input values.
/// * `alpha` – Smoothing factor in (0, 1]. Larger values follow the input more closely.
///
/// Both a forward and backward pass are applied (zero-phase filtering).
#[must_use]
pub fn ema_smooth(signal: &[f64], alpha: f64) -> SmoothedSignal {
    if signal.is_empty() {
        return SmoothedSignal {
            values: Vec::new(),
            algorithm: SmoothingAlgorithm::Ema,
        };
    }

    let alpha = alpha.clamp(1e-6, 1.0);

    // Forward pass
    let mut forward = vec![0.0f64; signal.len()];
    forward[0] = signal[0];
    for i in 1..signal.len() {
        forward[i] = alpha * signal[i] + (1.0 - alpha) * forward[i - 1];
    }

    // Backward pass (to make it zero-phase)
    let mut output = vec![0.0f64; signal.len()];
    let last = forward.len() - 1;
    output[last] = forward[last];
    for i in (0..last).rev() {
        output[i] = alpha * forward[i] + (1.0 - alpha) * output[i + 1];
    }

    SmoothedSignal {
        values: output,
        algorithm: SmoothingAlgorithm::Ema,
    }
}

/// Scalar Kalman filter state.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// Process noise variance (Q).
    pub process_noise: f64,
    /// Measurement noise variance (R).
    pub measurement_noise: f64,
    /// Current state estimate.
    pub state: f64,
    /// Estimate error covariance.
    pub covariance: f64,
}

impl KalmanFilter {
    /// Creates a new [`KalmanFilter`].
    ///
    /// # Arguments
    /// * `process_noise` – How much the underlying state is expected to change (Q).
    /// * `measurement_noise` – How noisy the measurements are (R).
    /// * `initial_state` – Initial state estimate.
    #[must_use]
    pub fn new(process_noise: f64, measurement_noise: f64, initial_state: f64) -> Self {
        Self {
            process_noise,
            measurement_noise,
            state: initial_state,
            covariance: 1.0,
        }
    }

    /// Updates the filter with a new measurement and returns the filtered value.
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Predict
        let predicted_covariance = self.covariance + self.process_noise;

        // Update (Kalman gain)
        let gain = predicted_covariance / (predicted_covariance + self.measurement_noise);
        self.state += gain * (measurement - self.state);
        self.covariance = (1.0 - gain) * predicted_covariance;

        self.state
    }
}

/// Applies a scalar Kalman filter to a signal.
///
/// # Arguments
/// * `signal` – Input values.
/// * `process_noise` – Q parameter of the Kalman filter.
/// * `measurement_noise` – R parameter of the Kalman filter.
#[must_use]
pub fn kalman_smooth(signal: &[f64], process_noise: f64, measurement_noise: f64) -> SmoothedSignal {
    if signal.is_empty() {
        return SmoothedSignal {
            values: Vec::new(),
            algorithm: SmoothingAlgorithm::Kalman,
        };
    }

    let mut kf = KalmanFilter::new(process_noise, measurement_noise, signal[0]);
    let values: Vec<f64> = signal.iter().map(|&m| kf.update(m)).collect();

    SmoothedSignal {
        values,
        algorithm: SmoothingAlgorithm::Kalman,
    }
}

/// Applies a uniform (box) filter with the given half-window.
///
/// Uses reflect-padding at the boundaries.
///
/// # Arguments
/// * `signal` – Input values.
/// * `half_window` – Number of samples on each side of the centre to include.
#[must_use]
pub fn box_smooth(signal: &[f64], half_window: usize) -> SmoothedSignal {
    if signal.is_empty() {
        return SmoothedSignal {
            values: Vec::new(),
            algorithm: SmoothingAlgorithm::Box,
        };
    }

    let n = signal.len();
    let window = 2 * half_window + 1;

    let values: Vec<f64> = (0..n)
        .map(|i| {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);
            let count = end - start;
            signal[start..end].iter().sum::<f64>() / count as f64
        })
        .collect();

    let _ = window; // suppress unused
    SmoothedSignal {
        values,
        algorithm: SmoothingAlgorithm::Box,
    }
}

/// Computes the mean absolute error between two signals.
///
/// Returns 0.0 if both are empty.
#[must_use]
pub fn mean_absolute_error(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    a[..len]
        .iter()
        .zip(&b[..len])
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>()
        / len as f64
}

/// Generates a Gaussian kernel of size `2*half_w + 1` with standard deviation `sigma`.
#[must_use]
pub fn gaussian_kernel(half_w: usize, sigma: f64) -> Vec<f64> {
    let kernel: Vec<f64> = (0..=2 * half_w)
        .map(|i| {
            let x = i as f64 - half_w as f64;
            (-0.5 * (x / sigma).powi(2)).exp() / (sigma * (2.0 * PI).sqrt())
        })
        .collect();
    let s: f64 = kernel.iter().sum();
    kernel.iter().map(|&k| k / s).collect()
}

/// Selects and applies the best smoother for a given noise-to-signal ratio.
///
/// - Low NSR (< 0.1): box filter (signal already clean)
/// - Medium NSR (0.1–0.5): Gaussian
/// - High NSR (> 0.5): Kalman
#[must_use]
pub fn adaptive_smooth(signal: &[f64], noise_to_signal_ratio: f64) -> SmoothedSignal {
    if noise_to_signal_ratio < 0.1 {
        box_smooth(signal, 5)
    } else if noise_to_signal_ratio < 0.5 {
        gaussian_smooth(signal, 5.0)
    } else {
        kalman_smooth(signal, 1e-3, 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    fn constant(n: usize, v: f64) -> Vec<f64> {
        vec![v; n]
    }

    #[test]
    fn test_gaussian_smooth_length_preserved() {
        let s = ramp(50);
        let out = gaussian_smooth(&s, 3.0);
        assert_eq!(out.values.len(), s.len());
    }

    #[test]
    fn test_gaussian_smooth_constant_unchanged() {
        let s = constant(50, 7.0);
        let out = gaussian_smooth(&s, 3.0);
        for &v in &out.values {
            assert!((v - 7.0).abs() < 1e-6, "value = {v}");
        }
    }

    #[test]
    fn test_gaussian_smooth_reduces_variance() {
        // Noisy signal: alternating ±1
        let noisy: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let out = gaussian_smooth(&noisy, 5.0);
        let var_in: f64 = noisy.iter().map(|&x| x * x).sum::<f64>() / noisy.len() as f64;
        let var_out: f64 = out.values.iter().map(|&x| x * x).sum::<f64>() / out.values.len() as f64;
        assert!(var_out < var_in, "smoothing should reduce variance");
    }

    #[test]
    fn test_gaussian_smooth_algorithm_tag() {
        let out = gaussian_smooth(&ramp(10), 2.0);
        assert_eq!(out.algorithm, SmoothingAlgorithm::Gaussian);
    }

    #[test]
    fn test_ema_smooth_constant_unchanged() {
        let s = constant(30, 5.0);
        let out = ema_smooth(&s, 0.3);
        for &v in &out.values {
            assert!((v - 5.0).abs() < 0.1, "value = {v}");
        }
    }

    #[test]
    fn test_ema_smooth_length_preserved() {
        let s = ramp(40);
        let out = ema_smooth(&s, 0.5);
        assert_eq!(out.values.len(), s.len());
    }

    #[test]
    fn test_ema_smooth_algorithm_tag() {
        let out = ema_smooth(&ramp(10), 0.5);
        assert_eq!(out.algorithm, SmoothingAlgorithm::Ema);
    }

    #[test]
    fn test_kalman_smooth_length_preserved() {
        let s = ramp(30);
        let out = kalman_smooth(&s, 1e-4, 0.1);
        assert_eq!(out.values.len(), s.len());
    }

    #[test]
    fn test_kalman_filter_update_converges() {
        let mut kf = KalmanFilter::new(1e-4, 0.1, 0.0);
        // Feed a constant measurement; filter should converge to it
        for _ in 0..200 {
            kf.update(5.0);
        }
        assert!((kf.state - 5.0).abs() < 0.05, "state = {}", kf.state);
    }

    #[test]
    fn test_kalman_smooth_algorithm_tag() {
        let out = kalman_smooth(&ramp(10), 1e-4, 0.1);
        assert_eq!(out.algorithm, SmoothingAlgorithm::Kalman);
    }

    #[test]
    fn test_box_smooth_constant_unchanged() {
        let s = constant(20, 3.0);
        let out = box_smooth(&s, 4);
        for &v in &out.values {
            assert!((v - 3.0).abs() < 1e-10, "value = {v}");
        }
    }

    #[test]
    fn test_box_smooth_length_preserved() {
        let s = ramp(30);
        let out = box_smooth(&s, 3);
        assert_eq!(out.values.len(), s.len());
    }

    #[test]
    fn test_box_smooth_algorithm_tag() {
        let out = box_smooth(&ramp(10), 2);
        assert_eq!(out.algorithm, SmoothingAlgorithm::Box);
    }

    #[test]
    fn test_mean_absolute_error_identical() {
        let s = ramp(20);
        let mae = mean_absolute_error(&s, &s);
        assert!(mae.abs() < 1e-10);
    }

    #[test]
    fn test_mean_absolute_error_known() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let mae = mean_absolute_error(&a, &b);
        assert!((mae - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel(5, 2.0);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "kernel sum = {sum}");
    }

    #[test]
    fn test_adaptive_smooth_returns_correct_length() {
        let s = ramp(50);
        let out = adaptive_smooth(&s, 0.3);
        assert_eq!(out.values.len(), s.len());
    }

    #[test]
    fn test_adaptive_smooth_low_nsr_uses_box() {
        let s = ramp(50);
        let out = adaptive_smooth(&s, 0.05);
        assert_eq!(out.algorithm, SmoothingAlgorithm::Box);
    }

    #[test]
    fn test_adaptive_smooth_high_nsr_uses_kalman() {
        let s = ramp(50);
        let out = adaptive_smooth(&s, 0.9);
        assert_eq!(out.algorithm, SmoothingAlgorithm::Kalman);
    }
}
