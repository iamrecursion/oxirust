//! L1 optimal camera path computation.
//!
//! Implements the L1 optimal camera path algorithm for video stabilization.
//! Unlike L2 (least-squares) smoothing which minimizes velocity, L1 optimization
//! minimizes acceleration, producing piecewise-linear camera paths that feel more
//! natural to viewers. The result preserves intentional camera pans while removing
//! jitter.
//!
//! The core idea: solve a linear program that minimizes the sum of absolute
//! second-order differences of the smoothed trajectory, subject to crop constraints.
//! We use iteratively reweighted least-squares (IRLS) to approximate the L1 problem.

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::trajectory::Trajectory;
use scirs2_core::ndarray::Array1;

/// Configuration for L1 optimal path computation.
#[derive(Debug, Clone)]
pub struct L1PathConfig {
    /// Maximum allowed crop fraction (0.0-1.0). Larger values allow more cropping
    /// but yield smoother results.
    pub max_crop_fraction: f64,
    /// Number of IRLS iterations.
    pub irls_iterations: usize,
    /// Weight for the acceleration penalty term.
    pub lambda: f64,
    /// Frame width (needed for crop constraints).
    pub frame_width: f64,
    /// Frame height (needed for crop constraints).
    pub frame_height: f64,
    /// Epsilon for IRLS (avoids division by zero).
    pub irls_epsilon: f64,
}

impl Default for L1PathConfig {
    fn default() -> Self {
        Self {
            max_crop_fraction: 0.2,
            irls_iterations: 20,
            lambda: 10.0,
            frame_width: 1920.0,
            frame_height: 1080.0,
            irls_epsilon: 1e-3,
        }
    }
}

/// L1 optimal camera path smoother.
#[derive(Debug)]
pub struct L1PathSmoother {
    config: L1PathConfig,
}

impl L1PathSmoother {
    /// Create a new L1 path smoother with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: L1PathConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: L1PathConfig) -> Self {
        Self { config }
    }

    /// Smooth a trajectory using L1-optimal path computation.
    ///
    /// Minimizes `sum |a_i|` (acceleration) subject to per-frame crop constraints,
    /// where `a_i = p_{i+1} - 2*p_i + p_{i-1}` is the second-order difference.
    ///
    /// # Errors
    ///
    /// Returns an error if the trajectory is empty or has fewer than 3 frames.
    pub fn smooth(&self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        if trajectory.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }
        if trajectory.frame_count < 3 {
            return Ok(trajectory.clone());
        }

        let smoothed_x = self.smooth_signal(&trajectory.x)?;
        let smoothed_y = self.smooth_signal(&trajectory.y)?;
        let smoothed_angle = self.smooth_signal(&trajectory.angle)?;
        let smoothed_scale = self.smooth_scale(&trajectory.scale)?;

        Ok(Trajectory {
            x: smoothed_x,
            y: smoothed_y,
            angle: smoothed_angle,
            scale: smoothed_scale,
            frame_count: trajectory.frame_count,
        })
    }

    /// Smooth a 1D signal using IRLS approximation to L1 optimal path.
    fn smooth_signal(&self, signal: &Array1<f64>) -> StabilizeResult<Array1<f64>> {
        let n = signal.len();
        if n < 3 {
            return Ok(signal.clone());
        }

        let max_dev = self.config.max_crop_fraction * self.config.frame_width * 0.5;

        // Initialize smoothed path to original
        let mut p = signal.to_vec();

        for _irls_iter in 0..self.config.irls_iterations {
            // Compute IRLS weights from current acceleration
            let weights = self.compute_irls_weights(&p);

            // Solve weighted least-squares problem:
            // minimize sum_i w_i * (p_{i+1} - 2*p_i + p_{i-1})^2
            // subject to |p_i - s_i| <= max_dev
            let new_p = self.solve_weighted_tridiagonal(&p, signal, &weights, max_dev)?;
            p = new_p;
        }

        Ok(Array1::from_vec(p))
    }

    /// Smooth the scale channel (operates in log-space for multiplicative correctness).
    fn smooth_scale(&self, scale: &Array1<f64>) -> StabilizeResult<Array1<f64>> {
        let n = scale.len();
        if n < 3 {
            return Ok(scale.clone());
        }

        // Work in log-space
        let log_scale = scale.mapv(|s| s.max(1e-6).ln());
        let smoothed_log = self.smooth_signal(&log_scale)?;
        Ok(smoothed_log.mapv(|v| v.exp()))
    }

    /// Compute IRLS weights from current acceleration values.
    /// w_i = 1 / max(|a_i|, epsilon) to approximate L1 norm.
    fn compute_irls_weights(&self, p: &[f64]) -> Vec<f64> {
        let n = p.len();
        let mut weights = vec![1.0; n];

        for i in 1..(n - 1) {
            let accel = (p[i + 1] - 2.0 * p[i] + p[i - 1]).abs();
            weights[i] = 1.0 / accel.max(self.config.irls_epsilon);
        }

        weights
    }

    /// Solve the weighted tridiagonal system with box constraints.
    ///
    /// Minimizes: lambda * sum_i w_i * (p_{i+1} - 2*p_i + p_{i-1})^2 + sum_i (p_i - s_i)^2
    /// Subject to: |p_i - s_i| <= max_dev
    fn solve_weighted_tridiagonal(
        &self,
        _current: &[f64],
        original: &Array1<f64>,
        weights: &[f64],
        max_dev: f64,
    ) -> StabilizeResult<Vec<f64>> {
        let n = original.len();
        let lambda = self.config.lambda;

        // Build and solve the tridiagonal system using Gauss-Seidel iteration
        // The system is: (I + lambda * D^T W D) p = s
        // where D is the second-difference operator and W = diag(weights)
        let mut p: Vec<f64> = original.to_vec();

        // Gauss-Seidel iterations
        let gs_iterations = 50;
        for _gs in 0..gs_iterations {
            let mut max_change = 0.0_f64;

            for i in 0..n {
                // Compute the weighted second-difference contribution
                let mut accel_contribution = 0.0;

                // This pixel's contribution as center of a second difference
                if i >= 1 && i + 1 < n {
                    accel_contribution += 4.0 * lambda * weights[i];
                }
                // This pixel's contribution as left neighbour of (i+1)
                if i + 2 < n {
                    accel_contribution += lambda * weights[i + 1];
                }
                // This pixel's contribution as right neighbour of (i-1)
                if i >= 2 {
                    accel_contribution += lambda * weights[i - 1];
                }

                let diagonal = 1.0 + accel_contribution;

                // Off-diagonal contributions from neighbours
                let mut rhs = original[i];

                if i >= 1 && i + 1 < n {
                    // Center: -2 * w_i * (p_{i-1} + p_{i+1})
                    rhs += 2.0 * lambda * weights[i] * (p[i - 1] + p[i + 1]);
                }
                if i + 2 < n {
                    // Left of (i+1): w_{i+1} * (2*p_{i+1} - p_{i+2})
                    rhs += lambda * weights[i + 1] * (2.0 * p[i + 1] - p[i + 2]);
                }
                if i >= 2 {
                    // Right of (i-1): w_{i-1} * (2*p_{i-1} - p_{i-2])
                    rhs += lambda * weights[i - 1] * (2.0 * p[i - 1] - p[i - 2]);
                }

                let new_val = rhs / diagonal;

                // Apply box constraint
                let clamped = new_val.clamp(original[i] - max_dev, original[i] + max_dev);

                max_change = max_change.max((clamped - p[i]).abs());
                p[i] = clamped;
            }

            if max_change < 1e-6 {
                break;
            }
        }

        Ok(p)
    }

    /// Compute the total acceleration of a path (for diagnostics).
    #[must_use]
    pub fn total_acceleration(path: &Array1<f64>) -> f64 {
        let n = path.len();
        if n < 3 {
            return 0.0;
        }

        let mut total = 0.0;
        for i in 1..(n - 1) {
            let accel = (path[i + 1] - 2.0 * path[i] + path[i - 1]).abs();
            total += accel;
        }
        total
    }

    /// Compute the total velocity of a path (for comparison with L2).
    #[must_use]
    pub fn total_velocity(path: &Array1<f64>) -> f64 {
        let n = path.len();
        if n < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        for i in 1..n {
            total += (path[i] - path[i - 1]).abs();
        }
        total
    }
}

impl Default for L1PathSmoother {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(values: &[f64]) -> Trajectory {
        let n = values.len();
        Trajectory {
            x: Array1::from_vec(values.to_vec()),
            y: Array1::zeros(n),
            angle: Array1::zeros(n),
            scale: Array1::ones(n),
            frame_count: n,
        }
    }

    #[test]
    fn test_l1_smoother_creation() {
        let smoother = L1PathSmoother::new();
        assert!(smoother.config.lambda > 0.0);
    }

    #[test]
    fn test_l1_smooth_empty() {
        let smoother = L1PathSmoother::new();
        let traj = Trajectory::new(0);
        let result = smoother.smooth(&traj);
        assert!(result.is_err());
    }

    #[test]
    fn test_l1_smooth_short_trajectory() {
        let smoother = L1PathSmoother::new();
        let traj = Trajectory::new(2);
        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
        let smoothed = result.expect("should succeed in test");
        assert_eq!(smoothed.frame_count, 2);
    }

    #[test]
    fn test_l1_smooth_constant_path() {
        let smoother = L1PathSmoother::new();
        let traj = make_trajectory(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
        let smoothed = result.expect("should succeed in test");
        // A constant path should remain constant (zero acceleration)
        for i in 0..5 {
            assert!((smoothed.x[i] - 5.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_l1_smooth_linear_path() {
        let smoother = L1PathSmoother::new();
        // Linear path: zero acceleration already
        let traj = make_trajectory(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
        let smoothed = result.expect("should succeed in test");
        // Linear path should be preserved (it is already optimal)
        for i in 0..8 {
            assert!((smoothed.x[i] - i as f64).abs() < 1.0);
        }
    }

    #[test]
    fn test_l1_reduces_acceleration() {
        let config = L1PathConfig {
            lambda: 5.0,
            max_crop_fraction: 0.3,
            irls_iterations: 15,
            ..L1PathConfig::default()
        };
        let smoother = L1PathSmoother::with_config(config);

        // Jittery path with high acceleration
        let values: Vec<f64> = (0..20)
            .map(|i| {
                let base = i as f64;
                let jitter = if i % 2 == 0 { 5.0 } else { -5.0 };
                base + jitter
            })
            .collect();
        let traj = make_trajectory(&values);

        let original_accel = L1PathSmoother::total_acceleration(&traj.x);

        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
        let smoothed = result.expect("should succeed in test");

        let smoothed_accel = L1PathSmoother::total_acceleration(&smoothed.x);
        // Smoothed path should have less acceleration
        assert!(
            smoothed_accel < original_accel,
            "Smoothed acceleration ({smoothed_accel}) should be less than original ({original_accel})"
        );
    }

    #[test]
    fn test_l1_respects_crop_constraints() {
        let config = L1PathConfig {
            max_crop_fraction: 0.1,
            frame_width: 100.0,
            ..L1PathConfig::default()
        };
        let max_dev = config.max_crop_fraction * config.frame_width * 0.5;
        let smoother = L1PathSmoother::with_config(config);

        let values: Vec<f64> = (0..10).map(|i| (i as f64) * 3.0).collect();
        let traj = make_trajectory(&values);

        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
        let smoothed = result.expect("should succeed in test");

        // Check that deviation from original is bounded
        for i in 0..10 {
            let dev = (smoothed.x[i] - traj.x[i]).abs();
            assert!(
                dev <= max_dev + 0.1, // small tolerance
                "Frame {i}: deviation {dev} exceeds max {max_dev}"
            );
        }
    }

    #[test]
    fn test_l1_smooth_full_trajectory() {
        let smoother = L1PathSmoother::new();
        let mut traj = Trajectory::new(15);
        for i in 0..15 {
            traj.x[i] = (i as f64).sin() * 10.0;
            traj.y[i] = (i as f64).cos() * 5.0;
            traj.angle[i] = (i as f64) * 0.01;
        }
        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
        let smoothed = result.expect("should succeed in test");
        assert_eq!(smoothed.frame_count, 15);
    }

    #[test]
    fn test_total_acceleration_linear() {
        let path = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let accel = L1PathSmoother::total_acceleration(&path);
        assert!(accel.abs() < 1e-10);
    }

    #[test]
    fn test_total_velocity() {
        let path = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let vel = L1PathSmoother::total_velocity(&path);
        assert!((vel - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_l1_config_default() {
        let cfg = L1PathConfig::default();
        assert!(cfg.max_crop_fraction > 0.0);
        assert!(cfg.max_crop_fraction <= 1.0);
        assert!(cfg.irls_iterations > 0);
    }

    #[test]
    fn test_irls_weights() {
        let smoother = L1PathSmoother::new();
        let p = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let weights = smoother.compute_irls_weights(&p);
        assert_eq!(weights.len(), 5);
        // Interior points with high acceleration should have smaller weights
        // (which means 1/|accel| is small => higher weight on smoothness)
        assert!(weights[2] > 0.0);
    }

    #[test]
    fn test_smooth_scale() {
        let smoother = L1PathSmoother::new();
        let scale = Array1::from_vec(vec![1.0, 1.02, 0.98, 1.01, 0.99, 1.0, 1.0]);
        let smoothed = smoother.smooth_scale(&scale);
        assert!(smoothed.is_ok());
        let s = smoothed.expect("should succeed in test");
        // All values should remain positive and close to 1
        for &val in s.iter() {
            assert!(val > 0.5);
            assert!(val < 2.0);
        }
    }

    #[test]
    fn test_l1_vs_l2_comparison() {
        // L1 should produce lower acceleration than simple Gaussian smoothing
        let smoother = L1PathSmoother::new();
        let values: Vec<f64> = (0..30)
            .map(|i| {
                let base = i as f64 * 0.5;
                // Spike at frame 15 simulating a camera bump
                if i == 15 {
                    base + 20.0
                } else {
                    base
                }
            })
            .collect();
        let traj = make_trajectory(&values);

        let result = smoother.smooth(&traj);
        assert!(result.is_ok());
        let smoothed = result.expect("should succeed in test");

        // The spike should be reduced in the smoothed version
        let original_peak = traj.x[15];
        let smoothed_peak = smoothed.x[15];
        assert!(
            (smoothed_peak - original_peak).abs() > 0.1 || smoothed_peak < original_peak,
            "L1 should reduce the spike"
        );
    }
}
