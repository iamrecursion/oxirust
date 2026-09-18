//! Gyroscope Extended Kalman Filter (EKF) fusion.
//!
//! Fuses gyroscope angular velocity data with visual motion estimates using an
//! Extended Kalman Filter. The gyroscope provides high-frequency, low-latency
//! orientation data but drifts over time, while visual estimates are accurate
//! on average but noisy and low-frequency. The EKF optimally combines both
//! sources to produce a drift-free, low-noise orientation stream.
//!
//! State vector: `[roll, pitch, yaw, bias_roll, bias_pitch, bias_yaw]`
//! - Angles in radians
//! - Biases in rad/s (gyroscope sensor biases that drift slowly)
//!
//! Process model: angles integrate bias-corrected gyro rates; biases follow
//! a random walk.
//!
//! Measurement model: visual motion estimates directly observe the angles.

use crate::error::{StabilizeError, StabilizeResult};
use crate::gyro::{GyroSample, GyroStream};

/// EKF state dimension.
const STATE_DIM: usize = 6;

/// A 6x6 matrix stored row-major.
#[derive(Debug, Clone)]
struct Mat6 {
    data: [f64; 36],
}

impl Mat6 {
    fn zeros() -> Self {
        Self { data: [0.0; 36] }
    }

    fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..STATE_DIM {
            m.set(i, i, 1.0);
        }
        m
    }

    fn diagonal(diag: &[f64; STATE_DIM]) -> Self {
        let mut m = Self::zeros();
        for i in 0..STATE_DIM {
            m.set(i, i, diag[i]);
        }
        m
    }

    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * STATE_DIM + c]
    }

    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * STATE_DIM + c] = v;
    }

    fn mul(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                let mut sum = 0.0;
                for k in 0..STATE_DIM {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    fn transpose(&self) -> Self {
        let mut result = Self::zeros();
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    fn add(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..36 {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    fn sub(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..36 {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }

    fn scale(&self, s: f64) -> Self {
        let mut result = Self::zeros();
        for i in 0..36 {
            result.data[i] = self.data[i] * s;
        }
        result
    }

    /// Invert a 6x6 matrix via Gauss-Jordan.
    fn try_inverse(&self) -> Option<Self> {
        let mut aug = [[0.0_f64; 12]; 6];
        for i in 0..6 {
            for j in 0..6 {
                aug[i][j] = self.get(i, j);
            }
            aug[i][i + 6] = 1.0;
        }

        for col in 0..6 {
            let mut max_val = aug[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..6 {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                return None;
            }
            if max_row != col {
                aug.swap(col, max_row);
            }

            let pivot = aug[col][col];
            for j in 0..12 {
                aug[col][j] /= pivot;
            }

            for row in 0..6 {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for j in 0..12 {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        let mut result = Self::zeros();
        for i in 0..6 {
            for j in 0..6 {
                result.set(i, j, aug[i][j + 6]);
            }
        }

        Some(result)
    }
}

/// A 6-element state vector.
#[derive(Debug, Clone)]
struct Vec6 {
    data: [f64; STATE_DIM],
}

impl Vec6 {
    fn zeros() -> Self {
        Self {
            data: [0.0; STATE_DIM],
        }
    }

    fn get(&self, i: usize) -> f64 {
        self.data[i]
    }

    fn set(&mut self, i: usize, v: f64) {
        self.data[i] = v;
    }
}

/// Configuration for the gyroscope EKF.
#[derive(Debug, Clone)]
pub struct GyroEkfConfig {
    /// Process noise for angle states (rad^2/s).
    pub angle_process_noise: f64,
    /// Process noise for bias states (rad^2/s^3).
    pub bias_process_noise: f64,
    /// Measurement noise for visual estimates (rad^2).
    pub visual_measurement_noise: f64,
    /// Initial angle uncertainty (rad^2).
    pub initial_angle_variance: f64,
    /// Initial bias uncertainty (rad^2/s^2).
    pub initial_bias_variance: f64,
}

impl Default for GyroEkfConfig {
    fn default() -> Self {
        Self {
            angle_process_noise: 1e-4,
            bias_process_noise: 1e-6,
            visual_measurement_noise: 1e-2,
            initial_angle_variance: 1.0,
            initial_bias_variance: 0.01,
        }
    }
}

/// A visual motion estimate at a given timestamp.
#[derive(Debug, Clone)]
pub struct VisualEstimate {
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
    /// Estimated roll angle (radians).
    pub roll: f64,
    /// Estimated pitch angle (radians).
    pub pitch: f64,
    /// Estimated yaw angle (radians).
    pub yaw: f64,
    /// Confidence in this estimate (0.0-1.0); used to scale measurement noise.
    pub confidence: f64,
}

/// Fused orientation output.
#[derive(Debug, Clone)]
pub struct FusedOrientation {
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
    /// Fused roll angle (radians).
    pub roll: f64,
    /// Fused pitch angle (radians).
    pub pitch: f64,
    /// Fused yaw angle (radians).
    pub yaw: f64,
    /// Estimated gyro bias: roll (rad/s).
    pub bias_roll: f64,
    /// Estimated gyro bias: pitch (rad/s).
    pub bias_pitch: f64,
    /// Estimated gyro bias: yaw (rad/s).
    pub bias_yaw: f64,
}

/// Gyroscope EKF for fusing gyro data with visual motion estimates.
#[derive(Debug)]
pub struct GyroEkf {
    config: GyroEkfConfig,
    /// Current state estimate: [roll, pitch, yaw, bias_roll, bias_pitch, bias_yaw].
    state: Vec6,
    /// Error covariance matrix (6x6).
    covariance: Mat6,
    /// Last update timestamp (microseconds).
    last_timestamp_us: Option<u64>,
    /// Whether the filter has been initialized.
    initialized: bool,
}

impl GyroEkf {
    /// Create a new EKF with default configuration.
    #[must_use]
    pub fn new() -> Self {
        let config = GyroEkfConfig::default();
        Self::with_config(config)
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: GyroEkfConfig) -> Self {
        let initial_p = Mat6::diagonal(&[
            config.initial_angle_variance,
            config.initial_angle_variance,
            config.initial_angle_variance,
            config.initial_bias_variance,
            config.initial_bias_variance,
            config.initial_bias_variance,
        ]);

        Self {
            config,
            state: Vec6::zeros(),
            covariance: initial_p,
            last_timestamp_us: None,
            initialized: false,
        }
    }

    /// Get the current fused orientation estimate.
    #[must_use]
    pub fn current_orientation(&self) -> FusedOrientation {
        FusedOrientation {
            timestamp_us: self.last_timestamp_us.unwrap_or(0),
            roll: self.state.get(0),
            pitch: self.state.get(1),
            yaw: self.state.get(2),
            bias_roll: self.state.get(3),
            bias_pitch: self.state.get(4),
            bias_yaw: self.state.get(5),
        }
    }

    /// Process a gyroscope sample (prediction step).
    ///
    /// Integrates the gyro angular rates into the state, correcting for
    /// estimated bias.
    pub fn predict(&mut self, gyro: &GyroSample) {
        let dt = if let Some(last_ts) = self.last_timestamp_us {
            if gyro.timestamp_us <= last_ts {
                return; // Don't go backward in time
            }
            (gyro.timestamp_us - last_ts) as f64 * 1e-6
        } else {
            // First sample: just initialize timestamp
            self.last_timestamp_us = Some(gyro.timestamp_us);
            return;
        };

        self.last_timestamp_us = Some(gyro.timestamp_us);

        // Convert gyro rates from deg/s to rad/s
        let gyro_roll = gyro.roll * std::f64::consts::PI / 180.0;
        let gyro_pitch = gyro.pitch * std::f64::consts::PI / 180.0;
        let gyro_yaw = gyro.yaw * std::f64::consts::PI / 180.0;

        // Bias-corrected rates
        let omega_roll = gyro_roll - self.state.get(3);
        let omega_pitch = gyro_pitch - self.state.get(4);
        let omega_yaw = gyro_yaw - self.state.get(5);

        // State prediction: angle += omega * dt, bias unchanged
        self.state.set(0, self.state.get(0) + omega_roll * dt);
        self.state.set(1, self.state.get(1) + omega_pitch * dt);
        self.state.set(2, self.state.get(2) + omega_yaw * dt);
        // Biases stay the same (random walk predicted by process noise)

        // Jacobian F = dF/dx
        // F is identity except: d(angle)/d(bias) = -dt
        let mut f = Mat6::identity();
        f.set(0, 3, -dt);
        f.set(1, 4, -dt);
        f.set(2, 5, -dt);

        // Process noise Q
        let q_angle = self.config.angle_process_noise * dt;
        let q_bias = self.config.bias_process_noise * dt;
        let q = Mat6::diagonal(&[q_angle, q_angle, q_angle, q_bias, q_bias, q_bias]);

        // P = F * P * F^T + Q
        let ft = f.transpose();
        let fp = f.mul(&self.covariance);
        self.covariance = fp.mul(&ft).add(&q);

        self.initialized = true;
    }

    /// Process a visual motion estimate (update/correction step).
    ///
    /// The measurement model directly observes [roll, pitch, yaw] from visual
    /// motion estimation.
    pub fn update(&mut self, visual: &VisualEstimate) {
        if !self.initialized {
            // Bootstrap from first visual estimate
            self.state.set(0, visual.roll);
            self.state.set(1, visual.pitch);
            self.state.set(2, visual.yaw);
            self.last_timestamp_us = Some(visual.timestamp_us);
            self.initialized = true;
            return;
        }

        // Measurement matrix H (3x6): H = [I_3x3, 0_3x3]
        // Innovation (measurement residual): y = z - H*x
        let z_roll = visual.roll;
        let z_pitch = visual.pitch;
        let z_yaw = visual.yaw;

        let y_roll = z_roll - self.state.get(0);
        let y_pitch = z_pitch - self.state.get(1);
        let y_yaw = z_yaw - self.state.get(2);

        // Scale measurement noise by inverse confidence
        let noise_scale = if visual.confidence > 0.01 {
            1.0 / visual.confidence
        } else {
            100.0
        };
        let r = self.config.visual_measurement_noise * noise_scale;

        // Innovation covariance: S = H*P*H^T + R
        // Since H picks first 3 rows/cols, S is the upper-left 3x3 of P + R*I
        // We compute the Kalman gain for each axis independently (assuming
        // the measurement noise is diagonal and the cross-correlations are
        // handled through P).
        // For a full 6-state update, we do it properly with 6x6 matrices.

        // H * P (3x6 result stored as rows)
        let mut hp = [[0.0_f64; 6]; 3];
        for j in 0..6 {
            hp[0][j] = self.covariance.get(0, j);
            hp[1][j] = self.covariance.get(1, j);
            hp[2][j] = self.covariance.get(2, j);
        }

        // S = H*P*H^T + R (3x3)
        let mut s = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] = hp[i][j]; // H*P*H^T (H^T picks first 3 cols)
            }
            s[i][i] += r; // Add R
        }

        // Invert S (3x3)
        let s_inv = match invert_3x3(&s) {
            Some(inv) => inv,
            None => return, // Singular, skip update
        };

        // Kalman gain K = P * H^T * S^{-1} (6x3)
        // P*H^T is the first 3 columns of P
        let mut k = [[0.0_f64; 3]; 6];
        for i in 0..6 {
            for j in 0..3 {
                let mut sum = 0.0;
                for m in 0..3 {
                    sum += self.covariance.get(i, m) * s_inv[m][j];
                }
                k[i][j] = sum;
            }
        }

        // State update: x = x + K * y
        let y = [y_roll, y_pitch, y_yaw];
        for i in 0..6 {
            let mut correction = 0.0;
            for j in 0..3 {
                correction += k[i][j] * y[j];
            }
            self.state.set(i, self.state.get(i) + correction);
        }

        // Covariance update: P = (I - K*H) * P
        // K*H is 6x6 where K is 6x3 and H is 3x6 (top-left identity)
        let mut kh = Mat6::zeros();
        for i in 0..6 {
            for j in 0..3 {
                // H[m][j] = delta(m, j) for m < 3
                kh.set(i, j, k[i][j]);
            }
        }

        let i_minus_kh = Mat6::identity().sub(&kh);
        self.covariance = i_minus_kh.mul(&self.covariance);
    }

    /// Run the full EKF pipeline on a gyro stream and visual estimates.
    ///
    /// Returns the fused orientation at each visual estimate timestamp.
    ///
    /// # Errors
    ///
    /// Returns an error if either input is empty.
    pub fn fuse(
        &mut self,
        gyro_stream: &GyroStream,
        visual_estimates: &[VisualEstimate],
    ) -> StabilizeResult<Vec<FusedOrientation>> {
        if gyro_stream.samples.is_empty() {
            return Err(StabilizeError::invalid_parameter(
                "gyro_stream",
                "empty gyro stream",
            ));
        }
        if visual_estimates.is_empty() {
            return Err(StabilizeError::invalid_parameter(
                "visual_estimates",
                "empty visual estimates",
            ));
        }

        let mut results = Vec::with_capacity(visual_estimates.len());

        // Merge gyro and visual events chronologically
        let mut gyro_idx = 0;
        let mut vis_idx = 0;

        while vis_idx < visual_estimates.len() {
            let vis_ts = visual_estimates[vis_idx].timestamp_us;

            // Process all gyro samples up to this visual timestamp
            while gyro_idx < gyro_stream.samples.len()
                && gyro_stream.samples[gyro_idx].timestamp_us <= vis_ts
            {
                self.predict(&gyro_stream.samples[gyro_idx]);
                gyro_idx += 1;
            }

            // Apply visual update
            self.update(&visual_estimates[vis_idx]);
            results.push(self.current_orientation());
            vis_idx += 1;
        }

        Ok(results)
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.state = Vec6::zeros();
        self.covariance = Mat6::diagonal(&[
            self.config.initial_angle_variance,
            self.config.initial_angle_variance,
            self.config.initial_angle_variance,
            self.config.initial_bias_variance,
            self.config.initial_bias_variance,
            self.config.initial_bias_variance,
        ]);
        self.last_timestamp_us = None;
        self.initialized = false;
    }
}

impl Default for GyroEkf {
    fn default() -> Self {
        Self::new()
    }
}

/// Invert a 3x3 matrix.
fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        return None;
    }

    let inv_det = 1.0 / det;

    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gyro_sample(ts: u64, r: f64, p: f64, y: f64) -> GyroSample {
        GyroSample::new(ts, r, p, y)
    }

    fn visual_est(ts: u64, r: f64, p: f64, y: f64) -> VisualEstimate {
        VisualEstimate {
            timestamp_us: ts,
            roll: r,
            pitch: p,
            yaw: y,
            confidence: 1.0,
        }
    }

    #[test]
    fn test_ekf_creation() {
        let ekf = GyroEkf::new();
        assert!(!ekf.initialized);
    }

    #[test]
    fn test_ekf_config_default() {
        let cfg = GyroEkfConfig::default();
        assert!(cfg.angle_process_noise > 0.0);
        assert!(cfg.visual_measurement_noise > 0.0);
    }

    #[test]
    fn test_ekf_predict_first_sample() {
        let mut ekf = GyroEkf::new();
        // First sample just sets timestamp
        ekf.predict(&gyro_sample(0, 0.0, 0.0, 0.0));
        assert!(!ekf.initialized);
    }

    #[test]
    fn test_ekf_predict_two_samples() {
        let mut ekf = GyroEkf::new();
        ekf.predict(&gyro_sample(0, 0.0, 0.0, 0.0));
        ekf.predict(&gyro_sample(1_000_000, 10.0, 0.0, 0.0)); // 10 deg/s for 1 second
        assert!(ekf.initialized);
        let orient = ekf.current_orientation();
        // Should integrate to ~10 degrees = ~0.1745 radians
        assert!(orient.roll.abs() > 0.1);
    }

    #[test]
    fn test_ekf_update_bootstrap() {
        let mut ekf = GyroEkf::new();
        let ve = visual_est(0, 0.1, 0.2, 0.3);
        ekf.update(&ve);
        assert!(ekf.initialized);
        let orient = ekf.current_orientation();
        assert!((orient.roll - 0.1).abs() < 1e-10);
        assert!((orient.pitch - 0.2).abs() < 1e-10);
        assert!((orient.yaw - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_ekf_predict_then_update() {
        let mut ekf = GyroEkf::new();

        // Initialize via visual update
        ekf.update(&visual_est(0, 0.0, 0.0, 0.0));

        // Predict with gyro (stationary)
        ekf.predict(&gyro_sample(0, 0.0, 0.0, 0.0));
        ekf.predict(&gyro_sample(100_000, 0.0, 0.0, 0.0)); // 100ms

        // Update with visual estimate
        ekf.update(&visual_est(100_000, 0.01, 0.0, 0.0));

        let orient = ekf.current_orientation();
        // Should be close to the visual estimate since gyro confirms ~0
        assert!(orient.roll.abs() < 0.1);
    }

    #[test]
    fn test_ekf_fuse_empty_gyro() {
        let mut ekf = GyroEkf::new();
        let stream = GyroStream::new(200.0);
        let visuals = vec![visual_est(0, 0.0, 0.0, 0.0)];
        let result = ekf.fuse(&stream, &visuals);
        assert!(result.is_err());
    }

    #[test]
    fn test_ekf_fuse_empty_visual() {
        let mut ekf = GyroEkf::new();
        let mut stream = GyroStream::new(200.0);
        stream.add_sample(gyro_sample(0, 0.0, 0.0, 0.0));
        let result = ekf.fuse(&stream, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ekf_fuse_basic() {
        let mut ekf = GyroEkf::new();

        let mut stream = GyroStream::new(1000.0);
        for i in 0..10u64 {
            stream.add_sample(gyro_sample(i * 1000, 0.0, 0.0, 0.0));
        }

        let visuals = vec![
            visual_est(0, 0.0, 0.0, 0.0),
            visual_est(5000, 0.01, 0.0, 0.0),
            visual_est(9000, 0.02, 0.0, 0.0),
        ];

        let result = ekf.fuse(&stream, &visuals);
        assert!(result.is_ok());
        let orientations = result.expect("should succeed in test");
        assert_eq!(orientations.len(), 3);
    }

    #[test]
    fn test_ekf_bias_estimation() {
        let mut ekf = GyroEkf::with_config(GyroEkfConfig {
            bias_process_noise: 1e-4,
            visual_measurement_noise: 1e-3,
            ..GyroEkfConfig::default()
        });

        // Simulate a gyro with constant bias of 5 deg/s on roll
        let bias = 5.0; // deg/s

        // Initialize
        ekf.update(&visual_est(0, 0.0, 0.0, 0.0));

        let n_steps = 50;
        for i in 1..=n_steps {
            let ts = (i as u64) * 10_000; // 10ms steps
                                          // Gyro reads bias (no actual rotation)
            ekf.predict(&gyro_sample(ts, bias, 0.0, 0.0));

            // Visual estimate says no rotation (ground truth)
            if i % 5 == 0 {
                ekf.update(&visual_est(ts, 0.0, 0.0, 0.0));
            }
        }

        let orient = ekf.current_orientation();
        // Bias estimate should converge toward the true bias
        let estimated_bias_deg = orient.bias_roll * 180.0 / std::f64::consts::PI;
        assert!(
            estimated_bias_deg > 0.5,
            "Bias estimate ({estimated_bias_deg} deg/s) should be positive toward {bias} deg/s"
        );
    }

    #[test]
    fn test_ekf_reset() {
        let mut ekf = GyroEkf::new();
        ekf.update(&visual_est(0, 0.5, 0.3, 0.1));
        assert!(ekf.initialized);

        ekf.reset();
        assert!(!ekf.initialized);
        let orient = ekf.current_orientation();
        assert!(orient.roll.abs() < 1e-10);
    }

    #[test]
    fn test_ekf_low_confidence_visual() {
        let mut ekf = GyroEkf::new();
        ekf.update(&visual_est(0, 0.0, 0.0, 0.0));
        ekf.predict(&gyro_sample(0, 0.0, 0.0, 0.0));
        ekf.predict(&gyro_sample(100_000, 0.0, 0.0, 0.0));

        // Low confidence visual estimate
        let ve = VisualEstimate {
            timestamp_us: 100_000,
            roll: 1.0,
            pitch: 0.0,
            yaw: 0.0,
            confidence: 0.01,
        };
        ekf.update(&ve);

        let orient = ekf.current_orientation();
        // With low confidence, the visual has less influence, but since there's
        // no strong gyro counter-evidence (stationary), the result can still be
        // pulled toward the visual. We just check it's less than the full visual
        // estimate of 1.0.
        assert!(
            orient.roll < 0.9,
            "Low-confidence visual should have less influence: roll = {}",
            orient.roll
        );
    }

    #[test]
    fn test_ekf_backward_timestamp_ignored() {
        let mut ekf = GyroEkf::new();
        ekf.predict(&gyro_sample(1000, 10.0, 0.0, 0.0));
        ekf.predict(&gyro_sample(2000, 10.0, 0.0, 0.0));

        let state_before = ekf.current_orientation().roll;
        // Try predicting with an earlier timestamp
        ekf.predict(&gyro_sample(500, 100.0, 0.0, 0.0));
        let state_after = ekf.current_orientation().roll;
        // Should be unchanged
        assert!((state_before - state_after).abs() < 1e-10);
    }

    #[test]
    fn test_mat6_inverse() {
        let m = Mat6::identity();
        let inv = m.try_inverse();
        assert!(inv.is_some());
        let inv = inv.expect("should succeed in test");
        for i in 0..6 {
            assert!((inv.get(i, i) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mat6_multiply_identity() {
        let a = Mat6::diagonal(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let i = Mat6::identity();
        let result = a.mul(&i);
        for idx in 0..6 {
            assert!((result.get(idx, idx) - a.get(idx, idx)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_invert_3x3() {
        let m = [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]];
        let inv = invert_3x3(&m);
        assert!(inv.is_some());
        let inv = inv.expect("should succeed in test");
        // Check M * M^{-1} ~ I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += m[i][k] * inv[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "M*M^-1 [{i}][{j}] = {sum}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_fused_orientation_fields() {
        let fo = FusedOrientation {
            timestamp_us: 1000,
            roll: 0.1,
            pitch: 0.2,
            yaw: 0.3,
            bias_roll: 0.001,
            bias_pitch: 0.002,
            bias_yaw: 0.003,
        };
        assert_eq!(fo.timestamp_us, 1000);
        assert!((fo.roll - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_ekf_fuse_convergence() {
        let mut ekf = GyroEkf::new();

        // Constant rotation of 10 deg/s yaw
        let rate = 10.0; // deg/s
        let rate_rad = rate * std::f64::consts::PI / 180.0;

        let mut stream = GyroStream::new(1000.0);
        let mut visuals = Vec::new();

        for i in 0..100u64 {
            let ts = i * 10_000; // 10ms
            stream.add_sample(gyro_sample(ts, 0.0, 0.0, rate));

            // Visual update every 5 frames
            if i % 5 == 0 {
                let angle = rate_rad * (ts as f64 * 1e-6);
                visuals.push(visual_est(ts, 0.0, 0.0, angle));
            }
        }

        let result = ekf.fuse(&stream, &visuals);
        assert!(result.is_ok());
        let orientations = result.expect("should succeed in test");

        // Final yaw should be close to the expected angle
        let final_ts = 99 * 10_000;
        let expected_yaw = rate_rad * (final_ts as f64 * 1e-6);
        let last = orientations.last().expect("should succeed in test");
        let error = (last.yaw - expected_yaw).abs();
        assert!(
            error < 0.3,
            "Final yaw error {error} rad (expected {expected_yaw}, got {})",
            last.yaw
        );
    }
}
