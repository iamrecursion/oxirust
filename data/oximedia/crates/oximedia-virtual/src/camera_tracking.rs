//! Camera tracking for virtual production.
//!
//! Provides low-pass filtering and pose interpolation for camera tracking
//! data used in LED volume and in-camera VFX workflows.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Tracked pose
// ---------------------------------------------------------------------------

/// A single tracked camera pose sample.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackedPose {
    /// X position in metres.
    pub x: f64,
    /// Y position in metres.
    pub y: f64,
    /// Z position in metres.
    pub z: f64,
    /// Rotation around X axis (pitch) in degrees.
    pub rx: f64,
    /// Rotation around Y axis (yaw) in degrees.
    pub ry: f64,
    /// Rotation around Z axis (roll) in degrees.
    pub rz: f64,
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl TrackedPose {
    /// Create a new pose at position `(x, y, z)` with zero rotation.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            timestamp_ms: 0,
        }
    }

    /// Euclidean distance to another pose (translation only).
    #[must_use]
    pub fn distance_to(&self, other: &TrackedPose) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Linearly interpolate between `self` and `other`.
    ///
    /// `t = 0.0` returns a copy of `self`; `t = 1.0` returns a copy of `other`.
    #[must_use]
    pub fn interpolate(&self, other: &TrackedPose, t: f64) -> TrackedPose {
        let lerp = |a: f64, b: f64| a + (b - a) * t;
        TrackedPose {
            x: lerp(self.x, other.x),
            y: lerp(self.y, other.y),
            z: lerp(self.z, other.z),
            rx: lerp(self.rx, other.rx),
            ry: lerp(self.ry, other.ry),
            rz: lerp(self.rz, other.rz),
            timestamp_ms: lerp(self.timestamp_ms as f64, other.timestamp_ms as f64) as u64,
        }
    }
}

// ---------------------------------------------------------------------------
// Low-pass filter helper
// ---------------------------------------------------------------------------

/// Single-pole low-pass filter.
///
/// `alpha` is in `[0, 1]`: 0 = no update, 1 = no filtering.
#[must_use]
pub fn low_pass_filter(current: f64, prev: f64, alpha: f64) -> f64 {
    alpha * current + (1.0 - alpha) * prev
}

// ---------------------------------------------------------------------------
// Camera tracker
// ---------------------------------------------------------------------------

/// Real-time camera tracker with low-pass filtering.
#[derive(Debug)]
pub struct CameraTracker {
    /// History of received poses (most recent last).
    pub poses: Vec<TrackedPose>,
    /// Low-pass filter coefficient in `[0, 1]`.
    pub filter_alpha: f64,
}

impl CameraTracker {
    /// Create a new tracker with the given filter coefficient.
    #[must_use]
    pub fn new(filter_alpha: f64) -> Self {
        Self {
            poses: Vec::new(),
            filter_alpha,
        }
    }

    /// Submit a new raw pose, apply smoothing, store, and return the filtered pose.
    pub fn update(&mut self, pose: TrackedPose) -> TrackedPose {
        let smoothed = self.smooth_pose(&pose);
        self.poses.push(smoothed.clone());
        smoothed
    }

    /// Return a smoothed version of `raw` based on the previous pose (if any).
    #[must_use]
    pub fn smooth_pose(&self, raw: &TrackedPose) -> TrackedPose {
        let alpha = self.filter_alpha;
        match self.poses.last() {
            None => raw.clone(),
            Some(prev) => TrackedPose {
                x: low_pass_filter(raw.x, prev.x, alpha),
                y: low_pass_filter(raw.y, prev.y, alpha),
                z: low_pass_filter(raw.z, prev.z, alpha),
                rx: low_pass_filter(raw.rx, prev.rx, alpha),
                ry: low_pass_filter(raw.ry, prev.ry, alpha),
                rz: low_pass_filter(raw.rz, prev.rz, alpha),
                timestamp_ms: raw.timestamp_ms,
            },
        }
    }

    /// Estimate instantaneous velocity `(vx, vy, vz)` in m/ms from the last two poses.
    ///
    /// Returns `None` if fewer than two poses have been recorded, or if the
    /// timestamps are identical.
    #[must_use]
    pub fn velocity(&self) -> Option<(f64, f64, f64)> {
        if self.poses.len() < 2 {
            return None;
        }
        let len = self.poses.len();
        let a = &self.poses[len - 2];
        let b = &self.poses[len - 1];
        let dt = b.timestamp_ms as f64 - a.timestamp_ms as f64;
        if dt == 0.0 {
            return None;
        }
        Some(((b.x - a.x) / dt, (b.y - a.y) / dt, (b.z - a.z) / dt))
    }

    /// Return a reference to the most recently recorded (filtered) pose.
    #[must_use]
    pub fn latest(&self) -> Option<&TrackedPose> {
        self.poses.last()
    }
}

// ---------------------------------------------------------------------------
// Frame prediction / motion extrapolation
// ---------------------------------------------------------------------------

/// Prediction model used for extrapolating camera motion forward in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionModel {
    /// Constant velocity: assumes the camera continues at its current velocity.
    ConstantVelocity,
    /// Constant acceleration: uses acceleration estimated from the last 3 poses.
    ConstantAcceleration,
    /// Quadratic extrapolation using a 3-point polynomial fit.
    Quadratic,
}

/// Configuration for the frame predictor.
#[derive(Debug, Clone)]
pub struct FramePredictorConfig {
    /// Prediction model to use.
    pub model: PredictionModel,
    /// Maximum number of poses to retain for prediction history.
    pub max_history: usize,
    /// Maximum prediction horizon in milliseconds. Predictions beyond this
    /// are clamped to avoid runaway extrapolation.
    pub max_prediction_ms: f64,
    /// Low-pass filter alpha applied to predicted velocity/acceleration.
    pub smoothing_alpha: f64,
}

impl Default for FramePredictorConfig {
    fn default() -> Self {
        Self {
            model: PredictionModel::ConstantVelocity,
            max_history: 32,
            max_prediction_ms: 50.0,
            smoothing_alpha: 0.7,
        }
    }
}

/// Frame predictor that extrapolates camera tracking data forward in time
/// to compensate for pipeline latency.
///
/// Uses recent pose history to estimate velocity (and optionally acceleration),
/// then projects the camera pose forward by the requested time offset.
#[derive(Debug)]
pub struct FramePredictor {
    config: FramePredictorConfig,
    /// Pose history (most recent last), kept trimmed to `max_history`.
    history: Vec<TrackedPose>,
    /// Smoothed velocity estimate (m/ms) for x, y, z.
    velocity: (f64, f64, f64),
    /// Smoothed acceleration estimate (m/ms^2) for x, y, z.
    acceleration: (f64, f64, f64),
    /// Smoothed angular velocity (deg/ms) for rx, ry, rz.
    angular_velocity: (f64, f64, f64),
}

impl FramePredictor {
    /// Create a new frame predictor.
    #[must_use]
    pub fn new(config: FramePredictorConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            velocity: (0.0, 0.0, 0.0),
            acceleration: (0.0, 0.0, 0.0),
            angular_velocity: (0.0, 0.0, 0.0),
        }
    }

    /// Feed a new tracked pose. Updates internal velocity/acceleration estimates.
    pub fn feed(&mut self, pose: TrackedPose) {
        self.history.push(pose);
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
        self.update_estimates();
    }

    /// Predict the pose `dt_ms` milliseconds into the future from the latest sample.
    ///
    /// Returns `None` if no poses have been fed yet.
    #[must_use]
    pub fn predict(&self, dt_ms: f64) -> Option<TrackedPose> {
        let latest = self.history.last()?;
        let dt = dt_ms.min(self.config.max_prediction_ms);

        let (vx, vy, vz) = self.velocity;
        let (avx, avy, avz) = self.angular_velocity;

        match self.config.model {
            PredictionModel::ConstantVelocity => Some(TrackedPose {
                x: latest.x + vx * dt,
                y: latest.y + vy * dt,
                z: latest.z + vz * dt,
                rx: latest.rx + avx * dt,
                ry: latest.ry + avy * dt,
                rz: latest.rz + avz * dt,
                timestamp_ms: latest.timestamp_ms + dt as u64,
            }),
            PredictionModel::ConstantAcceleration => {
                let (ax, ay, az) = self.acceleration;
                Some(TrackedPose {
                    x: latest.x + vx * dt + 0.5 * ax * dt * dt,
                    y: latest.y + vy * dt + 0.5 * ay * dt * dt,
                    z: latest.z + vz * dt + 0.5 * az * dt * dt,
                    rx: latest.rx + avx * dt,
                    ry: latest.ry + avy * dt,
                    rz: latest.rz + avz * dt,
                    timestamp_ms: latest.timestamp_ms + dt as u64,
                })
            }
            PredictionModel::Quadratic => self.predict_quadratic(dt),
        }
    }

    /// Get the current smoothed velocity estimate in m/ms.
    #[must_use]
    pub fn velocity(&self) -> (f64, f64, f64) {
        self.velocity
    }

    /// Get the current smoothed acceleration estimate in m/ms^2.
    #[must_use]
    pub fn acceleration(&self) -> (f64, f64, f64) {
        self.acceleration
    }

    /// Get the current smoothed angular velocity estimate in deg/ms.
    #[must_use]
    pub fn angular_velocity(&self) -> (f64, f64, f64) {
        self.angular_velocity
    }

    /// Number of poses in the history buffer.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &FramePredictorConfig {
        &self.config
    }

    /// Compute prediction quality: ratio of prediction error to actual
    /// displacement over the last few frames. Returns `None` if insufficient data.
    ///
    /// A value close to 0.0 means excellent prediction quality.
    #[must_use]
    pub fn prediction_quality(&self) -> Option<f64> {
        if self.history.len() < 4 {
            return None;
        }

        // Use the second-to-last pose to predict the last pose
        let n = self.history.len();
        let p_prev = &self.history[n - 2];
        let p_actual = &self.history[n - 1];
        let dt = p_actual.timestamp_ms as f64 - p_prev.timestamp_ms as f64;
        if dt <= 0.0 {
            return None;
        }

        // Simple constant-velocity prediction from n-2
        let (vx, vy, vz) = self.velocity;
        let pred_x = p_prev.x + vx * dt;
        let pred_y = p_prev.y + vy * dt;
        let pred_z = p_prev.z + vz * dt;

        let error = ((pred_x - p_actual.x).powi(2)
            + (pred_y - p_actual.y).powi(2)
            + (pred_z - p_actual.z).powi(2))
        .sqrt();

        let displacement = p_prev.distance_to(p_actual);
        if displacement < 1e-12 {
            return Some(0.0);
        }

        Some(error / displacement)
    }

    // -----------------------------------------------------------------------
    // Internal estimation
    // -----------------------------------------------------------------------

    fn update_estimates(&mut self) {
        let alpha = self.config.smoothing_alpha;
        let n = self.history.len();

        // Velocity from last two poses
        if n >= 2 {
            let a = &self.history[n - 2];
            let b = &self.history[n - 1];
            let dt = b.timestamp_ms as f64 - a.timestamp_ms as f64;
            if dt > 0.0 {
                let raw_vx = (b.x - a.x) / dt;
                let raw_vy = (b.y - a.y) / dt;
                let raw_vz = (b.z - a.z) / dt;

                self.velocity.0 = low_pass_filter(raw_vx, self.velocity.0, alpha);
                self.velocity.1 = low_pass_filter(raw_vy, self.velocity.1, alpha);
                self.velocity.2 = low_pass_filter(raw_vz, self.velocity.2, alpha);

                let raw_avx = (b.rx - a.rx) / dt;
                let raw_avy = (b.ry - a.ry) / dt;
                let raw_avz = (b.rz - a.rz) / dt;

                self.angular_velocity.0 = low_pass_filter(raw_avx, self.angular_velocity.0, alpha);
                self.angular_velocity.1 = low_pass_filter(raw_avy, self.angular_velocity.1, alpha);
                self.angular_velocity.2 = low_pass_filter(raw_avz, self.angular_velocity.2, alpha);
            }
        }

        // Acceleration from last three poses (finite difference of velocity)
        if n >= 3 {
            let a = &self.history[n - 3];
            let b = &self.history[n - 2];
            let c = &self.history[n - 1];
            let dt1 = b.timestamp_ms as f64 - a.timestamp_ms as f64;
            let dt2 = c.timestamp_ms as f64 - b.timestamp_ms as f64;
            if dt1 > 0.0 && dt2 > 0.0 {
                let v1x = (b.x - a.x) / dt1;
                let v2x = (c.x - b.x) / dt2;
                let v1y = (b.y - a.y) / dt1;
                let v2y = (c.y - b.y) / dt2;
                let v1z = (b.z - a.z) / dt1;
                let v2z = (c.z - b.z) / dt2;

                let dt_avg = (dt1 + dt2) * 0.5;
                let raw_ax = (v2x - v1x) / dt_avg;
                let raw_ay = (v2y - v1y) / dt_avg;
                let raw_az = (v2z - v1z) / dt_avg;

                self.acceleration.0 = low_pass_filter(raw_ax, self.acceleration.0, alpha);
                self.acceleration.1 = low_pass_filter(raw_ay, self.acceleration.1, alpha);
                self.acceleration.2 = low_pass_filter(raw_az, self.acceleration.2, alpha);
            }
        }
    }

    /// Quadratic extrapolation using the last 3 poses.
    fn predict_quadratic(&self, dt_ms: f64) -> Option<TrackedPose> {
        let n = self.history.len();
        if n < 3 {
            // Fall back to constant velocity
            let latest = self.history.last()?;
            let (vx, vy, vz) = self.velocity;
            let (avx, avy, avz) = self.angular_velocity;
            let dt = dt_ms.min(self.config.max_prediction_ms);
            return Some(TrackedPose {
                x: latest.x + vx * dt,
                y: latest.y + vy * dt,
                z: latest.z + vz * dt,
                rx: latest.rx + avx * dt,
                ry: latest.ry + avy * dt,
                rz: latest.rz + avz * dt,
                timestamp_ms: latest.timestamp_ms + dt as u64,
            });
        }

        let p0 = &self.history[n - 3];
        let p1 = &self.history[n - 2];
        let p2 = &self.history[n - 1];

        // Use p2 as the reference point (t=0)
        let t0 = p0.timestamp_ms as f64 - p2.timestamp_ms as f64;
        let t1 = p1.timestamp_ms as f64 - p2.timestamp_ms as f64;
        let dt = dt_ms.min(self.config.max_prediction_ms);

        let extrapolate = |y0: f64, y1: f64, y2: f64| -> f64 {
            // Lagrange interpolation through (t0, y0), (t1, y1), (0, y2)
            // evaluated at t = dt
            let denom01 = t0 - t1;
            let denom02 = t0; // t0 - 0
            let denom12 = t1; // t1 - 0

            // Guard against zero denominators
            if denom01.abs() < 1e-15 || denom02.abs() < 1e-15 || denom12.abs() < 1e-15 {
                return y2; // degenerate: just return latest
            }

            let l0 = (dt - t1) * dt / (denom01 * denom02);
            let l1 = (dt - t0) * dt / ((-denom01) * denom12);
            let l2 = (dt - t0) * (dt - t1) / (denom02 * denom12);

            l0 * y0 + l1 * y1 + l2 * y2
        };

        Some(TrackedPose {
            x: extrapolate(p0.x, p1.x, p2.x),
            y: extrapolate(p0.y, p1.y, p2.y),
            z: extrapolate(p0.z, p1.z, p2.z),
            rx: extrapolate(p0.rx, p1.rx, p2.rx),
            ry: extrapolate(p0.ry, p1.ry, p2.ry),
            rz: extrapolate(p0.rz, p1.rz, p2.rz),
            timestamp_ms: p2.timestamp_ms + dt as u64,
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracked_pose_new() {
        let p = TrackedPose::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
        assert_eq!(p.rx, 0.0);
        assert_eq!(p.timestamp_ms, 0);
    }

    #[test]
    fn test_tracked_pose_distance_to_same() {
        let p = TrackedPose::new(0.0, 0.0, 0.0);
        assert_eq!(p.distance_to(&p), 0.0);
    }

    #[test]
    fn test_tracked_pose_distance_to_unit() {
        let a = TrackedPose::new(0.0, 0.0, 0.0);
        let b = TrackedPose::new(1.0, 0.0, 0.0);
        assert!((a.distance_to(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracked_pose_distance_to_3d() {
        let a = TrackedPose::new(0.0, 0.0, 0.0);
        let b = TrackedPose::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracked_pose_interpolate_t0() {
        let a = TrackedPose::new(0.0, 0.0, 0.0);
        let b = TrackedPose::new(10.0, 10.0, 10.0);
        let mid = a.interpolate(&b, 0.0);
        assert!((mid.x - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracked_pose_interpolate_t1() {
        let a = TrackedPose::new(0.0, 0.0, 0.0);
        let b = TrackedPose::new(10.0, 10.0, 10.0);
        let mid = a.interpolate(&b, 1.0);
        assert!((mid.x - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracked_pose_interpolate_midpoint() {
        let a = TrackedPose::new(0.0, 0.0, 0.0);
        let b = TrackedPose::new(10.0, 20.0, 30.0);
        let mid = a.interpolate(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
        assert!((mid.y - 10.0).abs() < 1e-10);
        assert!((mid.z - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_low_pass_filter_alpha_one() {
        // alpha=1 means no filtering
        assert_eq!(low_pass_filter(5.0, 0.0, 1.0), 5.0);
    }

    #[test]
    fn test_low_pass_filter_alpha_zero() {
        // alpha=0 means no update
        assert_eq!(low_pass_filter(5.0, 3.0, 0.0), 3.0);
    }

    #[test]
    fn test_low_pass_filter_half() {
        let v = low_pass_filter(10.0, 0.0, 0.5);
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_tracker_no_poses_initially() {
        let t = CameraTracker::new(0.9);
        assert!(t.latest().is_none());
        assert!(t.velocity().is_none());
    }

    #[test]
    fn test_camera_tracker_update_stores_pose() {
        let mut t = CameraTracker::new(1.0); // alpha=1 → no filtering
        let p = TrackedPose::new(1.0, 2.0, 3.0);
        t.update(p);
        let latest = t.latest().expect("should succeed in test");
        assert!((latest.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_tracker_velocity() {
        let mut t = CameraTracker::new(1.0);
        let mut p1 = TrackedPose::new(0.0, 0.0, 0.0);
        p1.timestamp_ms = 0;
        let mut p2 = TrackedPose::new(1.0, 0.0, 0.0);
        p2.timestamp_ms = 1000;
        t.update(p1);
        t.update(p2);
        let (vx, vy, vz) = t.velocity().expect("should succeed in test");
        assert!((vx - 0.001).abs() < 1e-10); // 1m / 1000ms
        assert_eq!(vy, 0.0);
        assert_eq!(vz, 0.0);
    }

    // --- Frame predictor tests ---

    fn make_linear_poses(
        start_x: f64,
        velocity: f64,
        count: usize,
        dt_ms: u64,
    ) -> Vec<TrackedPose> {
        (0..count)
            .map(|i| {
                let mut p =
                    TrackedPose::new(start_x + velocity * (i as f64 * dt_ms as f64), 0.0, 0.0);
                p.timestamp_ms = i as u64 * dt_ms;
                p
            })
            .collect()
    }

    #[test]
    fn test_frame_predictor_creation() {
        let fp = FramePredictor::new(FramePredictorConfig::default());
        assert_eq!(fp.history_len(), 0);
        assert!(fp.predict(10.0).is_none());
    }

    #[test]
    fn test_frame_predictor_constant_velocity_linear_motion() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            model: PredictionModel::ConstantVelocity,
            smoothing_alpha: 1.0, // no smoothing for clarity
            ..FramePredictorConfig::default()
        });

        // Camera moving at 0.001 m/ms (1 m/s) along X
        let poses = make_linear_poses(0.0, 0.001, 5, 10);
        for p in poses {
            fp.feed(p);
        }

        // Predict 10ms ahead: latest x = 0.04, velocity = 0.001 m/ms
        // => predicted x = 0.04 + 0.001 * 10 = 0.05
        let predicted = fp.predict(10.0).expect("should succeed in test");
        assert!(
            (predicted.x - 0.05).abs() < 1e-6,
            "predicted x: {}",
            predicted.x
        );
    }

    #[test]
    fn test_frame_predictor_constant_acceleration() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            model: PredictionModel::ConstantAcceleration,
            smoothing_alpha: 1.0,
            ..FramePredictorConfig::default()
        });

        // Accelerating motion: x = 0.5 * a * t^2, a = 0.0001 m/ms^2
        // t=0: x=0, t=10: x=0.005, t=20: x=0.02, t=30: x=0.045, t=40: x=0.08
        let a = 0.0001;
        for i in 0..5 {
            let t = i as f64 * 10.0;
            let mut p = TrackedPose::new(0.5 * a * t * t, 0.0, 0.0);
            p.timestamp_ms = t as u64;
            fp.feed(p);
        }

        // Predict 10ms from t=40 (x=0.08)
        // Velocity at t=40 ≈ a*40 = 0.004 m/ms
        // Predicted x ≈ 0.08 + 0.004*10 + 0.5*0.0001*100 = 0.08 + 0.04 + 0.005 = 0.125
        // (exact: 0.5 * 0.0001 * 50^2 = 0.125)
        let predicted = fp.predict(10.0).expect("should succeed in test");
        assert!(
            (predicted.x - 0.125).abs() < 0.02,
            "acceleration predicted x: {}",
            predicted.x
        );
    }

    #[test]
    fn test_frame_predictor_quadratic_model() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            model: PredictionModel::Quadratic,
            smoothing_alpha: 1.0,
            ..FramePredictorConfig::default()
        });

        // Quadratic motion: x(t) = 0.001 * t^2
        for i in 0..5 {
            let t = i as f64 * 10.0;
            let mut p = TrackedPose::new(0.001 * t * t, 0.0, 0.0);
            p.timestamp_ms = t as u64;
            fp.feed(p);
        }

        // Predict 10ms ahead from t=40 (x=1.6)
        // True x(50) = 0.001 * 2500 = 2.5
        let predicted = fp.predict(10.0).expect("should succeed in test");
        assert!(
            (predicted.x - 2.5).abs() < 0.1,
            "quadratic predicted x: {}",
            predicted.x
        );
    }

    #[test]
    fn test_frame_predictor_max_prediction_clamped() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            model: PredictionModel::ConstantVelocity,
            max_prediction_ms: 20.0,
            smoothing_alpha: 1.0,
            ..FramePredictorConfig::default()
        });

        let poses = make_linear_poses(0.0, 0.001, 3, 10);
        for p in poses {
            fp.feed(p);
        }

        // Request 100ms prediction, but should be clamped to 20ms
        let predicted = fp.predict(100.0).expect("should succeed in test");
        let expected = 0.02 + 0.001 * 20.0; // latest x + v * 20ms
        assert!(
            (predicted.x - expected).abs() < 1e-6,
            "clamped prediction: {} vs {}",
            predicted.x,
            expected
        );
    }

    #[test]
    fn test_frame_predictor_rotation_prediction() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            model: PredictionModel::ConstantVelocity,
            smoothing_alpha: 1.0,
            ..FramePredictorConfig::default()
        });

        // Camera rotating at 0.1 deg/ms around Y
        for i in 0..5 {
            let t = i as f64 * 10.0;
            let mut p = TrackedPose::new(0.0, 0.0, 0.0);
            p.ry = 0.1 * t;
            p.timestamp_ms = t as u64;
            fp.feed(p);
        }

        let predicted = fp.predict(10.0).expect("should succeed in test");
        // latest ry = 4.0, angular vel = 0.1 deg/ms => predicted ry = 5.0
        assert!(
            (predicted.ry - 5.0).abs() < 0.01,
            "predicted ry: {}",
            predicted.ry
        );
    }

    #[test]
    fn test_frame_predictor_velocity_estimation() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            smoothing_alpha: 1.0,
            ..FramePredictorConfig::default()
        });

        let poses = make_linear_poses(0.0, 0.002, 5, 10);
        for p in poses {
            fp.feed(p);
        }

        let (vx, vy, vz) = fp.velocity();
        assert!((vx - 0.002).abs() < 1e-6, "velocity x: {}", vx);
        assert!(vy.abs() < 1e-10);
        assert!(vz.abs() < 1e-10);
    }

    #[test]
    fn test_frame_predictor_prediction_quality() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            smoothing_alpha: 1.0,
            ..FramePredictorConfig::default()
        });

        // Perfect linear motion should have near-zero prediction error
        let poses = make_linear_poses(0.0, 0.001, 10, 10);
        for p in poses {
            fp.feed(p);
        }

        let quality = fp.prediction_quality().expect("should have enough data");
        assert!(
            quality < 0.01,
            "linear motion prediction quality should be near zero: {}",
            quality
        );
    }

    #[test]
    fn test_frame_predictor_no_data() {
        let fp = FramePredictor::new(FramePredictorConfig::default());
        assert!(fp.predict(10.0).is_none());
        assert!(fp.prediction_quality().is_none());
    }

    #[test]
    fn test_frame_predictor_single_pose() {
        let mut fp = FramePredictor::new(FramePredictorConfig::default());
        fp.feed(TrackedPose::new(1.0, 2.0, 3.0));

        // With only one pose, velocity is zero, so prediction = latest
        let predicted = fp.predict(10.0).expect("should succeed in test");
        assert!((predicted.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frame_predictor_history_trimming() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            max_history: 4,
            ..FramePredictorConfig::default()
        });

        for i in 0..10 {
            let mut p = TrackedPose::new(i as f64, 0.0, 0.0);
            p.timestamp_ms = i * 100;
            fp.feed(p);
        }

        assert_eq!(fp.history_len(), 4);
    }

    #[test]
    fn test_frame_predictor_quadratic_fallback_with_few_poses() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            model: PredictionModel::Quadratic,
            smoothing_alpha: 1.0,
            ..FramePredictorConfig::default()
        });

        // Only 2 poses: should fall back to constant velocity
        let mut p1 = TrackedPose::new(0.0, 0.0, 0.0);
        p1.timestamp_ms = 0;
        let mut p2 = TrackedPose::new(1.0, 0.0, 0.0);
        p2.timestamp_ms = 10;
        fp.feed(p1);
        fp.feed(p2);

        let predicted = fp.predict(10.0).expect("should succeed in test");
        assert!(
            (predicted.x - 2.0).abs() < 0.01,
            "fallback prediction: {}",
            predicted.x
        );
    }

    #[test]
    fn test_frame_predictor_stationary_camera() {
        let mut fp = FramePredictor::new(FramePredictorConfig {
            smoothing_alpha: 1.0,
            ..FramePredictorConfig::default()
        });

        for i in 0..5 {
            let mut p = TrackedPose::new(5.0, 3.0, -2.0);
            p.timestamp_ms = i * 10;
            fp.feed(p);
        }

        let predicted = fp.predict(10.0).expect("should succeed in test");
        assert!((predicted.x - 5.0).abs() < 1e-10);
        assert!((predicted.y - 3.0).abs() < 1e-10);
        assert!((predicted.z - (-2.0)).abs() < 1e-10);
    }
}
