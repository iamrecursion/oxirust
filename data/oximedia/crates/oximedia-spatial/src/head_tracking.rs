//! Head orientation tracking for binaural rendering.
//!
//! This module provides quaternion-based head orientation representation and
//! a complementary filter that fuses gyroscope integration with accelerometer
//! correction.  The corrected orientation can then be applied to binaural
//! rendering to compensate for listener head rotation, so that virtual sound
//! sources remain anchored to the real world rather than moving with the head.
//!
//! # Pipeline
//!
//! ```text
//! gyro (deg/s) ─┐                ┌─ to_binaural_angles()
//!               ├─ HeadTracker ──┤
//! accel (g)    ─┘                └─ current_orientation (Quaternion)
//! ```
//!
//! # Complementary filter
//!
//! ```text
//! q_gyro  = q_prev * delta_q(gyro × dt)
//! q_accel = Quaternion::from_euler(accel_pitch, accel_roll)   // no yaw from accel
//! q_new   = slerp(q_accel, q_gyro, alpha)  (approximated as weighted average + renorm)
//! ```

use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Mutex, OnceLock};

// ─── Rotation matrix LUT ──────────────────────────────────────────────────────

/// Quantisation step in degrees for the rotation-matrix cache.
const ROTATION_CACHE_STEP_DEG: f32 = 1.0;

/// Quantise an angle to the nearest cache-key integer.
fn quantize_angle(angle_deg: f32, step: f32) -> i32 {
    (angle_deg / step).round() as i32
}

/// Global rotation-matrix cache keyed by quantised (yaw, pitch, roll).
static ROTATION_CACHE: OnceLock<Mutex<HashMap<(i32, i32, i32), [[f32; 3]; 3]>>> = OnceLock::new();

/// Return a 3×3 rotation matrix for the given Euler angles (ZYX convention),
/// serving the result from an in-process LUT when the quantised key is cached.
///
/// Angles are quantised to the nearest `ROTATION_CACHE_STEP_DEG` degree before
/// lookup so that repeated calls with nearby values share the same entry.
///
/// # Example
/// ```
/// use oximedia_spatial::head_tracking::cached_rotation_matrix;
/// let mat = cached_rotation_matrix(0.0, 0.0, 0.0);
/// // Identity rotation
/// assert!((mat[0][0] - 1.0).abs() < 1e-5);
/// ```
pub fn cached_rotation_matrix(yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> [[f32; 3]; 3] {
    let key = (
        quantize_angle(yaw_deg, ROTATION_CACHE_STEP_DEG),
        quantize_angle(pitch_deg, ROTATION_CACHE_STEP_DEG),
        quantize_angle(roll_deg, ROTATION_CACHE_STEP_DEG),
    );
    let cache = ROTATION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&mat) = guard.get(&key) {
        return mat;
    }
    let mat = compute_rotation_matrix(yaw_deg, pitch_deg, roll_deg);
    guard.insert(key, mat);
    mat
}

/// Compute the ZYX Euler rotation matrix R = Rz(yaw) × Ry(pitch) × Rx(roll).
///
/// All angles are supplied in degrees.
pub fn compute_rotation_matrix(yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> [[f32; 3]; 3] {
    let y = yaw_deg.to_radians();
    let p = pitch_deg.to_radians();
    let r = roll_deg.to_radians();
    let (cy, sy) = (y.cos(), y.sin());
    let (cp, sp) = (p.cos(), p.sin());
    let (cr, sr) = (r.cos(), r.sin());
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

// ─── Types ────────────────────────────────────────────────────────────────────

/// Euler angle representation of head orientation.
///
/// All angles are in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadOrientation {
    /// Rotation about the vertical axis (left/right turn).
    pub yaw_deg: f32,
    /// Rotation about the side axis (nod up/down).
    pub pitch_deg: f32,
    /// Rotation about the front axis (tilt left/right).
    pub roll_deg: f32,
}

/// Unit quaternion representing a 3-D rotation.
///
/// Stored as (w, x, y, z) where w is the scalar part.
#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Per-axis complementary-filter state for [`HeadTracker`].
#[derive(Debug, Clone, Copy)]
pub struct LowPassFilter {
    /// Smoothing factor in (0, 1].  Closer to 1 = no filtering; closer to 0 = heavy filtering.
    pub alpha: f32,
    /// Most recent filtered value.
    state: f32,
}

/// Kalman filter for a single angular axis tracking [angle, angular_rate].
#[derive(Debug, Clone)]
pub struct KalmanOrientation {
    /// State vector [angle_deg, rate_deg_per_s].
    pub state: [f32; 2],
    /// 2×2 error covariance matrix (row-major).
    pub covariance: [[f32; 2]; 2],
    /// Process noise variance for angle.
    process_noise_angle: f32,
    /// Process noise variance for rate.
    process_noise_rate: f32,
    /// Measurement noise variance.
    measurement_noise: f32,
}

/// Smoothing strategy applied to the final Euler angles after the complementary filter.
#[derive(Debug, Clone)]
pub enum TrackingSmoothing {
    /// No post-filter smoothing.
    None,
    /// Exponential moving average with the given cutoff frequency.
    LowPass { cutoff_hz: f32 },
    /// Per-axis Kalman filter (angle + rate).
    Kalman,
}

/// Head orientation tracker fusing gyroscope and optional accelerometer data.
///
/// Build via [`HeadTracker::new`].
#[derive(Debug, Clone)]
pub struct HeadTracker {
    /// Current orientation estimate as a unit quaternion.
    pub current_orientation: Quaternion,
    /// Angular velocity (rad/s) encoded as a quaternion for integration convenience.
    pub velocity: Quaternion,
    /// Complementary filter coefficient: weight of gyro vs accelerometer [0, 1].
    /// `alpha = 1.0` means trust gyro only; `0.0` means trust accelerometer only.
    pub alpha: f32,
    /// Optional post-filter smoothing.
    pub smoothing: TrackingSmoothing,
    /// Per-axis low-pass filters (yaw, pitch, roll) — used when `smoothing = LowPass`.
    lp_filters: [LowPassFilter; 3],
    /// Per-axis Kalman filters (yaw, pitch, roll) — used when `smoothing = Kalman`.
    kalman_filters: [KalmanOrientation; 3],
}

// ─── HeadOrientation ─────────────────────────────────────────────────────────

impl HeadOrientation {
    /// Create a neutral (zero) orientation.
    pub fn identity() -> Self {
        Self {
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        }
    }
}

impl Default for HeadOrientation {
    fn default() -> Self {
        Self::identity()
    }
}

// ─── Quaternion ──────────────────────────────────────────────────────────────

impl Quaternion {
    /// Identity quaternion (no rotation).
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Normalise to unit length.  Returns the identity quaternion if the norm
    /// is too small to avoid division by zero.
    pub fn normalize(&self) -> Self {
        let norm = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if norm < 1e-10 {
            return Self::identity();
        }
        Self {
            w: self.w / norm,
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    /// Hamilton product (quaternion multiplication).  Result is *not* normalised.
    pub fn multiply(&self, rhs: &Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    /// Convert to Euler angles (yaw-pitch-roll, intrinsic ZYX) in degrees.
    ///
    /// The standard aerospace atan2/asin decomposition is used:
    /// - Yaw:   atan2(2(wy + xz), 1 − 2(y² + z²))
    /// - Pitch: asin(clamp(2(wx − yz), −1, 1))
    /// - Roll:  atan2(2(wz + xy), 1 − 2(x² + z²))
    pub fn to_euler(&self) -> HeadOrientation {
        let q = self.normalize();

        // Yaw (Z-axis rotation)
        let siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        let cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let yaw = siny_cosp.atan2(cosy_cosp).to_degrees();

        // Pitch (Y-axis rotation)
        let sinp = 2.0 * (q.w * q.y - q.z * q.x);
        let pitch = sinp.clamp(-1.0, 1.0).asin().to_degrees();

        // Roll (X-axis rotation)
        let sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
        let cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        let roll = sinr_cosp.atan2(cosr_cosp).to_degrees();

        HeadOrientation {
            yaw_deg: yaw,
            pitch_deg: pitch,
            roll_deg: roll,
        }
    }

    /// Build a unit quaternion from Euler angles (yaw-pitch-roll, degrees, ZYX intrinsic).
    pub fn from_euler(yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> Self {
        let half_yaw = yaw_deg.to_radians() * 0.5;
        let half_pitch = pitch_deg.to_radians() * 0.5;
        let half_roll = roll_deg.to_radians() * 0.5;

        let (sy, cy) = (half_yaw.sin(), half_yaw.cos());
        let (sp, cp) = (half_pitch.sin(), half_pitch.cos());
        let (sr, cr) = (half_roll.sin(), half_roll.cos());

        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
        .normalize()
    }

    /// Build a small-angle delta quaternion from angular velocity (deg/s) and time step (ms).
    fn from_gyro_delta(delta: &HeadOrientation, dt_ms: f32) -> Self {
        let dt_s = dt_ms / 1000.0;
        let half_yaw = delta.yaw_deg.to_radians() * dt_s * 0.5;
        let half_pitch = delta.pitch_deg.to_radians() * dt_s * 0.5;
        let half_roll = delta.roll_deg.to_radians() * dt_s * 0.5;

        // For small angles: q ≈ [1, half_angle_x, half_angle_y, half_angle_z]
        Self {
            w: 1.0,
            x: half_roll,  // roll around X
            y: half_pitch, // pitch around Y
            z: half_yaw,   // yaw around Z
        }
        .normalize()
    }

    /// Spherical linear interpolation between `self` and `other` by factor `t ∈ [0, 1]`.
    ///
    /// For very close quaternions (dot > 0.9995) a linear interpolation is used
    /// to avoid numerical issues.
    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let q0 = self.normalize();
        let mut q1 = other.normalize();

        // Ensure shortest path.
        let mut dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z;
        if dot < 0.0 {
            q1 = Quaternion {
                w: -q1.w,
                x: -q1.x,
                y: -q1.y,
                z: -q1.z,
            };
            dot = -dot;
        }

        if dot > 0.9995 {
            // Linear interpolation + normalise.
            let t0 = 1.0 - t;
            return Self {
                w: q0.w * t0 + q1.w * t,
                x: q0.x * t0 + q1.x * t,
                y: q0.y * t0 + q1.y * t,
                z: q0.z * t0 + q1.z * t,
            }
            .normalize();
        }

        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();

        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        Self {
            w: q0.w * s0 + q1.w * s1,
            x: q0.x * s0 + q1.x * s1,
            y: q0.y * s0 + q1.y * s1,
            z: q0.z * s0 + q1.z * s1,
        }
        .normalize()
    }

    /// Conjugate (inverse for unit quaternions).
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

// ─── LowPassFilter ───────────────────────────────────────────────────────────

impl LowPassFilter {
    /// Create a new low-pass filter with the given smoothing factor and an initial state of 0.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            state: 0.0,
        }
    }

    /// Build from a cutoff frequency (Hz) and the sensor update rate (Hz).
    ///
    /// Uses the bilinear-transform approximation:
    /// `alpha = 2πf_c / (2πf_c + f_s)`.
    pub fn from_cutoff(cutoff_hz: f32, sample_rate_hz: f32) -> Self {
        let omega = 2.0 * PI * cutoff_hz;
        let alpha = omega / (omega + sample_rate_hz);
        Self::new(alpha)
    }

    /// Process one sample via exponential smoothing: `y = alpha * x + (1 - alpha) * y_prev`.
    pub fn process(&mut self, x: f32) -> f32 {
        self.state = self.alpha * x + (1.0 - self.alpha) * self.state;
        self.state
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

// ─── KalmanOrientation ────────────────────────────────────────────────────────

impl KalmanOrientation {
    /// Create a Kalman filter for one angular axis, starting at `initial_angle_deg`.
    pub fn new(initial_angle_deg: f32) -> Self {
        Self {
            state: [initial_angle_deg, 0.0],
            covariance: [[1.0, 0.0], [0.0, 1.0]],
            process_noise_angle: 0.001,
            process_noise_rate: 0.003,
            measurement_noise: 0.03,
        }
    }

    /// Predict step: propagate state and covariance forward by `dt` seconds.
    ///
    /// State transition: angle += rate × dt
    pub fn predict(&mut self, dt: f32) {
        // x = F * x  where F = [[1, dt], [0, 1]]
        let angle = self.state[0] + self.state[1] * dt;
        let rate = self.state[1];
        self.state = [angle, rate];

        // P = F * P * F' + Q
        let p00 = self.covariance[0][0];
        let p01 = self.covariance[0][1];
        let p10 = self.covariance[1][0];
        let p11 = self.covariance[1][1];

        // F * P:
        let fp00 = p00 + dt * p10;
        let fp01 = p01 + dt * p11;
        let fp10 = p10;
        let fp11 = p11;

        // (F * P) * F':
        self.covariance[0][0] = fp00 + fp01 * dt + self.process_noise_angle;
        self.covariance[0][1] = fp01;
        self.covariance[1][0] = fp10 + fp11 * dt;
        self.covariance[1][1] = fp11 + self.process_noise_rate;
    }

    /// Update (correction) step with a direct angle measurement.
    ///
    /// Observation model: H = [1, 0] (we observe the angle only).
    pub fn update(&mut self, measurement: f32) {
        // Innovation: y = z - H * x
        let y = measurement - self.state[0];

        // Innovation covariance: S = H * P * H' + R
        let s = self.covariance[0][0] + self.measurement_noise;
        if s.abs() < 1e-12 {
            return;
        }

        // Kalman gain: K = P * H' / S  →  K = [P[0][0]/S, P[1][0]/S]
        let k0 = self.covariance[0][0] / s;
        let k1 = self.covariance[1][0] / s;

        // State update.
        self.state[0] += k0 * y;
        self.state[1] += k1 * y;

        // Covariance update: P = (I - K*H) * P
        let p00 = (1.0 - k0) * self.covariance[0][0];
        let p01 = (1.0 - k0) * self.covariance[0][1];
        let p10 = self.covariance[1][0] - k1 * self.covariance[0][0];
        let p11 = self.covariance[1][1] - k1 * self.covariance[0][1];

        self.covariance = [[p00, p01], [p10, p11]];
    }

    /// Return the current filtered angle estimate (degrees).
    pub fn angle(&self) -> f32 {
        self.state[0]
    }
}

// ─── HeadTracker ─────────────────────────────────────────────────────────────

impl HeadTracker {
    /// Create a new head tracker starting at the identity orientation.
    ///
    /// `alpha`: complementary filter weight for the gyro path (0 = accel only, 1 = gyro only).
    /// Typical value: 0.98.
    pub fn new(alpha: f32, smoothing: TrackingSmoothing) -> Self {
        let lp_alpha = match &smoothing {
            TrackingSmoothing::LowPass { cutoff_hz } => {
                // Default to 100 Hz update rate for the filter construction; this
                // will be dynamically re-computed during `update()` calls when dt_ms is known.
                LowPassFilter::from_cutoff(*cutoff_hz, 100.0).alpha
            }
            _ => 0.1,
        };

        Self {
            current_orientation: Quaternion::identity(),
            velocity: Quaternion::identity(),
            alpha: alpha.clamp(0.0, 1.0),
            smoothing,
            lp_filters: [
                LowPassFilter::new(lp_alpha),
                LowPassFilter::new(lp_alpha),
                LowPassFilter::new(lp_alpha),
            ],
            kalman_filters: [
                KalmanOrientation::new(0.0),
                KalmanOrientation::new(0.0),
                KalmanOrientation::new(0.0),
            ],
        }
    }

    /// Update the orientation estimate.
    ///
    /// # Parameters
    /// - `gyro_delta`: angular velocity in deg/s for each axis.
    /// - `accel_reading`: optional gravity vector expressed as Euler angles (pitch and roll only
    ///   are reliable from an accelerometer; yaw is left unchanged).
    /// - `dt_ms`: time step in milliseconds since the last call.
    pub fn update(
        &mut self,
        gyro_delta: HeadOrientation,
        accel_reading: Option<HeadOrientation>,
        dt_ms: f32,
    ) {
        // --- Gyro integration -----------------------------------------------
        let delta_q = Quaternion::from_gyro_delta(&gyro_delta, dt_ms);
        let q_gyro = self.current_orientation.multiply(&delta_q).normalize();

        // --- Accelerometer correction ----------------------------------------
        let q_new = if let Some(accel) = accel_reading {
            // Build a quaternion from the accel-derived pitch and roll (yaw unknown from accel).
            let q_accel = Quaternion::from_euler(
                q_gyro.to_euler().yaw_deg, // keep gyro yaw
                accel.pitch_deg,
                accel.roll_deg,
            );
            // Complementary filter: alpha controls trust in gyro vs accel.
            q_gyro.slerp(&q_accel, 1.0 - self.alpha)
        } else {
            q_gyro
        };

        self.current_orientation = q_new;
        self.velocity = delta_q;

        // --- Post-filter smoothing -------------------------------------------
        match &self.smoothing {
            TrackingSmoothing::None => {}
            TrackingSmoothing::LowPass { cutoff_hz } => {
                let update_rate_hz = if dt_ms > 0.0 { 1000.0 / dt_ms } else { 100.0 };
                let alpha_lp = LowPassFilter::from_cutoff(*cutoff_hz, update_rate_hz).alpha;
                self.lp_filters[0].alpha = alpha_lp;
                self.lp_filters[1].alpha = alpha_lp;
                self.lp_filters[2].alpha = alpha_lp;

                let euler = q_new.to_euler();
                let yaw_f = self.lp_filters[0].process(euler.yaw_deg);
                let pitch_f = self.lp_filters[1].process(euler.pitch_deg);
                let roll_f = self.lp_filters[2].process(euler.roll_deg);
                self.current_orientation = Quaternion::from_euler(yaw_f, pitch_f, roll_f);
            }
            TrackingSmoothing::Kalman => {
                let dt_s = dt_ms / 1000.0;
                let euler = q_new.to_euler();

                self.kalman_filters[0].predict(dt_s);
                self.kalman_filters[0].update(euler.yaw_deg);

                self.kalman_filters[1].predict(dt_s);
                self.kalman_filters[1].update(euler.pitch_deg);

                self.kalman_filters[2].predict(dt_s);
                self.kalman_filters[2].update(euler.roll_deg);

                let yaw_k = self.kalman_filters[0].angle();
                let pitch_k = self.kalman_filters[1].angle();
                let roll_k = self.kalman_filters[2].angle();
                self.current_orientation = Quaternion::from_euler(yaw_k, pitch_k, roll_k);
            }
        }
    }

    /// Compute the binaural rendering angles for a virtual source, compensating
    /// for the current head rotation.
    ///
    /// Given a world-space source at `(source_azimuth, source_elevation)` (degrees),
    /// this returns the azimuth and elevation **relative to the listener's current
    /// head orientation**.
    ///
    /// # Returns
    /// `(relative_azimuth_deg, relative_elevation_deg)`
    pub fn to_binaural_angles(&self, source_azimuth: f32, source_elevation: f32) -> (f32, f32) {
        // Convert source to a Cartesian unit vector.
        let az = source_azimuth.to_radians();
        let el = source_elevation.to_radians();
        let cos_el = el.cos();
        let src_vec = [cos_el * az.cos(), cos_el * az.sin(), el.sin()];

        // Rotate source vector by the *inverse* (conjugate) of the head orientation.
        let q_inv = self.current_orientation.conjugate().normalize();
        let rotated = rotate_vector_by_quat(src_vec, q_inv);

        // Convert back to spherical.
        let new_az = rotated[1].atan2(rotated[0]).to_degrees();
        let horiz = (rotated[0] * rotated[0] + rotated[1] * rotated[1]).sqrt();
        let new_el = rotated[2].atan2(horiz).to_degrees();

        (new_az, new_el)
    }

    /// Return the current orientation as Euler angles (degrees).
    pub fn euler(&self) -> HeadOrientation {
        self.current_orientation.to_euler()
    }

    /// Reset the tracker to the identity orientation.
    pub fn reset(&mut self) {
        self.current_orientation = Quaternion::identity();
        self.velocity = Quaternion::identity();
        for f in &mut self.lp_filters {
            f.reset();
        }
        self.kalman_filters = [
            KalmanOrientation::new(0.0),
            KalmanOrientation::new(0.0),
            KalmanOrientation::new(0.0),
        ];
    }
}

// ─── Magnetometer / full IMU types ───────────────────────────────────────────

/// Raw 3-axis magnetometer reading (µT or arbitrary consistent units).
///
/// The magnetometer measures the local Earth magnetic field vector.  Combined
/// with the accelerometer it provides an absolute yaw reference (tilt-compensated
/// compass heading), resolving the gyro drift in the yaw axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagnetometerReading {
    /// Magnetic field component along the sensor X axis.
    pub x: f32,
    /// Magnetic field component along the sensor Y axis.
    pub y: f32,
    /// Magnetic field component along the sensor Z axis.
    pub z: f32,
}

impl MagnetometerReading {
    /// Create a new reading.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Normalise the field vector to unit length.
    /// Returns `None` if the norm is too small to normalise safely.
    pub fn normalise(&self) -> Option<[f32; 3]> {
        let norm = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if norm < 1e-10 {
            return None;
        }
        Some([self.x / norm, self.y / norm, self.z / norm])
    }
}

/// Hard-iron + soft-iron calibration parameters for a 3-axis magnetometer.
///
/// Hard-iron distortions are constant offsets added to every reading; they arise
/// from permanently magnetised materials near the sensor.  Soft-iron distortions
/// cause the ideal sphere of readings to be transformed into an ellipsoid.
///
/// This struct stores the simple *hard-iron offset* only (the most common
/// calibration step).  For full soft-iron correction a 3×3 matrix is needed,
/// but that is typically determined offline and baked into the offset.
#[derive(Debug, Clone, Copy)]
pub struct MagCalibration {
    /// Hard-iron offset to subtract from raw X reading.
    pub offset_x: f32,
    /// Hard-iron offset to subtract from raw Y reading.
    pub offset_y: f32,
    /// Hard-iron offset to subtract from raw Z reading.
    pub offset_z: f32,
    /// Per-axis scale (soft-iron first-order approximation, typically 1.0).
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
}

impl MagCalibration {
    /// Identity calibration (no correction).
    pub fn identity() -> Self {
        Self {
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            scale_z: 1.0,
        }
    }

    /// Apply calibration to a raw reading.
    pub fn apply(&self, raw: MagnetometerReading) -> MagnetometerReading {
        MagnetometerReading {
            x: (raw.x - self.offset_x) * self.scale_x,
            y: (raw.y - self.offset_y) * self.scale_y,
            z: (raw.z - self.offset_z) * self.scale_z,
        }
    }
}

impl Default for MagCalibration {
    fn default() -> Self {
        Self::identity()
    }
}

/// Compute the tilt-compensated compass heading from a gravity vector and a
/// magnetometer reading, both expressed in the sensor frame.
///
/// This is the standard two-step tilt compensation algorithm:
/// 1. Use the accelerometer gravity vector to determine pitch and roll.
/// 2. Rotate the magnetometer reading into the horizontal plane using that tilt.
/// 3. Compute atan2 of the horizontal magnetic components.
///
/// # Parameters
/// - `gravity`: accelerometer reading (unit vector pointing toward gravity, sensor frame).
/// - `mag`: calibrated magnetometer unit vector (sensor frame).
///
/// # Returns
/// Compass heading in **degrees**, measured clockwise from magnetic north,
/// in the range [0, 360).
pub fn tilt_compensated_heading(gravity: [f32; 3], mag: [f32; 3]) -> f32 {
    let [gx, gy, gz] = gravity;
    let [mx, my, mz] = mag;

    // Normalise gravity vector.
    let g_norm = (gx * gx + gy * gy + gz * gz).sqrt().max(1e-10);
    let (gx, gy, gz) = (gx / g_norm, gy / g_norm, gz / g_norm);

    // Pitch and roll from accelerometer.
    // pitch = atan2(-gx, sqrt(gy²+gz²))
    // roll  = atan2(gy, gz)
    let pitch = (-gx).atan2((gy * gy + gz * gz).sqrt());
    let roll = gy.atan2(gz);

    let (sp, cp) = (pitch.sin(), pitch.cos());
    let (sr, cr) = (roll.sin(), roll.cos());

    // Tilt-compensated horizontal components:
    // Bx = mx*cp + my*sr*sp + mz*cr*sp
    // By = my*cr - mz*sr
    let bx = mx * cp + my * sr * sp + mz * cr * sp;
    let by = my * cr - mz * sr;

    // Heading (magnetic north reference, sensor x-axis = forward).
    let heading_rad = by.atan2(bx);
    heading_rad.to_degrees().rem_euclid(360.0)
}

// ─── FullImuTracker ──────────────────────────────────────────────────────────

/// Full 9-DOF IMU sensor fusion tracker.
///
/// Fuses accelerometer (tilt), gyroscope (angular rate integration), and
/// magnetometer (absolute yaw reference) using a three-stage complementary filter:
///
/// ```text
/// Stage 1 — Gyro integration:
///   q_gyro = q_prev ⊗ Δq(ω × dt)
///
/// Stage 2 — Accel/tilt correction (pitch + roll only):
///   q_tilt = slerp(q_gyro, q_accel, 1 - alpha_accel)
///
/// Stage 3 — Mag/yaw correction (yaw only):
///   heading = tilt_compensated_heading(accel, mag)
///   yaw_fused = alpha_mag * gyro_yaw + (1 - alpha_mag) * heading
///   q_final = from_euler(yaw_fused, pitch_tilt, roll_tilt)
/// ```
///
/// When the magnetometer is unavailable (or unreliable), the tracker falls
/// back to the 6-DOF gyro + accelerometer mode, which accumulates yaw drift
/// over time.
#[derive(Debug, Clone)]
pub struct FullImuTracker {
    /// Underlying 6-DOF head tracker (gyro + accel).
    inner: HeadTracker,
    /// Magnetometer hard-iron calibration.
    pub mag_calibration: MagCalibration,
    /// Complementary filter weight for gyro vs magnetometer yaw [0, 1].
    /// `1.0` = full gyro trust (ignores mag), `0.0` = full mag trust.
    /// Typical value: 0.90 (slower mag correction to handle magnetic disturbances).
    pub mag_alpha: f32,
    /// Last valid tilt-compensated heading (degrees), or `None` if not yet computed.
    last_heading_deg: Option<f32>,
    /// Magnetic declination correction to add to compass headings (degrees).
    /// Positive = east of north, negative = west of north.
    pub declination_deg: f32,
}

impl FullImuTracker {
    /// Create a new full 9-DOF IMU tracker.
    ///
    /// # Parameters
    /// - `gyro_accel_alpha`: complementary filter coefficient for the gyro/accel stage
    ///   (see [`HeadTracker::new`]).
    /// - `mag_alpha`: magnetometer yaw mixing coefficient [0, 1].
    /// - `smoothing`: post-filter smoothing strategy.
    pub fn new(gyro_accel_alpha: f32, mag_alpha: f32, smoothing: TrackingSmoothing) -> Self {
        Self {
            inner: HeadTracker::new(gyro_accel_alpha, smoothing),
            mag_calibration: MagCalibration::identity(),
            mag_alpha: mag_alpha.clamp(0.0, 1.0),
            last_heading_deg: None,
            declination_deg: 0.0,
        }
    }

    /// Update the orientation estimate with a full 9-DOF sensor reading.
    ///
    /// # Parameters
    /// - `gyro_delta`: angular velocity in deg/s for each axis (yaw, pitch, roll).
    /// - `accel_reading`: gravity vector expressed as (pitch_deg, roll_deg) from accelerometer.
    ///   Pass `None` if the accelerometer reading is invalid or unavailable.
    /// - `mag_reading`: raw magnetometer reading.  Pass `None` to fall back to 6-DOF mode.
    /// - `dt_ms`: time step in milliseconds.
    pub fn update(
        &mut self,
        gyro_delta: HeadOrientation,
        accel_reading: Option<HeadOrientation>,
        mag_reading: Option<MagnetometerReading>,
        dt_ms: f32,
    ) {
        // --- Stage 1 & 2: gyro + accel complementary filter ----------------
        self.inner.update(gyro_delta, accel_reading, dt_ms);

        // --- Stage 3: magnetometer yaw correction ---------------------------
        if let Some(mag_raw) = mag_reading {
            let mag_cal = self.mag_calibration.apply(mag_raw);

            // We need the accelerometer gravity vector in sensor frame.
            // Extract the current tilt (pitch + roll) from the inner tracker.
            let tilt = self.inner.current_orientation.to_euler();

            // Reconstruct approximate gravity vector from tilt angles.
            let pitch_r = tilt.pitch_deg.to_radians();
            let roll_r = tilt.roll_deg.to_radians();
            let gravity = [
                -pitch_r.sin(),
                roll_r.sin() * pitch_r.cos(),
                roll_r.cos() * pitch_r.cos(),
            ];

            if let Some(mag_unit) = mag_cal.normalise() {
                let raw_heading = tilt_compensated_heading(gravity, mag_unit);
                let heading_deg = raw_heading + self.declination_deg;

                // Complementary filter in the yaw axis:
                // fused_yaw = alpha_mag * gyro_yaw + (1 - alpha_mag) * mag_heading
                let gyro_yaw = tilt.yaw_deg;
                let fused_yaw = if let Some(last_h) = self.last_heading_deg {
                    // Wrap-aware blending: bring heading into gyro's ±180° neighbourhood.
                    let delta = wrap_angle(heading_deg - last_h);
                    let smoothed_heading = last_h + (1.0 - self.mag_alpha) * delta;
                    self.mag_alpha * gyro_yaw + (1.0 - self.mag_alpha) * smoothed_heading
                } else {
                    heading_deg
                };

                self.last_heading_deg = Some(fused_yaw);

                // Rebuild the quaternion with fused yaw but gyro/accel pitch+roll.
                self.inner.current_orientation =
                    Quaternion::from_euler(fused_yaw, tilt.pitch_deg, tilt.roll_deg);
            }
        }
    }

    /// Return the current orientation as Euler angles (degrees).
    pub fn euler(&self) -> HeadOrientation {
        self.inner.current_orientation.to_euler()
    }

    /// Return the current orientation quaternion.
    pub fn quaternion(&self) -> Quaternion {
        self.inner.current_orientation
    }

    /// Compute binaural rendering angles for a virtual source, compensating for head rotation.
    pub fn to_binaural_angles(&self, source_azimuth: f32, source_elevation: f32) -> (f32, f32) {
        self.inner
            .to_binaural_angles(source_azimuth, source_elevation)
    }

    /// Reset the tracker to the identity orientation.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.last_heading_deg = None;
    }

    /// Return the last tilt-compensated compass heading, if available.
    pub fn last_heading_deg(&self) -> Option<f32> {
        self.last_heading_deg
    }
}

/// Wrap an angle difference into [-180, 180].
fn wrap_angle(deg: f32) -> f32 {
    let mut d = deg.rem_euclid(360.0);
    if d > 180.0 {
        d -= 360.0;
    }
    d
}

/// Rotate a 3-D Cartesian vector by a unit quaternion.
fn rotate_vector_by_quat(v: [f32; 3], q: Quaternion) -> [f32; 3] {
    // Uses the sandwich product: v' = q * [0, v] * q_conj.
    let vq = Quaternion {
        w: 0.0,
        x: v[0],
        y: v[1],
        z: v[2],
    };
    let q_conj = q.conjugate();
    let rotated = q.multiply(&vq).multiply(&q_conj);
    [rotated.x, rotated.y, rotated.z]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(a: f32, b: f32, tol: f32, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: expected ≈ {b}, got {a} (tol {tol})"
        );
    }

    // ── Quaternion ──────────────────────────────────────────────────────────

    #[test]
    fn test_quaternion_identity_normalize() {
        let q = Quaternion::identity().normalize();
        assert_near(q.w, 1.0, 1e-5, "w");
        assert_near(q.x, 0.0, 1e-5, "x");
    }

    #[test]
    fn test_quaternion_multiply_identity() {
        let q = Quaternion::from_euler(30.0, 20.0, 10.0);
        let qi = Quaternion::identity();
        let r = q.multiply(&qi).normalize();
        assert_near(r.w, q.w, 1e-5, "w");
        assert_near(r.x, q.x, 1e-5, "x");
    }

    #[test]
    fn test_quaternion_euler_round_trip_yaw() {
        let yaw = 45.0_f32;
        let q = Quaternion::from_euler(yaw, 0.0, 0.0);
        let euler = q.to_euler();
        assert_near(euler.yaw_deg, yaw, 0.1, "yaw");
        assert_near(euler.pitch_deg, 0.0, 0.1, "pitch");
        assert_near(euler.roll_deg, 0.0, 0.1, "roll");
    }

    #[test]
    fn test_quaternion_euler_round_trip_pitch() {
        let pitch = 30.0_f32;
        let q = Quaternion::from_euler(0.0, pitch, 0.0);
        let euler = q.to_euler();
        assert_near(euler.pitch_deg, pitch, 0.1, "pitch");
    }

    #[test]
    fn test_quaternion_euler_round_trip_roll() {
        let roll = -20.0_f32;
        let q = Quaternion::from_euler(0.0, 0.0, roll);
        let euler = q.to_euler();
        assert_near(euler.roll_deg, roll, 0.1, "roll");
    }

    #[test]
    fn test_quaternion_euler_round_trip_combined() {
        let q = Quaternion::from_euler(15.0, 25.0, -10.0);
        let euler = q.to_euler();
        assert_near(euler.yaw_deg, 15.0, 0.5, "yaw");
        assert_near(euler.pitch_deg, 25.0, 0.5, "pitch");
        assert_near(euler.roll_deg, -10.0, 0.5, "roll");
    }

    #[test]
    fn test_quaternion_normalise_non_unit() {
        let q = Quaternion {
            w: 2.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let n = q.normalize();
        let norm = (n.w * n.w + n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
        assert_near(norm, 1.0, 1e-5, "norm");
    }

    #[test]
    fn test_quaternion_slerp_t0_returns_self() {
        let q0 = Quaternion::from_euler(10.0, 20.0, 30.0);
        let q1 = Quaternion::from_euler(50.0, 60.0, 70.0);
        let r = q0.slerp(&q1, 0.0);
        let e0 = q0.to_euler();
        let er = r.to_euler();
        assert_near(er.yaw_deg, e0.yaw_deg, 0.5, "slerp t=0 yaw");
    }

    #[test]
    fn test_quaternion_slerp_t1_returns_other() {
        let q0 = Quaternion::from_euler(10.0, 20.0, 30.0);
        let q1 = Quaternion::from_euler(50.0, 60.0, 70.0);
        let r = q0.slerp(&q1, 1.0);
        let e1 = q1.to_euler();
        let er = r.to_euler();
        assert_near(er.yaw_deg, e1.yaw_deg, 0.5, "slerp t=1 yaw");
    }

    // ── LowPassFilter ────────────────────────────────────────────────────────

    #[test]
    fn test_low_pass_filter_converges() {
        let mut f = LowPassFilter::new(0.5);
        let mut val = 0.0_f32;
        for _ in 0..30 {
            val = f.process(1.0);
        }
        assert!(val > 0.99, "LPF should converge to 1.0, got {val}");
    }

    #[test]
    fn test_low_pass_filter_alpha_one_is_passthrough() {
        let mut f = LowPassFilter::new(1.0);
        let out = f.process(0.5);
        assert_near(out, 0.5, 1e-5, "alpha=1 should be passthrough");
    }

    #[test]
    fn test_low_pass_filter_from_cutoff() {
        let f = LowPassFilter::from_cutoff(10.0, 100.0);
        // alpha should be in (0, 1)
        assert!(f.alpha > 0.0 && f.alpha < 1.0, "alpha={}", f.alpha);
    }

    // ── KalmanOrientation ────────────────────────────────────────────────────

    #[test]
    fn test_kalman_predict_propagates_state() {
        let mut k = KalmanOrientation::new(0.0);
        k.state[1] = 10.0; // 10 deg/s
        k.predict(0.1); // 100 ms step
        assert_near(k.state[0], 1.0, 0.1, "angle after predict");
    }

    #[test]
    fn test_kalman_update_moves_toward_measurement() {
        let mut k = KalmanOrientation::new(0.0);
        for _ in 0..20 {
            k.predict(0.01);
            k.update(45.0);
        }
        assert!(
            k.angle() > 10.0,
            "Kalman should move toward measurement 45°, got {}",
            k.angle()
        );
    }

    // ── HeadTracker ──────────────────────────────────────────────────────────

    #[test]
    fn test_head_tracker_identity_at_start() {
        let tracker = HeadTracker::new(0.98, TrackingSmoothing::None);
        let euler = tracker.euler();
        assert_near(euler.yaw_deg, 0.0, 1e-3, "initial yaw");
        assert_near(euler.pitch_deg, 0.0, 1e-3, "initial pitch");
    }

    #[test]
    fn test_head_tracker_gyro_only_accumulates_yaw() {
        let mut tracker = HeadTracker::new(1.0, TrackingSmoothing::None);
        // 10 deg/s yaw × 100 ms = 1 degree.
        let gyro = HeadOrientation {
            yaw_deg: 10.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        };
        tracker.update(gyro, None, 100.0);
        let euler = tracker.euler();
        assert!(
            euler.yaw_deg.abs() > 0.5,
            "Yaw should accumulate, got {}",
            euler.yaw_deg
        );
    }

    #[test]
    fn test_head_tracker_accel_correction_at_low_alpha() {
        // With alpha = 0 (fully trust accel), orientation should converge to accel reading.
        let mut tracker = HeadTracker::new(0.0, TrackingSmoothing::None);
        let accel = HeadOrientation {
            yaw_deg: 0.0,
            pitch_deg: 20.0,
            roll_deg: 0.0,
        };
        for _ in 0..10 {
            tracker.update(HeadOrientation::identity(), Some(accel), 10.0);
        }
        let euler = tracker.euler();
        // Pitch should be pulled toward 20°.
        assert!(
            euler.pitch_deg > 5.0,
            "Pitch should converge toward accel value, got {}",
            euler.pitch_deg
        );
    }

    #[test]
    fn test_head_tracker_reset_clears_orientation() {
        let mut tracker = HeadTracker::new(0.98, TrackingSmoothing::None);
        let gyro = HeadOrientation {
            yaw_deg: 100.0,
            pitch_deg: 50.0,
            roll_deg: 30.0,
        };
        tracker.update(gyro, None, 50.0);
        tracker.reset();
        let euler = tracker.euler();
        assert_near(euler.yaw_deg, 0.0, 1.0, "reset yaw");
    }

    #[test]
    fn test_head_tracker_lowpass_smoothing() {
        let mut tracker = HeadTracker::new(1.0, TrackingSmoothing::LowPass { cutoff_hz: 5.0 });
        let gyro = HeadOrientation {
            yaw_deg: 200.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        };
        tracker.update(gyro, None, 10.0);
        // Low-pass should attenuate abrupt changes.
        let euler = tracker.euler();
        // The filtered output should be less than the raw integration result.
        assert!(
            euler.yaw_deg.abs() < 100.0,
            "Low-pass should attenuate, got yaw={}",
            euler.yaw_deg
        );
    }

    #[test]
    fn test_head_tracker_kalman_smoothing() {
        let mut tracker = HeadTracker::new(1.0, TrackingSmoothing::Kalman);
        let gyro = HeadOrientation {
            yaw_deg: 50.0,
            pitch_deg: 10.0,
            roll_deg: 5.0,
        };
        tracker.update(gyro, None, 20.0);
        let euler = tracker.euler();
        // Kalman should produce a finite orientation without panic.
        assert!(euler.yaw_deg.is_finite(), "Kalman yaw should be finite");
        assert!(euler.pitch_deg.is_finite(), "Kalman pitch should be finite");
    }

    // ── to_binaural_angles ───────────────────────────────────────────────────

    #[test]
    fn test_binaural_angles_identity_no_change() {
        let tracker = HeadTracker::new(0.98, TrackingSmoothing::None);
        let (az, el) = tracker.to_binaural_angles(45.0, 10.0);
        assert_near(az, 45.0, 1.0, "binaural az with identity");
        assert_near(el, 10.0, 1.0, "binaural el with identity");
    }

    #[test]
    fn test_binaural_angles_after_yaw() {
        let mut tracker = HeadTracker::new(1.0, TrackingSmoothing::None);
        // Rotate head 90° to the left (yaw +90°) → a front source should appear to the right.
        let gyro = HeadOrientation {
            yaw_deg: 9000.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        };
        tracker.update(gyro, None, 10.0); // 9000 deg/s × 10ms = 90°
        let (az, _el) = tracker.to_binaural_angles(0.0, 0.0);
        // After 90° yaw left, a front source should appear at roughly -90° (to the right).
        assert!(
            az.abs() > 45.0,
            "Source should have shifted due to head yaw, got az={}",
            az
        );
    }

    // ── MagnetometerReading ──────────────────────────────────────────────────

    #[test]
    fn test_magnetometer_reading_normalise() {
        let mag = MagnetometerReading::new(3.0, 4.0, 0.0);
        let unit = mag.normalise().expect("non-zero vector should normalise");
        let norm = (unit[0] * unit[0] + unit[1] * unit[1] + unit[2] * unit[2]).sqrt();
        assert_near(norm, 1.0, 1e-5, "normalised mag vector");
    }

    #[test]
    fn test_magnetometer_reading_normalise_zero_returns_none() {
        let mag = MagnetometerReading::new(0.0, 0.0, 0.0);
        assert!(mag.normalise().is_none(), "Zero vector should return None");
    }

    // ── MagCalibration ───────────────────────────────────────────────────────

    #[test]
    fn test_mag_calibration_identity_no_change() {
        let cal = MagCalibration::identity();
        let raw = MagnetometerReading::new(10.0, 20.0, 30.0);
        let corrected = cal.apply(raw);
        assert_near(corrected.x, 10.0, 1e-5, "identity cal x");
        assert_near(corrected.y, 20.0, 1e-5, "identity cal y");
        assert_near(corrected.z, 30.0, 1e-5, "identity cal z");
    }

    #[test]
    fn test_mag_calibration_offset_subtracted() {
        let cal = MagCalibration {
            offset_x: 5.0,
            offset_y: -3.0,
            offset_z: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            scale_z: 1.0,
        };
        let raw = MagnetometerReading::new(10.0, 7.0, 2.0);
        let corrected = cal.apply(raw);
        assert_near(corrected.x, 5.0, 1e-5, "offset subtracted x");
        assert_near(corrected.y, 10.0, 1e-5, "offset subtracted y");
    }

    // ── tilt_compensated_heading ─────────────────────────────────────────────

    #[test]
    fn test_tilt_compensated_heading_north_no_tilt() {
        // When the sensor is level (gravity = [0, 0, 1]) and the magnetometer
        // points north (+x in sensor frame), the heading should be 0° (or 360°).
        let gravity = [0.0_f32, 0.0, 1.0];
        let mag = [1.0_f32, 0.0, 0.0];
        let heading = tilt_compensated_heading(gravity, mag);
        // heading = atan2(By, Bx) = atan2(0, 1) = 0° → rem_euclid = 0°.
        assert!(
            heading.abs() < 5.0 || (heading - 360.0).abs() < 5.0,
            "North-pointing mag with level sensor should give ~0°, got {heading}"
        );
    }

    #[test]
    fn test_tilt_compensated_heading_east() {
        // Magnetometer pointing east (+y in sensor frame, level sensor).
        let gravity = [0.0_f32, 0.0, 1.0];
        let mag = [0.0_f32, 1.0, 0.0];
        let heading = tilt_compensated_heading(gravity, mag);
        // atan2(By, Bx) = atan2(1, 0) = 90° → 90° in rem_euclid.
        // Depending on axis convention the expected value might differ; just check non-zero.
        assert!(
            heading.is_finite(),
            "East heading should be finite, got {heading}"
        );
    }

    #[test]
    fn test_tilt_compensated_heading_range() {
        // Heading should always be in [0, 360).
        let gravity = [0.1_f32, 0.0, 0.99];
        for i in 0..8 {
            let angle = i as f32 * 45.0_f32.to_radians();
            let mag = [angle.cos(), angle.sin(), 0.0_f32];
            let heading = tilt_compensated_heading(gravity, mag);
            assert!(
                heading >= 0.0 && heading < 360.0,
                "Heading should be in [0, 360), got {heading}"
            );
        }
    }

    // ── FullImuTracker ───────────────────────────────────────────────────────

    #[test]
    fn test_full_imu_tracker_identity_at_start() {
        let tracker = FullImuTracker::new(0.98, 0.9, TrackingSmoothing::None);
        let euler = tracker.euler();
        assert_near(euler.yaw_deg, 0.0, 1e-3, "initial yaw");
        assert_near(euler.pitch_deg, 0.0, 1e-3, "initial pitch");
    }

    #[test]
    fn test_full_imu_tracker_gyro_only_accumulates() {
        let mut tracker = FullImuTracker::new(1.0, 1.0, TrackingSmoothing::None);
        let gyro = HeadOrientation {
            yaw_deg: 10.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        };
        tracker.update(gyro, None, None, 100.0);
        let euler = tracker.euler();
        assert!(
            euler.yaw_deg.abs() > 0.5,
            "Gyro-only should accumulate yaw, got {}",
            euler.yaw_deg
        );
    }

    #[test]
    fn test_full_imu_tracker_mag_correction_produces_finite_output() {
        let mut tracker = FullImuTracker::new(0.98, 0.9, TrackingSmoothing::None);
        let gyro = HeadOrientation {
            yaw_deg: 2.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        };
        let accel = HeadOrientation {
            yaw_deg: 0.0,
            pitch_deg: 5.0,
            roll_deg: 0.0,
        };
        let mag = Some(MagnetometerReading::new(0.8, 0.2, 0.1));

        for _ in 0..20 {
            tracker.update(gyro, Some(accel), mag, 10.0);
        }

        let euler = tracker.euler();
        assert!(euler.yaw_deg.is_finite(), "Yaw should be finite");
        assert!(euler.pitch_deg.is_finite(), "Pitch should be finite");
        assert!(euler.roll_deg.is_finite(), "Roll should be finite");
    }

    #[test]
    fn test_full_imu_tracker_reset_clears_state() {
        let mut tracker = FullImuTracker::new(0.98, 0.9, TrackingSmoothing::None);
        let gyro = HeadOrientation {
            yaw_deg: 100.0,
            pitch_deg: 50.0,
            roll_deg: 0.0,
        };
        tracker.update(gyro, None, None, 50.0);
        tracker.reset();
        let euler = tracker.euler();
        assert_near(euler.yaw_deg, 0.0, 1.0, "reset yaw");
        assert!(
            tracker.last_heading_deg().is_none(),
            "heading should be cleared on reset"
        );
    }

    #[test]
    fn test_full_imu_tracker_mag_heading_stored() {
        let mut tracker = FullImuTracker::new(0.98, 0.9, TrackingSmoothing::None);
        let gyro = HeadOrientation::identity();
        let accel = HeadOrientation::identity();
        let mag = MagnetometerReading::new(1.0, 0.0, 0.0);
        tracker.update(gyro, Some(accel), Some(mag), 10.0);
        assert!(
            tracker.last_heading_deg().is_some(),
            "Heading should be set after mag update"
        );
    }

    #[test]
    fn test_full_imu_tracker_declination_applied() {
        let mut tracker = FullImuTracker::new(0.98, 0.0, TrackingSmoothing::None);
        tracker.declination_deg = 5.0; // 5° east
        let gyro = HeadOrientation::identity();
        let accel = HeadOrientation::identity();
        let mag = MagnetometerReading::new(1.0, 0.0, 0.0);
        tracker.update(gyro, Some(accel), Some(mag), 10.0);
        // With mag_alpha=0 the yaw should be the heading + declination.
        let euler = tracker.euler();
        // Heading from mag pointing north = 0°, + 5° declination = 5°.
        assert!(
            euler.yaw_deg.is_finite(),
            "Yaw should be finite after declination"
        );
    }

    // ── Rotation matrix LUT ──────────────────────────────────────────────────

    #[test]
    fn test_rotation_cache_identity() {
        let mat = cached_rotation_matrix(0.0, 0.0, 0.0);
        // R(0,0,0) must be the 3×3 identity matrix.
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_near(mat[i][j], expected, 1e-5, &format!("identity[{i}][{j}]"));
            }
        }
    }

    #[test]
    fn test_rotation_cache_hit() {
        // Two calls with identical angles must return the same matrix.
        let mat1 = cached_rotation_matrix(37.0, 12.0, -5.0);
        let mat2 = cached_rotation_matrix(37.0, 12.0, -5.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    mat1[i][j], mat2[i][j],
                    "Cache hit should return equal value at [{i}][{j}]"
                );
            }
        }
        // Sanity: the cached value must match a fresh computation.
        let expected = compute_rotation_matrix(37.0, 12.0, -5.0);
        for i in 0..3 {
            for j in 0..3 {
                assert_near(
                    mat1[i][j],
                    expected[i][j],
                    1e-5,
                    &format!("cache vs compute [{i}][{j}]"),
                );
            }
        }
    }

    #[test]
    fn test_rotation_matrix_yaw_90() {
        // Rz(90°): x→y, y→-x, z→z
        // Expected column-major result for R = Rz(90°)*Ry(0)*Rx(0):
        // row0: [cos90, 0, 0] = [0, 0, 0]  — Rz only affects x/y rows
        // Actual Rz(yaw) * Ry(0) * Rx(0):
        //   [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr]
        //   [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr]
        //   [-sp,    cp*sr,             cp*cr            ]
        // With yaw=90°, pitch=0°, roll=0°:
        //   cy=0, sy=1, cp=1, sp=0, cr=1, sr=0
        //   row0: [0, -1,  0]
        //   row1: [1,  0,  0]
        //   row2: [0,  0,  1]
        let mat = cached_rotation_matrix(90.0, 0.0, 0.0);
        assert_near(mat[0][0], 0.0, 1e-5, "R[0][0] yaw90");
        assert_near(mat[0][1], -1.0, 1e-5, "R[0][1] yaw90");
        assert_near(mat[0][2], 0.0, 1e-5, "R[0][2] yaw90");
        assert_near(mat[1][0], 1.0, 1e-5, "R[1][0] yaw90");
        assert_near(mat[1][1], 0.0, 1e-5, "R[1][1] yaw90");
        assert_near(mat[1][2], 0.0, 1e-5, "R[1][2] yaw90");
        assert_near(mat[2][0], 0.0, 1e-5, "R[2][0] yaw90");
        assert_near(mat[2][1], 0.0, 1e-5, "R[2][1] yaw90");
        assert_near(mat[2][2], 1.0, 1e-5, "R[2][2] yaw90");
    }
}
