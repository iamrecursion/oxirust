//! 360° video stabilization using gyroscope / IMU metadata.
//!
//! Camera shake and unwanted rotation in 360° video can be corrected by
//! integrating the IMU (Inertial Measurement Unit) gyroscope data to obtain a
//! per-frame orientation, computing the deviation from a desired smooth
//! trajectory, and applying the inverse rotation to each frame.
//!
//! ## Pipeline
//!
//! 1. **IMU integration** — `ImuSample` readings (angular velocity in rad/s)
//!    are integrated over time to produce per-frame absolute orientations
//!    (`Orientation` quaternion/matrix).
//! 2. **Trajectory smoothing** — The raw orientation sequence is smoothed with
//!    a configurable window (Gaussian-weighted average of quaternions).
//! 3. **Correction computation** — Each frame's correction rotation is the
//!    difference between the raw and the smoothed orientation.
//! 4. **Frame warping** — [`stabilize_frame`] applies the correction rotation
//!    to an equirectangular frame using bilinear resampling.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::stabilization::{ImuSample, ImuIntegrator, StabilizationConfig};
//!
//! // Build a sequence of IMU samples
//! let samples = vec![
//!     ImuSample { timestamp_s: 0.0, gyro_x: 0.01, gyro_y: 0.0, gyro_z: 0.0 },
//!     ImuSample { timestamp_s: 0.033, gyro_x: 0.0, gyro_y: 0.0, gyro_z: 0.0 },
//! ];
//!
//! let integrator = ImuIntegrator::new(samples);
//! let orientations = integrator.integrate();
//! ```

use crate::{
    orientation::RotMat3,
    projection::{bilinear_sample_u8, equirect_to_sphere, sphere_to_equirect, UvCoord},
    VrError,
};

// ─── IMU sample ───────────────────────────────────────────────────────────────

/// A single IMU gyroscope reading.
///
/// The angular velocity components are given in the camera's local coordinate
/// frame (right-hand rule):
/// * `gyro_x` — rotation around the X axis (pitch rate), rad/s
/// * `gyro_y` — rotation around the Y axis (yaw rate), rad/s
/// * `gyro_z` — rotation around the Z axis (roll rate), rad/s
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSample {
    /// Timestamp in seconds from the start of the clip.
    pub timestamp_s: f64,
    /// Pitch angular velocity in rad/s.
    pub gyro_x: f32,
    /// Yaw angular velocity in rad/s.
    pub gyro_y: f32,
    /// Roll angular velocity in rad/s.
    pub gyro_z: f32,
}

// ─── Integrated orientation frame ────────────────────────────────────────────

/// Absolute orientation at a specific video frame timestamp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameOrientation {
    /// Frame timestamp in seconds.
    pub timestamp_s: f64,
    /// Absolute orientation represented as a unit quaternion `(w, x, y, z)`.
    pub quaternion: [f32; 4],
}

impl FrameOrientation {
    /// Identity orientation (no rotation).
    pub fn identity(timestamp_s: f64) -> Self {
        Self {
            timestamp_s,
            quaternion: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Convert to a 3×3 rotation matrix.
    pub fn to_rot_mat(&self) -> RotMat3 {
        let [w, x, y, z] = self.quaternion;
        RotMat3([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ])
    }
}

// ─── IMU integrator ───────────────────────────────────────────────────────────

/// Integrates gyroscope samples into absolute orientation quaternions.
///
/// Uses simple first-order integration (Euler integration) which is accurate
/// enough for typical video frame rates (≥ 30 fps) with well-sampled IMU data
/// (≥ 200 Hz).
pub struct ImuIntegrator {
    samples: Vec<ImuSample>,
}

impl ImuIntegrator {
    /// Create a new integrator from a sequence of IMU samples, sorted by
    /// timestamp.
    pub fn new(mut samples: Vec<ImuSample>) -> Self {
        samples.sort_by(|a, b| {
            a.timestamp_s
                .partial_cmp(&b.timestamp_s)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { samples }
    }

    /// Integrate all samples and return one [`FrameOrientation`] per IMU sample.
    ///
    /// The first sample always has the identity orientation.  Each subsequent
    /// sample updates the quaternion by applying the small-angle rotation
    /// implied by the angular velocity and the time delta.
    pub fn integrate(&self) -> Vec<FrameOrientation> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let mut orientations = Vec::with_capacity(self.samples.len());
        let mut q = [1.0_f32, 0.0, 0.0, 0.0]; // identity quaternion [w,x,y,z]

        orientations.push(FrameOrientation {
            timestamp_s: self.samples[0].timestamp_s,
            quaternion: q,
        });

        for window in self.samples.windows(2) {
            let dt = (window[1].timestamp_s - window[0].timestamp_s) as f32;
            if dt <= 0.0 {
                orientations.push(FrameOrientation {
                    timestamp_s: window[1].timestamp_s,
                    quaternion: q,
                });
                continue;
            }

            // Angular velocity from the more recent sample
            let gx = window[1].gyro_x;
            let gy = window[1].gyro_y;
            let gz = window[1].gyro_z;

            // Small rotation quaternion: dq = [1, gx*dt/2, gy*dt/2, gz*dt/2] (normalised)
            let half_dt = dt * 0.5;
            let dq = [1.0_f32, gx * half_dt, gy * half_dt, gz * half_dt];

            // Compose: q = q ⊗ dq  (Hamilton product)
            q = quat_mul(q, dq);
            q = quat_normalise(q);

            orientations.push(FrameOrientation {
                timestamp_s: window[1].timestamp_s,
                quaternion: q,
            });
        }

        orientations
    }

    /// Integrate and return orientations at specific frame timestamps.
    ///
    /// Linearly interpolates (SLERP) between the nearest IMU samples.
    /// Timestamps outside the IMU range use the boundary orientation.
    pub fn orientations_at_timestamps(&self, timestamps: &[f64]) -> Vec<FrameOrientation> {
        let integrated = self.integrate();
        if integrated.is_empty() {
            return timestamps
                .iter()
                .map(|&t| FrameOrientation::identity(t))
                .collect();
        }

        timestamps
            .iter()
            .map(|&t| {
                // Find bracketing samples
                let pos = integrated.partition_point(|o| o.timestamp_s <= t);

                let result_q = if pos == 0 {
                    integrated[0].quaternion
                } else if pos >= integrated.len() {
                    integrated[integrated.len() - 1].quaternion
                } else {
                    let a = &integrated[pos - 1];
                    let b = &integrated[pos];
                    let span = (b.timestamp_s - a.timestamp_s) as f32;
                    let alpha = if span > 1e-9 {
                        ((t - a.timestamp_s) as f32) / span
                    } else {
                        0.0
                    };
                    quat_slerp(a.quaternion, b.quaternion, alpha)
                };

                FrameOrientation {
                    timestamp_s: t,
                    quaternion: result_q,
                }
            })
            .collect()
    }
}

// ─── Trajectory smoothing ─────────────────────────────────────────────────────

/// Configuration for the stabilization pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilizationConfig {
    /// Number of frames to include on each side of the smoothing window
    /// (total window = `2 × half_window + 1`).  Larger values produce
    /// smoother motion at the cost of more latency / look-ahead.
    pub half_window: usize,
    /// Standard deviation for the Gaussian weighting of the smoothing window,
    /// expressed in frames.  Typical range: 3–15.
    pub gaussian_sigma: f32,
}

impl Default for StabilizationConfig {
    fn default() -> Self {
        Self {
            half_window: 15,
            gaussian_sigma: 7.0,
        }
    }
}

/// Smooth a sequence of frame orientations using a Gaussian-weighted window.
///
/// Returns a vector of smoothed orientations with the same length as `raw`.
pub fn smooth_orientations(
    raw: &[FrameOrientation],
    config: &StabilizationConfig,
) -> Vec<FrameOrientation> {
    if raw.is_empty() {
        return Vec::new();
    }

    let n = raw.len();
    let hw = config.half_window;
    let sigma = config.gaussian_sigma;

    // Pre-compute Gaussian weights for offsets -hw..=hw
    let weights: Vec<f32> = (-(hw as i64)..=(hw as i64))
        .map(|k| {
            let kf = k as f32;
            (-(kf * kf) / (2.0 * sigma * sigma)).exp()
        })
        .collect();

    raw.iter()
        .enumerate()
        .map(|(i, fo)| {
            let mut acc_q = [0.0_f32; 4];
            let mut weight_sum = 0.0_f32;

            // Reference quaternion for consistent hemisphere
            let ref_q = fo.quaternion;

            for (ki, &w) in weights.iter().enumerate() {
                let offset = ki as i64 - hw as i64;
                let j = i as i64 + offset;
                if j < 0 || j >= n as i64 {
                    continue;
                }
                let jq = raw[j as usize].quaternion;
                // Flip quaternion to same hemisphere as reference
                let jq_adj = if quat_dot(ref_q, jq) < 0.0 {
                    [-jq[0], -jq[1], -jq[2], -jq[3]]
                } else {
                    jq
                };
                for c in 0..4 {
                    acc_q[c] += w * jq_adj[c];
                }
                weight_sum += w;
            }

            let smoothed_q = if weight_sum > 1e-9 {
                quat_normalise([
                    acc_q[0] / weight_sum,
                    acc_q[1] / weight_sum,
                    acc_q[2] / weight_sum,
                    acc_q[3] / weight_sum,
                ])
            } else {
                fo.quaternion
            };

            FrameOrientation {
                timestamp_s: fo.timestamp_s,
                quaternion: smoothed_q,
            }
        })
        .collect()
}

// ─── Frame stabilization ──────────────────────────────────────────────────────

/// Stabilize a single equirectangular frame by applying a correction rotation.
///
/// The correction rotation is computed as `R_smooth · R_raw⁻¹`, which
/// transforms pixel directions from the raw (shaky) frame orientation to the
/// desired smooth orientation.
///
/// * `src`        — equirectangular pixel data (RGB, 3 bpp, row-major)
/// * `width`      — image width in pixels
/// * `height`     — image height in pixels
/// * `raw`        — frame's raw (integrated gyro) orientation
/// * `smooth`     — frame's desired smooth orientation
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero.
/// Returns [`VrError::BufferTooSmall`] if `src` is too small.
pub fn stabilize_frame(
    src: &[u8],
    width: u32,
    height: u32,
    raw: &FrameOrientation,
    smooth: &FrameOrientation,
) -> Result<Vec<u8>, VrError> {
    if width == 0 || height == 0 {
        return Err(VrError::InvalidDimensions(
            "width and height must be > 0".into(),
        ));
    }
    let expected = width as usize * height as usize * 3;
    if src.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: src.len(),
        });
    }

    // Correction: R_smooth · R_raw^{-1}
    // Applied to each output pixel's world direction to find source direction.
    let mat_smooth = smooth.to_rot_mat();
    let mat_raw = raw.to_rot_mat();

    // R_raw^{-1} = R_raw^T (rotation matrix inverse = transpose)
    let mat_raw_inv = RotMat3([
        [mat_raw.0[0][0], mat_raw.0[1][0], mat_raw.0[2][0]],
        [mat_raw.0[0][1], mat_raw.0[1][1], mat_raw.0[2][1]],
        [mat_raw.0[0][2], mat_raw.0[1][2], mat_raw.0[2][2]],
    ]);

    // Correction = R_smooth · R_raw^{-1}
    let mat_corr = mat_smooth.mul(&mat_raw_inv);

    const CH: u32 = 3;
    let mut out = vec![0u8; expected];

    for oy in 0..height {
        for ox in 0..width {
            let u = (ox as f32 + 0.5) / width as f32;
            let v = (oy as f32 + 0.5) / height as f32;

            let sphere_out = equirect_to_sphere(&UvCoord { u, v });

            // Convert to Cartesian
            let x = sphere_out.elevation_rad.cos() * sphere_out.azimuth_rad.sin();
            let y = sphere_out.elevation_rad.sin();
            let z = sphere_out.elevation_rad.cos() * sphere_out.azimuth_rad.cos();

            // Apply the inverse correction to find the source direction
            // Source direction = R_corr^{-1} · output direction = R_corr^T · output direction
            let sv = mat_corr.apply_inverse([x, y, z]);
            let el = sv[1].clamp(-1.0, 1.0).asin();
            let az = sv[0].atan2(sv[2]);

            let src_uv = sphere_to_equirect(&crate::projection::SphericalCoord {
                azimuth_rad: az,
                elevation_rad: el,
            });
            let sample = bilinear_sample_u8(src, width, height, src_uv.u, src_uv.v, CH);
            let dst = (oy * width + ox) as usize * CH as usize;
            out[dst..dst + CH as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

// ─── Quaternion helpers ───────────────────────────────────────────────────────

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

fn quat_normalise(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len < 1e-10 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

fn quat_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn quat_slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = quat_dot(a, b).clamp(-1.0, 1.0);
    let b_adj = if dot < 0.0 {
        dot = -dot;
        [-b[0], -b[1], -b[2], -b[3]]
    } else {
        b
    };

    if dot > 0.9995 {
        // Linear interpolation for very close quaternions
        return quat_normalise([
            a[0] + t * (b_adj[0] - a[0]),
            a[1] + t * (b_adj[1] - a[1]),
            a[2] + t * (b_adj[2] - a[2]),
            a[3] + t * (b_adj[3] - a[3]),
        ]);
    }

    let theta_0 = dot.acos();
    let sin_theta_0 = theta_0.sin();
    let s0 = ((1.0 - t) * theta_0).sin() / sin_theta_0;
    let s1 = (t * theta_0).sin() / sin_theta_0;

    [
        s0 * a[0] + s1 * b_adj[0],
        s0 * a[1] + s1 * b_adj[1],
        s0 * a[2] + s1 * b_adj[2],
        s0 * a[3] + s1 * b_adj[3],
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    // ── Quaternion helpers ───────────────────────────────────────────────────

    #[test]
    fn quat_identity_mul_is_identity() {
        let id = [1.0_f32, 0.0, 0.0, 0.0];
        let q = [0.7071_f32, 0.7071, 0.0, 0.0];
        let result = quat_mul(id, q);
        for (a, b) in result.iter().zip(q.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a} b={b}");
        }
    }

    #[test]
    fn quat_normalise_unit_stays_unit() {
        let q = quat_normalise([1.0, 0.0, 0.0, 0.0]);
        assert!((q[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn quat_slerp_t0_returns_a() {
        let a = [1.0_f32, 0.0, 0.0, 0.0];
        let b = [0.7071_f32, 0.7071, 0.0, 0.0];
        let r = quat_slerp(a, b, 0.0);
        for (x, y) in r.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-4);
        }
    }

    #[test]
    fn quat_slerp_t1_returns_b() {
        let a = [1.0_f32, 0.0, 0.0, 0.0];
        let b = quat_normalise([0.9659, 0.2588, 0.0, 0.0]); // 30° around X
        let r = quat_slerp(a, b, 1.0);
        for (x, y) in r.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-3, "x={x} y={y}");
        }
    }

    // ── ImuIntegrator ────────────────────────────────────────────────────────

    #[test]
    fn imu_integrator_empty_input_returns_empty() {
        let integrator = ImuIntegrator::new(vec![]);
        let result = integrator.integrate();
        assert!(result.is_empty());
    }

    #[test]
    fn imu_integrator_single_sample_is_identity() {
        let samples = vec![ImuSample {
            timestamp_s: 0.0,
            gyro_x: 1.0,
            gyro_y: 0.5,
            gyro_z: -0.3,
        }];
        let integrator = ImuIntegrator::new(samples);
        let orientations = integrator.integrate();
        assert_eq!(orientations.len(), 1);
        // First sample is always identity
        let q = orientations[0].quaternion;
        assert!((q[0] - 1.0).abs() < 1e-6, "w={}", q[0]);
        assert!(q[1].abs() < 1e-6);
        assert!(q[2].abs() < 1e-6);
        assert!(q[3].abs() < 1e-6);
    }

    #[test]
    fn imu_integrator_zero_rotation_stays_identity() {
        let samples: Vec<ImuSample> = (0..10)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.033,
                gyro_x: 0.0,
                gyro_y: 0.0,
                gyro_z: 0.0,
            })
            .collect();
        let integrator = ImuIntegrator::new(samples);
        let orientations = integrator.integrate();
        assert_eq!(orientations.len(), 10);
        for o in &orientations {
            assert!(
                (o.quaternion[0] - 1.0).abs() < 1e-5,
                "w={}",
                o.quaternion[0]
            );
        }
    }

    #[test]
    fn imu_integrator_nonzero_rotation_changes_quaternion() {
        // A constant yaw rotation should accumulate over time
        let samples: Vec<ImuSample> = (0..5)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.033,
                gyro_x: 0.0,
                gyro_y: 1.0, // 1 rad/s yaw
                gyro_z: 0.0,
            })
            .collect();
        let integrator = ImuIntegrator::new(samples);
        let orientations = integrator.integrate();
        assert_eq!(orientations.len(), 5);
        // Last orientation should differ from identity
        let last_q = orientations[4].quaternion;
        let is_identity = (last_q[0] - 1.0).abs() < 1e-3
            && last_q[1].abs() < 1e-3
            && last_q[2].abs() < 1e-3
            && last_q[3].abs() < 1e-3;
        assert!(!is_identity, "expected non-identity after integration");
    }

    #[test]
    fn imu_orientations_at_timestamps_returns_correct_count() {
        let samples: Vec<ImuSample> = (0..20)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.01,
                gyro_x: 0.0,
                gyro_y: 0.0,
                gyro_z: 0.0,
            })
            .collect();
        let integrator = ImuIntegrator::new(samples);
        let timestamps = [0.0, 0.05, 0.10, 0.15, 0.19];
        let result = integrator.orientations_at_timestamps(&timestamps);
        assert_eq!(result.len(), timestamps.len());
    }

    // ── smooth_orientations ──────────────────────────────────────────────────

    #[test]
    fn smooth_orientations_returns_same_length() {
        let raw: Vec<FrameOrientation> = (0..20)
            .map(|i| FrameOrientation::identity(i as f64 * 0.033))
            .collect();
        let config = StabilizationConfig {
            half_window: 5,
            gaussian_sigma: 3.0,
        };
        let smoothed = smooth_orientations(&raw, &config);
        assert_eq!(smoothed.len(), raw.len());
    }

    #[test]
    fn smooth_identity_sequence_stays_identity() {
        let raw: Vec<FrameOrientation> = (0..10)
            .map(|i| FrameOrientation::identity(i as f64 * 0.033))
            .collect();
        let config = StabilizationConfig::default();
        let smoothed = smooth_orientations(&raw, &config);
        for s in &smoothed {
            assert!(
                (s.quaternion[0] - 1.0).abs() < 1e-4,
                "w={}",
                s.quaternion[0]
            );
        }
    }

    // ── stabilize_frame ──────────────────────────────────────────────────────

    #[test]
    fn stabilize_frame_identity_is_noop() {
        let src = solid_rgb(64, 32, 128, 64, 32);
        let id = FrameOrientation::identity(0.0);
        let out = stabilize_frame(&src, 64, 32, &id, &id).expect("ok");
        assert_eq!(out.len(), src.len());
        // With identity correction, centre pixel should be close to original
        let base = (16 * 64 + 32) * 3;
        assert!((out[base] as i32 - 128).abs() <= 3);
    }

    #[test]
    fn stabilize_frame_zero_dimensions_error() {
        let src = solid_rgb(64, 32, 0, 0, 0);
        let id = FrameOrientation::identity(0.0);
        assert!(stabilize_frame(&src, 0, 32, &id, &id).is_err());
        assert!(stabilize_frame(&src, 64, 0, &id, &id).is_err());
    }

    #[test]
    fn stabilize_frame_buffer_too_small_error() {
        let id = FrameOrientation::identity(0.0);
        assert!(stabilize_frame(&[0u8; 10], 64, 32, &id, &id).is_err());
    }

    #[test]
    fn stabilize_frame_correct_output_size() {
        let src = solid_rgb(32, 16, 200, 100, 50);
        let id = FrameOrientation::identity(0.0);
        let out = stabilize_frame(&src, 32, 16, &id, &id).expect("ok");
        assert_eq!(out.len(), 32 * 16 * 3);
    }
}
