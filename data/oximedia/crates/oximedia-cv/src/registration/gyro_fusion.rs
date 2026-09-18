//! Gyroscope-visual complementary filter for video stabilization.
//!
//! Blends high-frequency gyroscope predictions with low-frequency visual
//! motion estimates using a first-order complementary filter.
//!
//! # Background
//!
//! Gyroscopes measure angular velocity with low noise at high frequency but
//! accumulate drift over time. Visual optical-flow estimates are accurate at
//! low frequency but noisy and subject to failure during fast motion or blur.
//! The complementary filter exploits the complementary spectral properties:
//!
//! ```text
//! θ_fused(t) = α · (θ_prev + Δθ_gyro) + (1 - α) · θ_visual
//! ```
//!
//! where `α ≈ 0.95..0.98` trusts the gyro for high-frequency content and the
//! visual estimate for slow, long-term drift correction.
//!
//! # References
//!
//! - Bouguet J-Y (2000). Pyramidal implementation of the Lucas-Kanade feature tracker.
//! - Karpenko et al. (2011). Digital Video Stabilization and Rolling Shutter
//!   Correction Using Gyroscopes. Stanford CSTR 2011-03.
//! - Hanning et al. (2011). Stabilizing Cell Phone Video Using Inertial
//!   Measurement Units. ICCV Workshops.

use crate::registration::{TransformMatrix, TransformationType};

/// A single gyroscope measurement.
#[derive(Debug, Clone, Copy)]
pub struct GyroSample {
    /// Timestamp in nanoseconds (monotonically increasing).
    pub t_ns: i64,
    /// Angular velocity in rad/s in the IMU body frame: [omega_x, omega_y, omega_z].
    ///
    /// Convention: omega_y is the yaw rate (rotation around vertical axis),
    /// omega_x is pitch, omega_z is roll.
    pub omega_xyz: [f32; 3],
}

/// Pinhole camera intrinsic parameters.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    /// Focal length in pixels along the x axis.
    pub fx: f32,
    /// Focal length in pixels along the y axis.
    pub fy: f32,
    /// Principal point x coordinate in pixels.
    pub cx: f32,
    /// Principal point y coordinate in pixels.
    pub cy: f32,
}

impl CameraIntrinsics {
    /// Create default intrinsics for a 1080p camera with 70° diagonal FoV.
    #[must_use]
    pub fn default_1080p() -> Self {
        Self {
            fx: 1500.0,
            fy: 1500.0,
            cx: 960.0,
            cy: 540.0,
        }
    }
}

/// State maintained by the complementary filter.
#[derive(Debug, Clone)]
struct FusionState {
    /// Current fused x-translation (pixels).
    tx: f64,
    /// Current fused y-translation (pixels).
    ty: f64,
    /// Current fused rotation angle (radians, yaw).
    theta: f64,
}

impl FusionState {
    fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            theta: 0.0,
        }
    }
}

/// Complementary filter blending gyroscope predictions with visual motion.
///
/// # Parameters
///
/// * `alpha` — High-pass weight for gyro channel (0.0..1.0).
///   Typical values: 0.95 (more visual correction) to 0.98 (mostly gyro).
/// * `intrinsics` — Camera focal lengths used to project angular velocity
///   into pixel-space translations.
/// * `fps` — Video frame rate (frames per second); used to compute the
///   integration time step `Δt = 1 / fps`.
///
/// # Example
///
/// ```
/// use oximedia_cv::registration::gyro_fusion::{
///     CameraIntrinsics, GyroFusionFilter, GyroSample,
/// };
///
/// let intrinsics = CameraIntrinsics::default_1080p();
/// let mut filter = GyroFusionFilter::new(0.97, intrinsics, 30.0);
///
/// let samples = vec![GyroSample { t_ns: 0, omega_xyz: [0.0, 0.0, 0.0] }];
/// let visual = oximedia_cv::registration::TransformMatrix::identity();
/// let fused = filter.step(visual, &samples);
/// ```
#[derive(Debug, Clone)]
pub struct GyroFusionFilter {
    /// High-pass weight for gyro (α).
    alpha: f64,
    /// Camera intrinsics for angular-to-pixel projection.
    intrinsics: CameraIntrinsics,
    /// Frame rate (Hz).
    fps: f64,
    /// Internal complementary filter state.
    state: FusionState,
}

impl GyroFusionFilter {
    /// Construct a new filter.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `fps` is not positive.
    #[must_use]
    pub fn new(alpha: f32, intrinsics: CameraIntrinsics, fps: f32) -> Self {
        debug_assert!(fps > 0.0, "fps must be positive");
        Self {
            alpha: alpha.clamp(0.0, 1.0) as f64,
            intrinsics,
            fps: fps as f64,
            state: FusionState::identity(),
        }
    }

    /// Fuse a visual motion estimate with gyroscope measurements for one frame.
    ///
    /// `gyro` should contain all samples spanning the current frame interval
    /// `[t_prev, t_curr]`. They are integrated in timestamp order.
    ///
    /// Returns the fused `TransformMatrix` (translation type) representing the
    /// stabilisation correction for the current frame.
    pub fn step(&mut self, visual: TransformMatrix, gyro: &[GyroSample]) -> TransformMatrix {
        let dt = 1.0 / self.fps;

        // Integrate gyro samples over the frame interval.
        let (delta_tx, delta_ty, _delta_theta) = integrate_gyro(gyro, dt, &self.intrinsics);

        // Gyro prediction: propagate previous state.
        let pred_tx = self.state.tx + delta_tx;
        let pred_ty = self.state.ty + delta_ty;

        // Visual estimate.
        let (vtx, vty) = visual.get_translation();

        // Complementary blend.
        let fused_tx = self.alpha * pred_tx + (1.0 - self.alpha) * vtx;
        let fused_ty = self.alpha * pred_ty + (1.0 - self.alpha) * vty;

        // Update state.
        self.state.tx = fused_tx;
        self.state.ty = fused_ty;

        TransformMatrix {
            data: [1.0, 0.0, fused_tx, 0.0, 1.0, fused_ty, 0.0, 0.0, 1.0],
            transform_type: TransformationType::Translation,
        }
    }

    /// Predict-only step used when no visual estimate is available (frame drop,
    /// motion blur, etc.).  The filter advances purely on the gyro signal.
    pub fn predict_only(&mut self, gyro: &[GyroSample]) -> TransformMatrix {
        let dt = 1.0 / self.fps;
        let (delta_tx, delta_ty, _delta_theta) = integrate_gyro(gyro, dt, &self.intrinsics);

        self.state.tx += delta_tx;
        self.state.ty += delta_ty;

        TransformMatrix {
            data: [
                1.0,
                0.0,
                self.state.tx,
                0.0,
                1.0,
                self.state.ty,
                0.0,
                0.0,
                1.0,
            ],
            transform_type: TransformationType::Translation,
        }
    }

    /// Reset filter state to identity.
    pub fn reset(&mut self) {
        self.state = FusionState::identity();
    }

    /// Read the current internal state translation (tx, ty).
    #[must_use]
    pub fn current_translation(&self) -> (f64, f64) {
        (self.state.tx, self.state.ty)
    }
}

/// Integrate a slice of gyro samples over a frame interval `dt`.
///
/// Returns `(Δtx_pixels, Δty_pixels, Δtheta_rad)`.
///
/// Linear interpolation between consecutive samples is used; the total
/// integrated time is `dt` regardless of sample count (samples are treated
/// as evenly spanning the interval when no timestamps span > dt).
///
/// Small-angle approximation:
/// - `Δtx = omega_y · fx · dt_total`  (yaw → horizontal pixel shift)
/// - `Δty = omega_x · fy · dt_total`  (pitch → vertical pixel shift)
fn integrate_gyro(
    samples: &[GyroSample],
    dt_frame: f64,
    intrinsics: &CameraIntrinsics,
) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let fx = intrinsics.fx as f64;
    let fy = intrinsics.fy as f64;

    let mut omega_x_int = 0.0_f64; // pitch integral (rad)
    let mut omega_y_int = 0.0_f64; // yaw integral (rad)
    let mut omega_z_int = 0.0_f64; // roll integral (rad)

    if samples.len() == 1 {
        // Single sample: assume it covers the entire frame interval.
        let s = &samples[0];
        omega_x_int = s.omega_xyz[0] as f64 * dt_frame;
        omega_y_int = s.omega_xyz[1] as f64 * dt_frame;
        omega_z_int = s.omega_xyz[2] as f64 * dt_frame;
    } else {
        // Multiple samples: integrate using timestamp differences.
        let t_start = samples[0].t_ns as f64 * 1e-9;
        let t_end = samples[samples.len() - 1].t_ns as f64 * 1e-9;
        let span = t_end - t_start;

        for i in 0..samples.len() - 1 {
            let t0 = samples[i].t_ns as f64 * 1e-9;
            let t1 = samples[i + 1].t_ns as f64 * 1e-9;
            let sub_dt = (t1 - t0).max(0.0);

            // Trapezoidal rule
            let ax = (samples[i].omega_xyz[0] + samples[i + 1].omega_xyz[0]) as f64 * 0.5;
            let ay = (samples[i].omega_xyz[1] + samples[i + 1].omega_xyz[1]) as f64 * 0.5;
            let az = (samples[i].omega_xyz[2] + samples[i + 1].omega_xyz[2]) as f64 * 0.5;

            omega_x_int += ax * sub_dt;
            omega_y_int += ay * sub_dt;
            omega_z_int += az * sub_dt;
        }

        // Rescale to actual frame dt if sample span doesn't match.
        if span > 1e-9 {
            let scale = dt_frame / span;
            omega_x_int *= scale;
            omega_y_int *= scale;
            omega_z_int *= scale;
        }
    }

    // Project angular motion to pixel space (small-angle approximation at unit depth).
    let delta_tx = omega_y_int * fx; // yaw → horizontal
    let delta_ty = omega_x_int * fy; // pitch → vertical (note: sign depends on convention)
    let delta_theta = omega_z_int; // roll stays in radians

    (delta_tx, delta_ty, delta_theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registration::TransformMatrix;

    fn make_intrinsics(fx: f32) -> CameraIntrinsics {
        CameraIntrinsics {
            fx,
            fy: fx,
            cx: 0.0,
            cy: 0.0,
        }
    }

    /// 1°/frame yaw at 60 fps with fx=500 should yield ≈8.73 px horizontal shift.
    #[test]
    fn test_gyro_fusion_pure_rotation() {
        let intrinsics = make_intrinsics(500.0);
        let mut filter = GyroFusionFilter::new(1.0, intrinsics, 60.0);

        // omega_y = 1°/frame * 60 fps = π/3 rad/s for one frame = π/180 rad
        let omega_y = (1.0_f32).to_radians() * 60.0; // rad/s
        let samples = vec![GyroSample {
            t_ns: 0,
            omega_xyz: [0.0, omega_y, 0.0],
        }];
        // alpha = 1.0 → pure gyro (no visual contribution)
        let visual = TransformMatrix::translation(0.0, 0.0);
        let fused = filter.step(visual, &samples);
        let (tx, _ty) = fused.get_translation();

        // Expected: π/180 * 500 ≈ 8.727 px
        let expected = std::f64::consts::PI / 180.0 * 500.0;
        assert!(
            (tx - expected).abs() < 1.0,
            "Expected x-translation ≈{expected:.3}, got {tx:.3}"
        );
    }

    /// Visual reports 100 px spurious shift; gyro reports near-zero.
    /// alpha=0.97 → fused output should be < 5 px.
    #[test]
    fn test_gyro_visual_disagree_alpha_high_trusts_gyro() {
        let intrinsics = make_intrinsics(500.0);
        // alpha=0.97 — highly trust gyro
        let mut filter = GyroFusionFilter::new(0.97, intrinsics, 30.0);

        // Near-zero gyro signal
        let samples = vec![GyroSample {
            t_ns: 0,
            omega_xyz: [0.0, 0.0, 0.0],
        }];
        // Visual reports 100 px shift
        let visual = TransformMatrix::translation(100.0, 0.0);
        let fused = filter.step(visual, &samples);
        let (tx, _ty) = fused.get_translation();

        // fused = 0.97 * 0.0 + 0.03 * 100.0 = 3.0
        assert!(
            tx < 5.0,
            "Expected fused tx < 5 px (alpha trusts gyro), got {tx:.3}"
        );
    }

    /// Predict-only: 10 frames of 1 rad/s gyro at 10 fps → 1 rad total, 500 px.
    #[test]
    fn test_gyro_only_predict_during_dropout() {
        let intrinsics = make_intrinsics(500.0);
        let mut filter = GyroFusionFilter::new(1.0, intrinsics, 10.0);

        let samples = vec![GyroSample {
            t_ns: 0,
            omega_xyz: [0.0, 1.0, 0.0], // 1 rad/s yaw
        }];

        // 10 frames, each adds 0.1 rad * 500 = 50 px
        for _ in 0..10 {
            filter.predict_only(&samples);
        }

        let (tx, _ty) = filter.current_translation();
        let expected = 1.0_f64 * 500.0; // 1 rad * fx
        assert!(
            (tx - expected).abs() < 2.0,
            "Expected cumulative tx ≈{expected:.1} px, got {tx:.3}"
        );
    }

    #[test]
    fn test_gyro_reset_clears_state() {
        let intrinsics = make_intrinsics(500.0);
        let mut filter = GyroFusionFilter::new(0.97, intrinsics, 30.0);

        let samples = vec![GyroSample {
            t_ns: 0,
            omega_xyz: [0.0, 1.0, 0.0],
        }];
        let visual = TransformMatrix::identity();
        filter.predict_only(&samples);
        filter.reset();

        let (tx, ty) = filter.current_translation();
        assert!((tx).abs() < 1e-9);
        assert!((ty).abs() < 1e-9);

        // step after reset should be near-zero for zero gyro
        let out = filter.step(
            visual,
            &[GyroSample {
                t_ns: 0,
                omega_xyz: [0.0, 0.0, 0.0],
            }],
        );
        let (tx2, _) = out.get_translation();
        assert!(tx2.abs() < 1e-6);
    }

    #[test]
    fn test_gyro_multi_sample_integration() {
        let intrinsics = make_intrinsics(1000.0);
        let mut filter = GyroFusionFilter::new(1.0, intrinsics, 1.0); // 1 fps for easy maths

        // Two samples spanning 1 second: both report 1 rad/s yaw
        let samples = vec![
            GyroSample {
                t_ns: 0,
                omega_xyz: [0.0, 1.0, 0.0],
            },
            GyroSample {
                t_ns: 1_000_000_000,
                omega_xyz: [0.0, 1.0, 0.0],
            },
        ];
        let visual = TransformMatrix::translation(0.0, 0.0);
        let fused = filter.step(visual, &samples);
        let (tx, _ty) = fused.get_translation();

        // 1 rad * 1000 px/rad = 1000 px
        assert!(
            (tx - 1000.0).abs() < 2.0,
            "Expected tx ≈ 1000 px, got {tx:.3}"
        );
    }
}
