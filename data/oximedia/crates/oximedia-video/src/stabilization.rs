//! Video stabilization with Kalman-filtered motion smoothing.
//!
//! Accepts per-frame affine transforms (translation, rotation, scale)
//! and smooths them through independent Kalman filters on each degree of
//! freedom.  The smoothed transforms can then be applied back to frames
//! to produce stabilized output.
//!
//! Features:
//! - Full-state Kalman filter (predict → update) with state [x, y, angle, scale]
//! - Crop-and-scale stabilization (compute max crop, inverse transform + scale)
//! - Rolling shutter correction estimation
//! - Stabilization quality metric (smoothness ratio)

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// 2D affine transform: translation (dx, dy), rotation angle (da, radians),
/// and scale factor (ds, 1.0 = no change).
#[derive(Debug, Clone, PartialEq)]
pub struct AffineTransform {
    /// Horizontal translation in pixels.
    pub dx: f32,
    /// Vertical translation in pixels.
    pub dy: f32,
    /// Rotation angle in radians.
    pub da: f32,
    /// Scale factor (1.0 = identity).
    pub ds: f32,
}

impl AffineTransform {
    /// Identity transform (no movement).
    pub fn identity() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            da: 0.0,
            ds: 1.0,
        }
    }

    /// Compose this transform with another (self applied first, then other).
    pub fn compose(&self, other: &AffineTransform) -> AffineTransform {
        let cos_a = other.da.cos();
        let sin_a = other.da.sin();
        AffineTransform {
            dx: other.ds * (cos_a * self.dx - sin_a * self.dy) + other.dx,
            dy: other.ds * (sin_a * self.dx + cos_a * self.dy) + other.dy,
            da: self.da + other.da,
            ds: self.ds * other.ds,
        }
    }

    /// Invert this affine transform.
    pub fn inverse(&self) -> AffineTransform {
        let inv_ds = if self.ds.abs() < 1e-9 {
            1.0
        } else {
            1.0 / self.ds
        };
        let cos_a = (-self.da).cos();
        let sin_a = (-self.da).sin();
        AffineTransform {
            dx: inv_ds * (cos_a * (-self.dx) - sin_a * (-self.dy)),
            dy: inv_ds * (sin_a * (-self.dx) + cos_a * (-self.dy)),
            da: -self.da,
            ds: inv_ds,
        }
    }
}

/// One stabilization record pairing the raw and Kalman-smoothed transforms.
#[derive(Debug, Clone)]
pub struct StabilizationFrame {
    /// Raw per-frame transform as provided by the caller.
    pub transform: AffineTransform,
    /// Kalman-smoothed version of the transform.
    pub smoothed: AffineTransform,
}

/// Configuration for the video stabilizer.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilizationConfig {
    /// Number of frames to consider for smoothing (affects Kalman tuning hints).
    pub smoothing_radius: u32,
    /// Fraction of the frame to keep after stabilization crop (e.g., 0.9).
    pub crop_ratio: f32,
}

// ---------------------------------------------------------------------------
// Full-state Kalman filter: [x, velocity_x]
// ---------------------------------------------------------------------------

/// Two-state Kalman filter tracking [position, velocity] for a single DOF.
///
/// State vector: `[x, v]` where `x` is the quantity (e.g. cumulative dx)
/// and `v` is its rate of change per frame.
struct KalmanState2 {
    /// State estimate: [position, velocity].
    x: [f32; 2],
    /// 2x2 error covariance (row-major).
    p: [f32; 4],
    /// Process noise scalar (scaled into Q matrix).
    q: f32,
    /// Measurement noise covariance.
    r: f32,
    /// Time step (1.0 for frame-by-frame).
    dt: f32,
}

impl KalmanState2 {
    fn new(q: f32, r: f32) -> Self {
        Self {
            x: [0.0, 0.0],
            p: [1.0, 0.0, 0.0, 1.0], // identity
            q,
            r,
            dt: 1.0,
        }
    }

    /// Predict step: propagate state and covariance forward.
    fn predict(&mut self) {
        let dt = self.dt;
        // x_new = F * x_old where F = [[1, dt], [0, 1]]
        let new_x0 = self.x[0] + dt * self.x[1];
        let new_x1 = self.x[1];
        self.x = [new_x0, new_x1];

        // P_new = F * P * F^T + Q
        let p00 = self.p[0];
        let p01 = self.p[1];
        let p10 = self.p[2];
        let p11 = self.p[3];

        let fp00 = p00 + dt * p10;
        let fp01 = p01 + dt * p11;
        let fp10 = p10;
        let fp11 = p11;

        // F * P * F^T
        let new_p00 = fp00 + dt * fp01;
        let new_p01 = fp01;
        let new_p10 = fp10 + dt * fp11;
        let new_p11 = fp11;

        // Add process noise Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;
        self.p[0] = new_p00 + self.q * dt4 / 4.0;
        self.p[1] = new_p01 + self.q * dt3 / 2.0;
        self.p[2] = new_p10 + self.q * dt3 / 2.0;
        self.p[3] = new_p11 + self.q * dt2;
    }

    /// Update step: incorporate a new measurement of the position.
    fn update(&mut self, measurement: f32) -> f32 {
        // H = [1, 0], so innovation = z - H*x = z - x[0]
        let innovation = measurement - self.x[0];

        // S = H * P * H^T + R = P[0][0] + R
        let s = self.p[0] + self.r;
        if s.abs() < 1e-12 {
            return self.x[0];
        }

        // K = P * H^T / S = [P[0][0], P[1][0]]^T / S
        let k0 = self.p[0] / s;
        let k1 = self.p[2] / s;

        // x = x + K * innovation
        self.x[0] += k0 * innovation;
        self.x[1] += k1 * innovation;

        // P = (I - K*H) * P
        let new_p00 = (1.0 - k0) * self.p[0];
        let new_p01 = (1.0 - k0) * self.p[1];
        let new_p10 = self.p[2] - k1 * self.p[0];
        let new_p11 = self.p[3] - k1 * self.p[1];

        self.p = [new_p00, new_p01, new_p10, new_p11];
        self.x[0]
    }

    /// Run predict then update and return the smoothed position.
    fn step(&mut self, measurement: f32) -> f32 {
        self.predict();
        self.update(measurement)
    }
}

// ---------------------------------------------------------------------------
// Internal Kalman filter (simple scalar, preserved for compatibility)
// ---------------------------------------------------------------------------

/// Single-variable discrete Kalman filter for scalar measurements.
struct KalmanState {
    /// Current state estimate.
    x: f32,
    /// Error covariance.
    p: f32,
    /// Process noise covariance.
    q: f32,
    /// Measurement noise covariance.
    r: f32,
}

impl KalmanState {
    fn new(q: f32, r: f32) -> Self {
        Self {
            x: 0.0,
            p: 1.0,
            q,
            r,
        }
    }

    /// Update the filter with a new measurement; returns the updated estimate.
    fn update(&mut self, measurement: f32) -> f32 {
        // Predict step
        self.p += self.q;
        // Update step
        let k = self.p / (self.p + self.r);
        self.x += k * (measurement - self.x);
        self.p *= 1.0 - k;
        self.x
    }
}

// ---------------------------------------------------------------------------
// Rolling shutter correction
// ---------------------------------------------------------------------------

/// Estimated rolling shutter parameters for a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct RollingShutterEstimate {
    /// Estimated horizontal skew (pixels) from top to bottom of frame.
    pub horizontal_skew: f32,
    /// Estimated vertical skew (pixels) from top to bottom.
    pub vertical_skew: f32,
    /// Estimated rotational skew (radians) accumulated across scanlines.
    pub rotational_skew: f32,
    /// Readout time as fraction of frame interval (0.0..1.0).
    pub readout_fraction: f32,
}

/// Estimate rolling shutter distortion from two consecutive frame transforms.
///
/// The idea: if the camera moved during the rolling shutter readout, each
/// scanline was exposed at a slightly different time, causing a skew
/// proportional to the velocity and the readout fraction.
pub fn estimate_rolling_shutter(
    transform_prev: &AffineTransform,
    transform_curr: &AffineTransform,
    readout_fraction: f32,
) -> RollingShutterEstimate {
    let rf = readout_fraction.clamp(0.0, 1.0);

    // Inter-frame motion (velocity proxy)
    let delta_dx = transform_curr.dx - transform_prev.dx;
    let delta_dy = transform_curr.dy - transform_prev.dy;
    let delta_da = transform_curr.da - transform_prev.da;

    // During readout, the camera continues to move. The skew is proportional
    // to the velocity times the readout time.
    RollingShutterEstimate {
        horizontal_skew: delta_dx * rf,
        vertical_skew: delta_dy * rf,
        rotational_skew: delta_da * rf,
        readout_fraction: rf,
    }
}

/// Correct rolling shutter in an RGBA frame by applying per-scanline
/// inverse shifts derived from the rolling shutter estimate.
///
/// `frame` must be `width * height * 4` bytes (RGBA).
pub fn correct_rolling_shutter(
    frame: &[u8],
    estimate: &RollingShutterEstimate,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let n_pixels = (width * height) as usize;
    let mut out = vec![0u8; n_pixels * 4];
    let h_f = height as f32;
    let w_max = (width.saturating_sub(1)) as i32;

    for y in 0..height {
        // Scanline normalized position [0, 1]
        let t = if h_f > 1.0 {
            y as f32 / (h_f - 1.0)
        } else {
            0.0
        };
        // Shift for this scanline (linearly interpolated)
        let shift_x = (estimate.horizontal_skew * (t - 0.5)).round() as i32;

        for x in 0..width {
            let src_x = (x as i32 + shift_x).clamp(0, w_max) as u32;
            let src_idx = (y * width + src_x) as usize * 4;
            let dst_idx = (y * width + x) as usize * 4;

            for ch in 0..4 {
                out[dst_idx + ch] = *frame.get(src_idx + ch).unwrap_or(&0);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Crop-and-scale stabilization
// ---------------------------------------------------------------------------

/// Result of crop-and-scale analysis for a stabilized sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct CropScaleResult {
    /// Maximum horizontal displacement (pixels) across the sequence.
    pub max_dx: f32,
    /// Maximum vertical displacement (pixels) across the sequence.
    pub max_dy: f32,
    /// Maximum rotation (radians, absolute) across the sequence.
    pub max_da: f32,
    /// Recommended scale factor to fill the frame after crop.
    pub fill_scale: f32,
    /// Effective crop rectangle (left, top, right, bottom) in normalized \[0,1\] coords.
    pub crop_rect: [f32; 4],
}

/// Analyze a series of stabilization corrections and compute the optimal
/// crop-and-scale parameters to fill the frame without black borders.
pub fn compute_crop_scale(
    corrections: &[AffineTransform],
    frame_width: u32,
    frame_height: u32,
) -> CropScaleResult {
    if corrections.is_empty() {
        return CropScaleResult {
            max_dx: 0.0,
            max_dy: 0.0,
            max_da: 0.0,
            fill_scale: 1.0,
            crop_rect: [0.0, 0.0, 1.0, 1.0],
        };
    }

    let mut max_dx: f32 = 0.0;
    let mut max_dy: f32 = 0.0;
    let mut max_da: f32 = 0.0;

    for c in corrections {
        max_dx = max_dx.max(c.dx.abs());
        max_dy = max_dy.max(c.dy.abs());
        max_da = max_da.max(c.da.abs());
    }

    let fw = frame_width as f32;
    let fh = frame_height as f32;

    // The diagonal half-length rotated by max_da gives additional displacement
    let diag = (fw * fw + fh * fh).sqrt() / 2.0;
    let rotation_margin_x = diag * max_da.sin().abs();
    let rotation_margin_y = diag * max_da.sin().abs();

    let total_margin_x = max_dx + rotation_margin_x;
    let total_margin_y = max_dy + rotation_margin_y;

    // Scale factor to fill the frame
    let scale_x = if fw > 2.0 * total_margin_x {
        fw / (fw - 2.0 * total_margin_x)
    } else {
        2.0 // limit
    };
    let scale_y = if fh > 2.0 * total_margin_y {
        fh / (fh - 2.0 * total_margin_y)
    } else {
        2.0
    };
    let fill_scale = scale_x.max(scale_y).min(2.0);

    // Crop rectangle in normalized coordinates
    let crop_x = total_margin_x / fw;
    let crop_y = total_margin_y / fh;
    let crop_rect = [
        crop_x.min(0.49),
        crop_y.min(0.49),
        (1.0 - crop_x).max(0.51),
        (1.0 - crop_y).max(0.51),
    ];

    CropScaleResult {
        max_dx,
        max_dy,
        max_da,
        fill_scale,
        crop_rect,
    }
}

/// Apply crop-and-scale stabilization to a single RGBA frame.
///
/// Applies the inverse of `correction`, then scales up by `fill_scale`
/// so the cropped area fills the entire output dimensions.
///
/// `frame` must be `width * height * 4` bytes (RGBA).
pub fn apply_crop_scale(
    frame: &[u8],
    correction: &AffineTransform,
    fill_scale: f32,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let n_pixels = (width * height) as usize;
    let mut out = vec![0u8; n_pixels * 4];

    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;

    let inv = correction.inverse();
    let total_scale = fill_scale * inv.ds;
    let cos_a = inv.da.cos();
    let sin_a = inv.da.sin();

    let safe_scale = if total_scale.abs() < 1e-6 {
        1.0
    } else {
        total_scale
    };

    let w_max = (width.saturating_sub(1)) as i32;
    let h_max = (height.saturating_sub(1)) as i32;

    for oy in 0..height {
        for ox in 0..width {
            // Map output pixel to input through inverse correction + scale
            let px = (ox as f32 - cx) / safe_scale;
            let py = (oy as f32 - cy) / safe_scale;

            let sx_f = cx + cos_a * px + sin_a * py - inv.dx;
            let sy_f = cy + (-sin_a * px + cos_a * py) - inv.dy;

            let sx = (sx_f.round() as i32).clamp(0, w_max) as u32;
            let sy = (sy_f.round() as i32).clamp(0, h_max) as u32;

            let src_idx = (sy * width + sx) as usize * 4;
            let dst_idx = (oy * width + ox) as usize * 4;

            for ch in 0..4 {
                out[dst_idx + ch] = *frame.get(src_idx + ch).unwrap_or(&0);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Stabilization quality metric
// ---------------------------------------------------------------------------

/// Quality metric for a stabilized sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilizationQuality {
    /// Jitter of the original (raw) motion trajectory.
    pub original_jitter: f32,
    /// Jitter of the stabilized (smoothed) motion trajectory.
    pub stabilized_jitter: f32,
    /// Smoothness ratio: `stabilized_jitter / original_jitter`.
    /// Lower is better (0.0 = perfect stabilization, 1.0 = no improvement).
    pub smoothness_ratio: f32,
    /// Maximum correction magnitude applied.
    pub max_correction: f32,
}

/// Compute frame-to-frame jitter as RMS of inter-frame deltas.
fn compute_jitter(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mut sum_sq = 0.0f64;
    for i in 1..values.len() {
        let delta = (values[i] - values[i - 1]) as f64;
        sum_sq += delta * delta;
    }
    let rms = (sum_sq / (values.len() - 1) as f64).sqrt();
    rms as f32
}

/// Compute stabilization quality metrics from a sequence of stabilization frames.
pub fn compute_stabilization_quality(frames: &[StabilizationFrame]) -> StabilizationQuality {
    if frames.is_empty() {
        return StabilizationQuality {
            original_jitter: 0.0,
            stabilized_jitter: 0.0,
            smoothness_ratio: 1.0,
            max_correction: 0.0,
        };
    }

    let orig_dx: Vec<f32> = frames.iter().map(|f| f.transform.dx).collect();
    let orig_dy: Vec<f32> = frames.iter().map(|f| f.transform.dy).collect();
    let smooth_dx: Vec<f32> = frames.iter().map(|f| f.smoothed.dx).collect();
    let smooth_dy: Vec<f32> = frames.iter().map(|f| f.smoothed.dy).collect();

    let orig_jitter_x = compute_jitter(&orig_dx);
    let orig_jitter_y = compute_jitter(&orig_dy);
    let smooth_jitter_x = compute_jitter(&smooth_dx);
    let smooth_jitter_y = compute_jitter(&smooth_dy);

    let original_jitter = (orig_jitter_x * orig_jitter_x + orig_jitter_y * orig_jitter_y).sqrt();
    let stabilized_jitter =
        (smooth_jitter_x * smooth_jitter_x + smooth_jitter_y * smooth_jitter_y).sqrt();

    let smoothness_ratio = if original_jitter > 1e-9 {
        (stabilized_jitter / original_jitter).min(10.0)
    } else {
        1.0
    };

    let mut max_correction: f32 = 0.0;
    for f in frames {
        let cdx = f.smoothed.dx - f.transform.dx;
        let cdy = f.smoothed.dy - f.transform.dy;
        let mag = (cdx * cdx + cdy * cdy).sqrt();
        max_correction = max_correction.max(mag);
    }

    StabilizationQuality {
        original_jitter,
        stabilized_jitter,
        smoothness_ratio,
        max_correction,
    }
}

// ---------------------------------------------------------------------------
// VideoStabilizer
// ---------------------------------------------------------------------------

/// Video stabilizer that accumulates per-frame transforms and smooths them
/// through independent Kalman filters on dx, dy, da, and ds.
pub struct VideoStabilizer {
    /// Configuration.
    pub config: StabilizationConfig,
    /// Raw per-frame affine transforms (one per frame).
    pub transforms: Vec<AffineTransform>,
    // Kalman filters — reset on each call to get_smoothed_transforms.
    kalman_dx: KalmanState,
    kalman_dy: KalmanState,
    kalman_da: KalmanState,
    kalman_ds: KalmanState,
}

impl VideoStabilizer {
    /// Create a new stabilizer with the given configuration.
    pub fn new(config: StabilizationConfig) -> Self {
        // q=0.0001 (trust the model), r=0.01 (moderate measurement noise)
        Self {
            config,
            transforms: Vec::new(),
            kalman_dx: KalmanState::new(0.0001, 0.01),
            kalman_dy: KalmanState::new(0.0001, 0.01),
            kalman_da: KalmanState::new(0.0001, 0.01),
            kalman_ds: KalmanState::new(0.0001, 0.01),
        }
    }

    /// Append a raw per-frame affine transform.
    pub fn add_frame_transform(&mut self, t: AffineTransform) {
        self.transforms.push(t);
    }

    /// Return the number of accumulated transforms.
    pub fn frame_count(&self) -> usize {
        self.transforms.len()
    }

    /// Compute and return smoothed transforms for all accumulated frames.
    ///
    /// The Kalman filters are reset at the beginning of each call so the
    /// result is deterministic regardless of call order.
    pub fn get_smoothed_transforms(&mut self) -> Vec<StabilizationFrame> {
        // Reset Kalman state so the function is repeatable.
        self.kalman_dx = KalmanState::new(0.0001, 0.01);
        self.kalman_dy = KalmanState::new(0.0001, 0.01);
        self.kalman_da = KalmanState::new(0.0001, 0.01);
        self.kalman_ds = KalmanState::new(0.0001, 0.01);

        let mut result = Vec::with_capacity(self.transforms.len());

        for t in &self.transforms {
            let sdx = self.kalman_dx.update(t.dx);
            let sdy = self.kalman_dy.update(t.dy);
            let sda = self.kalman_da.update(t.da);
            let sds = self.kalman_ds.update(t.ds);

            result.push(StabilizationFrame {
                transform: t.clone(),
                smoothed: AffineTransform {
                    dx: sdx,
                    dy: sdy,
                    da: sda,
                    ds: sds,
                },
            });
        }

        result
    }

    /// Compute smoothed transforms using the full-state (position+velocity) Kalman filter.
    ///
    /// This uses `KalmanState2` which models velocity, giving better prediction for
    /// camera motions with constant or slowly changing velocity.
    pub fn get_smoothed_transforms_full_state(&self) -> Vec<StabilizationFrame> {
        let mut kf_dx = KalmanState2::new(0.001, 0.1);
        let mut kf_dy = KalmanState2::new(0.001, 0.1);
        let mut kf_da = KalmanState2::new(0.0001, 0.01);
        let mut kf_ds = KalmanState2::new(0.0001, 0.01);

        // Initialize ds filter to expect ~1.0
        kf_ds.x[0] = 1.0;

        let mut result = Vec::with_capacity(self.transforms.len());

        for t in &self.transforms {
            let sdx = kf_dx.step(t.dx);
            let sdy = kf_dy.step(t.dy);
            let sda = kf_da.step(t.da);
            let sds = kf_ds.step(t.ds);

            result.push(StabilizationFrame {
                transform: t.clone(),
                smoothed: AffineTransform {
                    dx: sdx,
                    dy: sdy,
                    da: sda,
                    ds: sds,
                },
            });
        }

        result
    }

    /// Compute the stabilization correction needed to map `original` to `smoothed`.
    ///
    /// The correction describes what additional transform must be applied to
    /// compensate for the difference between the raw and smoothed motion.
    pub fn compute_correction(
        original: &AffineTransform,
        smoothed: &AffineTransform,
    ) -> AffineTransform {
        AffineTransform {
            dx: smoothed.dx - original.dx,
            dy: smoothed.dy - original.dy,
            da: smoothed.da - original.da,
            ds: smoothed.ds / original.ds.abs().max(1e-6),
        }
    }

    /// Apply an affine transform to an RGBA frame and return the result.
    ///
    /// Uses nearest-neighbour sampling.  Pixels that map outside the source
    /// bounds are clamped to the nearest edge pixel.
    ///
    /// `frame` must be `width * height * 4` bytes (RGBA).
    pub fn apply_transform(
        frame: &[u8],
        transform: &AffineTransform,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let n_pixels = (width * height) as usize;
        let mut out = vec![0u8; n_pixels * 4];

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;

        let cos_a = transform.da.cos();
        let sin_a = transform.da.sin();
        let scale = if transform.ds.abs() < 1e-6 {
            1.0f32
        } else {
            transform.ds
        };

        let w_max = (width.saturating_sub(1)) as i32;
        let h_max = (height.saturating_sub(1)) as i32;

        for oy in 0..height {
            for ox in 0..width {
                // Translate to centre, undo translation, rotate by -da, scale by 1/ds.
                let tx = ox as f32 - cx - transform.dx;
                let ty = oy as f32 - cy - transform.dy;

                let sx_f = cx + (cos_a * tx + sin_a * ty) / scale;
                let sy_f = cy + (-sin_a * tx + cos_a * ty) / scale;

                let sx = (sx_f.round() as i32).clamp(0, w_max) as u32;
                let sy = (sy_f.round() as i32).clamp(0, h_max) as u32;

                let src_idx = (sy * width + sx) as usize * 4;
                let dst_idx = (oy * width + ox) as usize * 4;

                for ch in 0..4 {
                    out[dst_idx + ch] = *frame.get(src_idx + ch).unwrap_or(&0);
                }
            }
        }

        out
    }

    /// Apply bilinear-interpolated transform (higher quality than nearest-neighbour).
    ///
    /// `frame` must be `width * height * 4` bytes (RGBA).
    pub fn apply_transform_bilinear(
        frame: &[u8],
        transform: &AffineTransform,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let n_pixels = (width * height) as usize;
        let mut out = vec![0u8; n_pixels * 4];

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;

        let cos_a = transform.da.cos();
        let sin_a = transform.da.sin();
        let scale = if transform.ds.abs() < 1e-6 {
            1.0f32
        } else {
            transform.ds
        };

        let w_max = (width.saturating_sub(1)) as f32;
        let h_max = (height.saturating_sub(1)) as f32;

        for oy in 0..height {
            for ox in 0..width {
                let tx = ox as f32 - cx - transform.dx;
                let ty = oy as f32 - cy - transform.dy;

                let sx_f = cx + (cos_a * tx + sin_a * ty) / scale;
                let sy_f = cy + (-sin_a * tx + cos_a * ty) / scale;

                let sx_clamped = sx_f.clamp(0.0, w_max);
                let sy_clamped = sy_f.clamp(0.0, h_max);

                let x0 = sx_clamped.floor() as u32;
                let y0 = sy_clamped.floor() as u32;
                let x1 = (x0 + 1).min(width.saturating_sub(1));
                let y1 = (y0 + 1).min(height.saturating_sub(1));

                let fx = sx_clamped - x0 as f32;
                let fy = sy_clamped - y0 as f32;

                let dst_idx = (oy * width + ox) as usize * 4;

                for ch in 0..4 {
                    let i00 = (y0 * width + x0) as usize * 4 + ch;
                    let i10 = (y0 * width + x1) as usize * 4 + ch;
                    let i01 = (y1 * width + x0) as usize * 4 + ch;
                    let i11 = (y1 * width + x1) as usize * 4 + ch;

                    let v00 = *frame.get(i00).unwrap_or(&0) as f32;
                    let v10 = *frame.get(i10).unwrap_or(&0) as f32;
                    let v01 = *frame.get(i01).unwrap_or(&0) as f32;
                    let v11 = *frame.get(i11).unwrap_or(&0) as f32;

                    let top = v00 * (1.0 - fx) + v10 * fx;
                    let bot = v01 * (1.0 - fx) + v11 * fx;
                    let val = top * (1.0 - fy) + bot * fy;

                    out[dst_idx + ch] = val.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> StabilizationConfig {
        StabilizationConfig {
            smoothing_radius: 30,
            crop_ratio: 0.9,
        }
    }

    fn make_rgba(width: usize, height: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(width * height * 4);
        for _ in 0..(width * height) {
            v.extend_from_slice(&[r, g, b, a]);
        }
        v
    }

    // 1. New stabilizer has zero frames
    #[test]
    fn test_new_creates_empty_stabilizer() {
        let stab = VideoStabilizer::new(default_config());
        assert_eq!(stab.frame_count(), 0);
    }

    // 2. add_frame_transform increments count
    #[test]
    fn test_add_frame_transform_increments_count() {
        let mut stab = VideoStabilizer::new(default_config());
        stab.add_frame_transform(AffineTransform::identity());
        stab.add_frame_transform(AffineTransform::identity());
        assert_eq!(stab.frame_count(), 2);
    }

    // 3. get_smoothed_transforms returns one entry per frame
    #[test]
    fn test_get_smoothed_transforms_count_matches() {
        let mut stab = VideoStabilizer::new(default_config());
        for _ in 0..5 {
            stab.add_frame_transform(AffineTransform::identity());
        }
        let smoothed = stab.get_smoothed_transforms();
        assert_eq!(smoothed.len(), 5);
    }

    // 4. Zero-motion transforms → smoothed ≈ zero (dx/dy/da ≈ 0, ds ≈ 0 initially)
    #[test]
    fn test_get_smoothed_transforms_zero_motion() {
        let mut stab = VideoStabilizer::new(default_config());
        for _ in 0..10 {
            stab.add_frame_transform(AffineTransform::identity());
        }
        let result = stab.get_smoothed_transforms();
        // With identity (ds=1.0) inputs, smoothed ds converges towards 1.0
        for sf in &result {
            assert!(
                sf.smoothed.dx.abs() < 0.5,
                "dx should be near 0: {}",
                sf.smoothed.dx
            );
            assert!(
                sf.smoothed.dy.abs() < 0.5,
                "dy should be near 0: {}",
                sf.smoothed.dy
            );
        }
    }

    // 5. Constant dx=10.0 → smoothed converges to ~10.0
    #[test]
    fn test_get_smoothed_transforms_constant_dx() {
        let mut stab = VideoStabilizer::new(default_config());
        for _ in 0..50 {
            stab.add_frame_transform(AffineTransform {
                dx: 10.0,
                dy: 0.0,
                da: 0.0,
                ds: 1.0,
            });
        }
        let result = stab.get_smoothed_transforms();
        // After 50 identical measurements, Kalman should converge close to 10.0
        let last = &result.last().expect("should have results").smoothed;
        assert!(
            (last.dx - 10.0).abs() < 1.0,
            "smoothed dx should be near 10.0, got {}",
            last.dx
        );
    }

    // 6. Kalman smoothing reduces noise in dx
    #[test]
    fn test_kalman_smoothing_reduces_noise() {
        let noisy_dx: Vec<f32> = vec![
            0.0, 5.0, -3.0, 8.0, -6.0, 2.0, 10.0, -4.0, 7.0, -1.0, 1.0, -5.0, 4.0, -8.0, 6.0, -2.0,
            3.0, -7.0, 9.0, -3.0,
        ];
        let input_variance: f32 = {
            let mean = noisy_dx.iter().sum::<f32>() / noisy_dx.len() as f32;
            noisy_dx.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / noisy_dx.len() as f32
        };

        let mut stab = VideoStabilizer::new(default_config());
        for dx in &noisy_dx {
            stab.add_frame_transform(AffineTransform {
                dx: *dx,
                dy: 0.0,
                da: 0.0,
                ds: 1.0,
            });
        }
        let result = stab.get_smoothed_transforms();
        let smoothed_vals: Vec<f32> = result.iter().map(|sf| sf.smoothed.dx).collect();
        let mean_s = smoothed_vals.iter().sum::<f32>() / smoothed_vals.len() as f32;
        let smoothed_variance: f32 = smoothed_vals
            .iter()
            .map(|x| (x - mean_s).powi(2))
            .sum::<f32>()
            / smoothed_vals.len() as f32;

        assert!(
            smoothed_variance < input_variance,
            "smoothed variance {} should be less than input variance {}",
            smoothed_variance,
            input_variance
        );
    }

    // 7. Correction is zero when original == smoothed
    #[test]
    fn test_compute_correction_zero() {
        let t = AffineTransform {
            dx: 5.0,
            dy: -3.0,
            da: 0.1,
            ds: 1.0,
        };
        let corr = VideoStabilizer::compute_correction(&t, &t);
        assert!((corr.dx).abs() < 1e-5);
        assert!((corr.dy).abs() < 1e-5);
        assert!((corr.da).abs() < 1e-5);
        assert!((corr.ds - 1.0).abs() < 1e-5);
    }

    // 8. Correction for known translation
    #[test]
    fn test_compute_correction_translation() {
        let orig = AffineTransform {
            dx: 10.0,
            dy: 5.0,
            da: 0.0,
            ds: 1.0,
        };
        let smoothed = AffineTransform {
            dx: 8.0,
            dy: 4.0,
            da: 0.0,
            ds: 1.0,
        };
        let corr = VideoStabilizer::compute_correction(&orig, &smoothed);
        assert!(
            (corr.dx - (-2.0)).abs() < 1e-5,
            "dx correction: {}",
            corr.dx
        );
        assert!(
            (corr.dy - (-1.0)).abs() < 1e-5,
            "dy correction: {}",
            corr.dy
        );
    }

    // 9. Identity transform returns same frame
    #[test]
    fn test_apply_transform_identity() {
        let frame = make_rgba(8, 8, 100, 150, 200, 255);
        let id = AffineTransform::identity();
        let out = VideoStabilizer::apply_transform(&frame, &id, 8, 8);
        assert_eq!(out, frame);
    }

    // 10. Output size matches input size
    #[test]
    fn test_apply_transform_output_size() {
        let frame = make_rgba(16, 12, 80, 80, 80, 255);
        let t = AffineTransform::identity();
        let out = VideoStabilizer::apply_transform(&frame, &t, 16, 12);
        assert_eq!(out.len(), 16 * 12 * 4);
    }

    // 11. Translation shifts pixels
    #[test]
    fn test_apply_transform_translation() {
        let mut frame = vec![0u8; 16 * 16 * 4];
        // Put a bright pixel at (8, 8)
        let idx = (8 * 16 + 8) * 4;
        frame[idx] = 255;
        frame[idx + 3] = 255;

        let t = AffineTransform {
            dx: 2.0,
            dy: 2.0,
            da: 0.0,
            ds: 1.0,
        };
        let out = VideoStabilizer::apply_transform(&frame, &t, 16, 16);
        let dst_idx = (10 * 16 + 10) * 4;
        assert_eq!(out[dst_idx], 255, "bright pixel should appear at (10,10)");
    }

    // 12. StabilizationFrame derives Debug
    #[test]
    fn test_stabilization_frame_debug() {
        let sf = StabilizationFrame {
            transform: AffineTransform::identity(),
            smoothed: AffineTransform::identity(),
        };
        let s = format!("{:?}", sf);
        assert!(s.contains("StabilizationFrame"));
    }

    // 13. AffineTransform can be cloned
    #[test]
    fn test_affine_transform_clone() {
        let t = AffineTransform {
            dx: 1.5,
            dy: -2.3,
            da: 0.05,
            ds: 0.98,
        };
        let c = t.clone();
        assert_eq!(t, c);
    }

    // 14. StabilizationConfig fields are accessible
    #[test]
    fn test_config_default_values() {
        let cfg = StabilizationConfig {
            smoothing_radius: 15,
            crop_ratio: 0.85,
        };
        assert_eq!(cfg.smoothing_radius, 15);
        assert!((cfg.crop_ratio - 0.85).abs() < 1e-5);
    }

    // 15. get_smoothed_transforms returns Vec<StabilizationFrame> (type-level test)
    #[test]
    fn test_get_smoothed_returns_stabilization_frames() {
        let mut stab = VideoStabilizer::new(default_config());
        stab.add_frame_transform(AffineTransform::identity());
        let result: Vec<StabilizationFrame> = stab.get_smoothed_transforms();
        assert_eq!(result.len(), 1);
    }

    // 16. Kalman convergence: after many identical measurements, estimate is close
    #[test]
    fn test_kalman_convergence() {
        let mut stab = VideoStabilizer::new(default_config());
        for _ in 0..200 {
            stab.add_frame_transform(AffineTransform {
                dx: 7.0,
                dy: 3.0,
                da: 0.0,
                ds: 1.0,
            });
        }
        let result = stab.get_smoothed_transforms();
        let last = result.last().expect("should have results");
        assert!(
            (last.smoothed.dx - 7.0).abs() < 0.5,
            "Kalman dx should converge to 7.0, got {}",
            last.smoothed.dx
        );
        assert!(
            (last.smoothed.dy - 3.0).abs() < 0.5,
            "Kalman dy should converge to 3.0, got {}",
            last.smoothed.dy
        );
    }

    // 17. After a shift, output is still valid RGBA (all opaque pixels remain opaque)
    #[test]
    fn test_apply_transform_all_opaque_after_shift() {
        let frame = make_rgba(16, 16, 128, 128, 128, 255);
        let t = AffineTransform {
            dx: 3.0,
            dy: 1.0,
            da: 0.0,
            ds: 1.0,
        };
        let out = VideoStabilizer::apply_transform(&frame, &t, 16, 16);
        // All sampled pixels come from the uniform frame → alpha should be 255
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[3], 255, "alpha should be 255");
        }
    }

    // === NEW TESTS ===

    // 18. Full-state Kalman (predict→update) converges to constant
    #[test]
    fn test_full_state_kalman_converges() {
        let mut stab = VideoStabilizer::new(default_config());
        for _ in 0..100 {
            stab.add_frame_transform(AffineTransform {
                dx: 5.0,
                dy: -2.0,
                da: 0.0,
                ds: 1.0,
            });
        }
        let result = stab.get_smoothed_transforms_full_state();
        let last = result.last().expect("should have results");
        assert!(
            (last.smoothed.dx - 5.0).abs() < 1.0,
            "full-state dx should converge to 5.0, got {}",
            last.smoothed.dx
        );
        assert!(
            (last.smoothed.dy - (-2.0)).abs() < 1.0,
            "full-state dy should converge to -2.0, got {}",
            last.smoothed.dy
        );
    }

    // 19. Full-state Kalman reduces noise
    #[test]
    fn test_full_state_kalman_reduces_noise() {
        let noisy: Vec<f32> = vec![
            3.0, -2.0, 5.0, 1.0, -4.0, 6.0, 0.0, -3.0, 4.0, -1.0, 2.0, -5.0, 7.0, 1.0, -6.0, 3.0,
            -2.0, 8.0, 0.0, -4.0,
        ];
        let mean_n = noisy.iter().sum::<f32>() / noisy.len() as f32;
        let var_n: f32 =
            noisy.iter().map(|x| (x - mean_n).powi(2)).sum::<f32>() / noisy.len() as f32;

        let mut stab = VideoStabilizer::new(default_config());
        for &dx in &noisy {
            stab.add_frame_transform(AffineTransform {
                dx,
                dy: 0.0,
                da: 0.0,
                ds: 1.0,
            });
        }
        let result = stab.get_smoothed_transforms_full_state();
        let smooth: Vec<f32> = result.iter().map(|f| f.smoothed.dx).collect();
        let mean_s = smooth.iter().sum::<f32>() / smooth.len() as f32;
        let var_s: f32 =
            smooth.iter().map(|x| (x - mean_s).powi(2)).sum::<f32>() / smooth.len() as f32;

        assert!(
            var_s < var_n,
            "full-state smoothed variance {} should be < input variance {}",
            var_s,
            var_n
        );
    }

    // 20. Rolling shutter estimation with no motion → zero skew
    #[test]
    fn test_rolling_shutter_no_motion() {
        let t0 = AffineTransform::identity();
        let t1 = AffineTransform::identity();
        let est = estimate_rolling_shutter(&t0, &t1, 0.5);
        assert!((est.horizontal_skew).abs() < 1e-6);
        assert!((est.vertical_skew).abs() < 1e-6);
        assert!((est.rotational_skew).abs() < 1e-6);
    }

    // 21. Rolling shutter estimation with horizontal motion
    #[test]
    fn test_rolling_shutter_horizontal_motion() {
        let t0 = AffineTransform {
            dx: 0.0,
            dy: 0.0,
            da: 0.0,
            ds: 1.0,
        };
        let t1 = AffineTransform {
            dx: 10.0,
            dy: 0.0,
            da: 0.0,
            ds: 1.0,
        };
        let est = estimate_rolling_shutter(&t0, &t1, 0.5);
        assert!(
            (est.horizontal_skew - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            est.horizontal_skew
        );
    }

    // 22. Rolling shutter correction preserves frame size
    #[test]
    fn test_rolling_shutter_correction_size() {
        let frame = make_rgba(16, 16, 128, 128, 128, 255);
        let est = RollingShutterEstimate {
            horizontal_skew: 4.0,
            vertical_skew: 0.0,
            rotational_skew: 0.0,
            readout_fraction: 0.5,
        };
        let out = correct_rolling_shutter(&frame, &est, 16, 16);
        assert_eq!(out.len(), 16 * 16 * 4);
    }

    // 23. Rolling shutter correction with zero skew returns same frame
    #[test]
    fn test_rolling_shutter_correction_zero_skew() {
        let frame = make_rgba(8, 8, 200, 100, 50, 255);
        let est = RollingShutterEstimate {
            horizontal_skew: 0.0,
            vertical_skew: 0.0,
            rotational_skew: 0.0,
            readout_fraction: 0.5,
        };
        let out = correct_rolling_shutter(&frame, &est, 8, 8);
        assert_eq!(out, frame);
    }

    // 24. Crop-scale with no corrections → identity
    #[test]
    fn test_crop_scale_empty_corrections() {
        let result = compute_crop_scale(&[], 1920, 1080);
        assert!((result.fill_scale - 1.0).abs() < 1e-5);
        assert!((result.max_dx).abs() < 1e-5);
    }

    // 25. Crop-scale computes non-trivial scale for large corrections
    #[test]
    fn test_crop_scale_nontrivial() {
        let corrections = vec![
            AffineTransform {
                dx: 20.0,
                dy: 10.0,
                da: 0.02,
                ds: 1.0,
            },
            AffineTransform {
                dx: -15.0,
                dy: -8.0,
                da: -0.01,
                ds: 1.0,
            },
            AffineTransform {
                dx: 25.0,
                dy: 12.0,
                da: 0.03,
                ds: 1.0,
            },
        ];
        let result = compute_crop_scale(&corrections, 1920, 1080);
        assert!(result.max_dx > 0.0);
        assert!(
            result.fill_scale > 1.0,
            "fill_scale should be > 1.0, got {}",
            result.fill_scale
        );
        assert!(result.crop_rect[0] > 0.0);
        assert!(result.crop_rect[2] < 1.0);
    }

    // 26. apply_crop_scale preserves frame dimensions
    #[test]
    fn test_apply_crop_scale_size() {
        let frame = make_rgba(16, 16, 128, 128, 128, 255);
        let correction = AffineTransform {
            dx: 2.0,
            dy: 1.0,
            da: 0.0,
            ds: 1.0,
        };
        let out = apply_crop_scale(&frame, &correction, 1.05, 16, 16);
        assert_eq!(out.len(), 16 * 16 * 4);
    }

    // 27. apply_crop_scale with identity correction and scale=1.0 → same frame
    #[test]
    fn test_apply_crop_scale_identity() {
        let frame = make_rgba(8, 8, 100, 150, 200, 255);
        let correction = AffineTransform::identity();
        let out = apply_crop_scale(&frame, &correction, 1.0, 8, 8);
        assert_eq!(out, frame);
    }

    // 28. Quality metric: identical trajectories → smoothness_ratio = 1.0
    #[test]
    fn test_quality_metric_identical() {
        let frames: Vec<StabilizationFrame> = (0..10)
            .map(|_| StabilizationFrame {
                transform: AffineTransform::identity(),
                smoothed: AffineTransform::identity(),
            })
            .collect();
        let quality = compute_stabilization_quality(&frames);
        assert!(
            (quality.smoothness_ratio - 1.0).abs() < 1e-5,
            "smoothness ratio should be 1.0 for identical trajectories"
        );
    }

    // 29. Quality metric: stabilization reduces jitter
    #[test]
    fn test_quality_metric_stabilized() {
        let mut stab = VideoStabilizer::new(default_config());
        let noisy = [
            0.0f32, 5.0, -3.0, 8.0, -6.0, 2.0, 10.0, -4.0, 7.0, -1.0, 3.0, -5.0, 4.0, -8.0, 6.0,
            -2.0, 3.0, -7.0, 9.0, -3.0,
        ];
        for &dx in &noisy {
            stab.add_frame_transform(AffineTransform {
                dx,
                dy: dx * 0.5,
                da: 0.0,
                ds: 1.0,
            });
        }
        let frames = stab.get_smoothed_transforms();
        let quality = compute_stabilization_quality(&frames);
        assert!(
            quality.smoothness_ratio < 1.0,
            "smoothness ratio {} should be < 1.0",
            quality.smoothness_ratio
        );
        assert!(quality.stabilized_jitter < quality.original_jitter);
    }

    // 30. Quality metric: empty input
    #[test]
    fn test_quality_metric_empty() {
        let quality = compute_stabilization_quality(&[]);
        assert!((quality.original_jitter).abs() < 1e-6);
        assert!((quality.smoothness_ratio - 1.0).abs() < 1e-5);
    }

    // 31. AffineTransform inverse round-trip
    #[test]
    fn test_affine_inverse_round_trip() {
        let t = AffineTransform {
            dx: 3.0,
            dy: -2.0,
            da: 0.1,
            ds: 1.2,
        };
        let inv = t.inverse();
        let composed = t.compose(&inv);
        assert!((composed.dx).abs() < 0.1, "dx: {}", composed.dx);
        assert!((composed.dy).abs() < 0.1, "dy: {}", composed.dy);
        assert!((composed.da).abs() < 0.01, "da: {}", composed.da);
        assert!((composed.ds - 1.0).abs() < 0.01, "ds: {}", composed.ds);
    }

    // 32. AffineTransform compose with identity is identity
    #[test]
    fn test_affine_compose_identity() {
        let t = AffineTransform {
            dx: 5.0,
            dy: -3.0,
            da: 0.05,
            ds: 1.1,
        };
        let id = AffineTransform::identity();
        let composed = t.compose(&id);
        assert!((composed.dx - t.dx).abs() < 1e-5);
        assert!((composed.dy - t.dy).abs() < 1e-5);
        assert!((composed.da - t.da).abs() < 1e-5);
        assert!((composed.ds - t.ds).abs() < 1e-5);
    }

    // 33. Bilinear transform with identity returns same frame
    #[test]
    fn test_bilinear_transform_identity() {
        let frame = make_rgba(8, 8, 100, 150, 200, 255);
        let id = AffineTransform::identity();
        let out = VideoStabilizer::apply_transform_bilinear(&frame, &id, 8, 8);
        // Bilinear with identity should match exactly (no fractional coords)
        assert_eq!(out, frame);
    }

    // 34. Bilinear transform output size
    #[test]
    fn test_bilinear_transform_output_size() {
        let frame = make_rgba(16, 12, 80, 80, 80, 255);
        let t = AffineTransform {
            dx: 1.5,
            dy: 0.5,
            da: 0.0,
            ds: 1.0,
        };
        let out = VideoStabilizer::apply_transform_bilinear(&frame, &t, 16, 12);
        assert_eq!(out.len(), 16 * 12 * 4);
    }

    // 35. Quality metric: max_correction is computed
    #[test]
    fn test_quality_metric_max_correction() {
        let frames = vec![
            StabilizationFrame {
                transform: AffineTransform {
                    dx: 10.0,
                    dy: 0.0,
                    da: 0.0,
                    ds: 1.0,
                },
                smoothed: AffineTransform {
                    dx: 5.0,
                    dy: 0.0,
                    da: 0.0,
                    ds: 1.0,
                },
            },
            StabilizationFrame {
                transform: AffineTransform {
                    dx: -10.0,
                    dy: 0.0,
                    da: 0.0,
                    ds: 1.0,
                },
                smoothed: AffineTransform {
                    dx: -3.0,
                    dy: 0.0,
                    da: 0.0,
                    ds: 1.0,
                },
            },
        ];
        let quality = compute_stabilization_quality(&frames);
        assert!(
            quality.max_correction > 4.0,
            "max_correction: {}",
            quality.max_correction
        );
    }

    // 36. Rolling shutter readout fraction clamped
    #[test]
    fn test_rolling_shutter_readout_clamp() {
        let t0 = AffineTransform {
            dx: 0.0,
            dy: 0.0,
            da: 0.0,
            ds: 1.0,
        };
        let t1 = AffineTransform {
            dx: 10.0,
            dy: 0.0,
            da: 0.0,
            ds: 1.0,
        };
        let est = estimate_rolling_shutter(&t0, &t1, 2.0); // over 1.0
        assert!(
            (est.readout_fraction - 1.0).abs() < 1e-6,
            "should be clamped to 1.0"
        );
    }

    // 37. KalmanState2 predict-update cycle
    #[test]
    fn test_kalman_state2_predict_update() {
        let mut kf = KalmanState2::new(0.001, 0.1);
        // Feed constant value, should converge
        for _ in 0..50 {
            kf.step(10.0);
        }
        assert!(
            (kf.x[0] - 10.0).abs() < 1.0,
            "KalmanState2 should converge to 10.0, got {}",
            kf.x[0]
        );
    }

    // 38. Crop-scale crop_rect bounds are valid
    #[test]
    fn test_crop_scale_rect_bounds() {
        let corrections = vec![AffineTransform {
            dx: 50.0,
            dy: 30.0,
            da: 0.05,
            ds: 1.0,
        }];
        let result = compute_crop_scale(&corrections, 1920, 1080);
        assert!(result.crop_rect[0] >= 0.0 && result.crop_rect[0] < 0.5);
        assert!(result.crop_rect[1] >= 0.0 && result.crop_rect[1] < 0.5);
        assert!(result.crop_rect[2] > 0.5 && result.crop_rect[2] <= 1.0);
        assert!(result.crop_rect[3] > 0.5 && result.crop_rect[3] <= 1.0);
    }
}
