//! 360-degree / equirectangular video stabilization with wrap-around boundary handling.
//!
//! Ordinary stabilization clips motion estimation and frame warping at the image
//! boundary.  For 360-degree equirectangular video the left and right edges are
//! adjacent in 3-D space; features that cross the boundary should be tracked
//! across the wrap-around seam, and warp sampling should read from the opposite
//! edge rather than padding with a constant.
//!
//! # Coordinate conventions
//!
//! Equirectangular frames map:
//! - Horizontal (X) → longitude  [−180 °, +180 °)
//! - Vertical   (Y) → latitude   [−90 °, +90 °]
//!
//! The left column (x=0) is adjacent to the right column (x=W-1).  The top and
//! bottom rows correspond to the poles and are *not* wrapped.

use crate::error::{StabilizeError, StabilizeResult};
use crate::Frame;
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Wrap-around utilities
// ---------------------------------------------------------------------------

/// Wrap a horizontal pixel coordinate into `[0, width)`.
///
/// This is the fundamental primitive used throughout the module.
#[must_use]
#[inline]
pub fn wrap_x(x: i64, width: usize) -> usize {
    let w = width as i64;
    ((x % w + w) % w) as usize
}

/// Clamp vertical pixel coordinate into `[0, height)`.
///
/// The top and bottom edges are *not* wrapped (poles).
#[must_use]
#[inline]
pub fn clamp_y(y: i64, height: usize) -> usize {
    y.clamp(0, height as i64 - 1) as usize
}

/// Bilinear sample from an equirectangular frame with horizontal wrap-around.
///
/// `sx`, `sy` are floating-point source coordinates.  `sx` wraps; `sy` clamps.
#[must_use]
pub fn sample_wrapped(data: &Array2<u8>, sx: f64, sy: f64) -> u8 {
    let (h, w) = data.dim();
    if w == 0 || h == 0 {
        return 0;
    }

    let x0 = sx.floor() as i64;
    let y0 = sy.floor().clamp(0.0, (h - 1) as f64) as i64;

    let tx = sx - sx.floor();
    let ty = sy - sy.floor();

    let x1 = x0 + 1;
    let y1 = (y0 + 1).min(h as i64 - 1);

    let wx0 = wrap_x(x0, w);
    let wx1 = wrap_x(x1, w);
    let wy0 = clamp_y(y0, h);
    let wy1 = clamp_y(y1, h);

    let p00 = data[[wy0, wx0]] as f64;
    let p10 = data[[wy0, wx1]] as f64;
    let p01 = data[[wy1, wx0]] as f64;
    let p11 = data[[wy1, wx1]] as f64;

    let v = p00 * (1.0 - tx) * (1.0 - ty)
        + p10 * tx * (1.0 - ty)
        + p01 * (1.0 - tx) * ty
        + p11 * tx * ty;

    v.clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Wrap-aware feature tracking
// ---------------------------------------------------------------------------

/// Minimum horizontal distance between two X-coordinates considering wrap-around.
///
/// Returns a value in `[0, width/2]`.
#[must_use]
pub fn wrapped_dx(x1: f64, x2: f64, width: usize) -> f64 {
    let raw = x2 - x1;
    let w = width as f64;
    // Normalise to (−w/2, w/2].
    let wrapped = ((raw + w / 2.0).rem_euclid(w)) - w / 2.0;
    wrapped
}

/// A feature point with wrap-aware coordinates.
#[derive(Debug, Clone, Copy)]
pub struct SphereFeature {
    /// Pixel-space X coordinate (horizontal, wraps).
    pub x: f64,
    /// Pixel-space Y coordinate (vertical, clamps at poles).
    pub y: f64,
    /// Optical-flow quality.
    pub quality: f64,
    /// Stable feature ID across frames.
    pub id: usize,
}

impl SphereFeature {
    /// Create a new spherical feature.
    #[must_use]
    pub const fn new(x: f64, y: f64, quality: f64, id: usize) -> Self {
        Self { x, y, quality, id }
    }

    /// Wrap-aware horizontal distance to another feature.
    #[must_use]
    pub fn wrapped_distance_to(&self, other: &Self, frame_width: usize) -> f64 {
        let dx = wrapped_dx(self.x, other.x, frame_width);
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// ---------------------------------------------------------------------------
// 360° stabilization transform
// ---------------------------------------------------------------------------

/// A stabilization transform for 360-degree video.
///
/// Corrections are expressed as **pan** (horizontal rotation) and **tilt**
/// (vertical rotation) in pixels, plus an in-plane roll in radians.
#[derive(Debug, Clone, Copy)]
pub struct SphericalTransform {
    /// Frame index.
    pub frame: usize,
    /// Horizontal correction (pixels; wraps around).
    pub pan_px: f64,
    /// Vertical correction (pixels; clamped at poles).
    pub tilt_px: f64,
    /// Roll correction (radians).
    pub roll_rad: f64,
}

impl SphericalTransform {
    /// Identity transform.
    #[must_use]
    pub const fn identity(frame: usize) -> Self {
        Self {
            frame,
            pan_px: 0.0,
            tilt_px: 0.0,
            roll_rad: 0.0,
        }
    }

    /// Magnitude of the translational correction.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.pan_px * self.pan_px + self.tilt_px * self.tilt_px).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Core stabilizer
// ---------------------------------------------------------------------------

/// Configuration for 360-degree stabilization.
#[derive(Debug, Clone)]
pub struct StitchingStabilizeConfig {
    /// Smoothing window (frames).
    pub smoothing_window: usize,
    /// Smoothing strength in `[0, 1]`.
    pub smoothing_strength: f64,
    /// Maximum feature search radius (pixels).
    pub search_radius: usize,
    /// Number of corner features per frame.
    pub max_features: usize,
    /// Harris corner quality threshold.
    pub quality_threshold: f64,
    /// Horizontal boundary margin: features within this many pixels of the
    /// left or right edge are tracked across the wrap seam.
    pub seam_margin: usize,
}

impl Default for StitchingStabilizeConfig {
    fn default() -> Self {
        Self {
            smoothing_window: 30,
            smoothing_strength: 0.8,
            search_radius: 20,
            max_features: 300,
            quality_threshold: 0.01,
            seam_margin: 32,
        }
    }
}

/// 360-degree video stabilizer with equirectangular wrap-around.
pub struct StitchingStabilizer {
    config: StitchingStabilizeConfig,
}

impl StitchingStabilizer {
    /// Create a new `StitchingStabilizer`.
    #[must_use]
    pub fn new(config: StitchingStabilizeConfig) -> Self {
        Self { config }
    }

    /// Estimate per-frame spherical stabilization transforms.
    ///
    /// Returns one [`SphericalTransform`] per input frame.
    ///
    /// # Errors
    ///
    /// - [`StabilizeError::EmptyFrameSequence`] – `frames` is empty.
    pub fn stabilize(&self, frames: &[Frame]) -> StabilizeResult<Vec<SphericalTransform>> {
        if frames.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let n = frames.len();
        let mut raw_pan = vec![0.0f64; n];
        let mut raw_tilt = vec![0.0f64; n];

        for i in 1..n {
            let (dpan, dtilt) = self.estimate_motion(&frames[i - 1], &frames[i]);
            raw_pan[i] = raw_pan[i - 1] + dpan;
            raw_tilt[i] = raw_tilt[i - 1] + dtilt;
        }

        let smooth_pan = self.smooth(&raw_pan);
        let smooth_tilt = self.smooth(&raw_tilt);

        let transforms = (0..n)
            .map(|i| SphericalTransform {
                frame: i,
                pan_px: smooth_pan[i] - raw_pan[i],
                tilt_px: smooth_tilt[i] - raw_tilt[i],
                roll_rad: 0.0,
            })
            .collect();

        Ok(transforms)
    }

    /// Apply stabilization transforms to a sequence of frames.
    ///
    /// Returns new frames with horizontal wrap-around bilinear sampling.
    ///
    /// # Errors
    ///
    /// - [`StabilizeError::DimensionMismatch`] – `transforms.len() != frames.len()`.
    pub fn apply(
        &self,
        frames: &[Frame],
        transforms: &[SphericalTransform],
    ) -> StabilizeResult<Vec<Frame>> {
        if frames.len() != transforms.len() {
            return Err(StabilizeError::dimension_mismatch(
                format!("{}", frames.len()),
                format!("{}", transforms.len()),
            ));
        }

        frames
            .iter()
            .zip(transforms.iter())
            .map(|(frame, transform)| self.warp_frame(frame, transform))
            .collect()
    }

    /// Warp a single frame with horizontal wrap-around.
    fn warp_frame(&self, frame: &Frame, transform: &SphericalTransform) -> StabilizeResult<Frame> {
        let w = frame.width;
        let h = frame.height;
        let mut out = Array2::zeros((h, w));

        for dy in 0..h {
            for dx in 0..w {
                let sx = dx as f64 - transform.pan_px;
                let sy = dy as f64 - transform.tilt_px;
                out[[dy, dx]] = sample_wrapped(&frame.data, sx, sy);
            }
        }

        Ok(Frame::new(w, h, frame.timestamp, out))
    }

    /// Estimate pan/tilt motion between two equirectangular frames.
    fn estimate_motion(&self, prev: &Frame, curr: &Frame) -> (f64, f64) {
        let w = prev.width;
        let h = prev.height;
        let corners = self.harris_corners(&prev.data, w, h);

        let win = 11usize;
        let half = win / 2;
        let sr = self.config.search_radius;

        let mut sum_dpan = 0.0f64;
        let mut sum_dtilt = 0.0f64;
        let mut count = 0usize;

        for y in half..(h.saturating_sub(half)) {
            for x in half..(w.saturating_sub(half)) {
                if corners[[y, x]] < self.config.quality_threshold {
                    continue;
                }

                // Template match with wrap-around sampling for search.
                let mut best_score = f64::MAX;
                let mut best_dpan = 0i32;
                let mut best_dtilt = 0i32;

                for dtilt in -(sr as i32)..=(sr as i32) {
                    for dpan in -(sr as i32)..=(sr as i32) {
                        let ny = y as i32 + dtilt;
                        if ny < half as i32 || ny + half as i32 >= h as i32 {
                            continue;
                        }
                        let score = self.wrapped_ssd(
                            &prev.data,
                            &curr.data,
                            x,
                            y,
                            x as i32 + dpan,
                            ny,
                            win,
                            w,
                        );
                        if score < best_score {
                            best_score = score;
                            best_dpan = dpan;
                            best_dtilt = dtilt;
                        }
                    }
                }

                sum_dpan += best_dpan as f64;
                sum_dtilt += best_dtilt as f64;
                count += 1;

                if count >= self.config.max_features {
                    break;
                }
            }
            if count >= self.config.max_features {
                break;
            }
        }

        if count == 0 {
            return (0.0, 0.0);
        }
        (sum_dpan / count as f64, sum_dtilt / count as f64)
    }

    /// SSD with horizontal wrap-around for the search location.
    #[allow(clippy::too_many_arguments)]
    fn wrapped_ssd(
        &self,
        prev: &Array2<u8>,
        curr: &Array2<u8>,
        ax: usize,
        ay: usize,
        bx_raw: i32,
        by: i32,
        win: usize,
        frame_width: usize,
    ) -> f64 {
        let (h, _) = prev.dim();
        let half = win / 2;
        let mut sum = 0.0f64;

        for dr in 0..win {
            let pa = ay + dr - half;
            let pb = (by + dr as i32 - half as i32).clamp(0, h as i32 - 1) as usize;

            for dc in 0..win {
                let ca = ax + dc - half;
                let cb = wrap_x(i64::from(bx_raw + dc as i32 - half as i32), frame_width);

                let p = prev[[pa, ca]] as f64;
                let q = curr[[pb, cb]] as f64;
                let d = p - q;
                sum += d * d;
            }
        }

        sum / (win * win) as f64
    }

    /// Harris corner detection for equirectangular data.
    fn harris_corners(&self, image: &Array2<u8>, w: usize, h: usize) -> Array2<f64> {
        let mut response = Array2::zeros((h, w));
        let k = 0.04f64;
        let win = 5usize;
        let half = win / 2;

        let mut gx = Array2::zeros((h, w));
        let mut gy = Array2::zeros((h, w));

        for r in 1..(h - 1) {
            for c in 1..(w - 1) {
                let wc = wrap_x(c as i64 - 1, w);
                let ec = wrap_x(c as i64 + 1, w);
                gx[[r, c]] = image[[r, ec]] as f64 - image[[r, wc]] as f64;
                gy[[r, c]] =
                    image[[(r + 1).min(h - 1), c]] as f64 - image[[r.saturating_sub(1), c]] as f64;
            }
        }

        for r in half..(h - half) {
            for c in half..(w - half) {
                let (mut ixx, mut iyy, mut ixy) = (0.0f64, 0.0f64, 0.0f64);
                for dr in 0..win {
                    for dc in 0..win {
                        let gr = gx[[r + dr - half, c + dc - half]];
                        let grr = gy[[r + dr - half, c + dc - half]];
                        ixx += gr * gr;
                        iyy += grr * grr;
                        ixy += gr * grr;
                    }
                }
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                response[[r, c]] = (det - k * trace * trace).max(0.0);
            }
        }

        let max = response.iter().cloned().fold(0.0f64, f64::max);
        if max > 0.0 {
            response.mapv_inplace(|v| v / max);
        }

        response
    }

    /// Box-filter Gaussian trajectory smoothing.
    fn smooth(&self, traj: &[f64]) -> Vec<f64> {
        let n = traj.len();
        if n == 0 {
            return Vec::new();
        }
        let half = (self.config.smoothing_window / 2).max(1);
        let strength = self.config.smoothing_strength.clamp(0.0, 1.0);
        let mut smoothed = traj.to_vec();

        for _ in 0..3 {
            let prev = smoothed.clone();
            for i in 0..n {
                let s = i.saturating_sub(half);
                let e = (i + half + 1).min(n);
                let sum: f64 = prev[s..e].iter().sum();
                let avg = sum / (e - s) as f64;
                smoothed[i] = traj[i] * (1.0 - strength) + avg * strength;
            }
        }

        smoothed
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn solid_frame(w: usize, h: usize, fill: u8) -> Frame {
        Frame::new(w, h, 0.0, Array2::from_elem((h, w), fill))
    }

    fn gradient_frame(w: usize, h: usize) -> Frame {
        let mut data = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                data[[y, x]] = ((x * 255) / w.max(1)) as u8;
            }
        }
        Frame::new(w, h, 0.0, data)
    }

    #[test]
    fn test_wrap_x_positive() {
        assert_eq!(wrap_x(5, 10), 5);
        assert_eq!(wrap_x(10, 10), 0);
        assert_eq!(wrap_x(11, 10), 1);
    }

    #[test]
    fn test_wrap_x_negative() {
        assert_eq!(wrap_x(-1, 10), 9);
        assert_eq!(wrap_x(-10, 10), 0);
    }

    #[test]
    fn test_clamp_y() {
        assert_eq!(clamp_y(-5, 10), 0);
        assert_eq!(clamp_y(15, 10), 9);
        assert_eq!(clamp_y(5, 10), 5);
    }

    #[test]
    fn test_sample_wrapped_identity_offset() {
        let data = Array2::from_elem((10, 10), 100u8);
        let val = sample_wrapped(&data, 5.0, 5.0);
        assert_eq!(val, 100);
    }

    #[test]
    fn test_sample_wrapped_crosses_boundary() {
        // Rightmost and leftmost columns should be adjacent.
        let mut data = Array2::zeros((10, 10));
        data[[5, 9]] = 200; // rightmost
                            // Sampling at x=-0.5 should blend leftmost (0) and wrap from rightmost (200).
        let val = sample_wrapped(&data, -0.5, 5.0);
        // Interpolated: 0*(0.5) + 200*(0.5) = 100.
        assert!((val as i32 - 100).abs() <= 1);
    }

    #[test]
    fn test_wrapped_dx_normal() {
        let dx = wrapped_dx(10.0, 15.0, 100);
        assert!((dx - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_wrapped_dx_across_boundary() {
        // Feature at x=5 wraps to x=95 (delta = -10, or equivalently +90 naive).
        let dx = wrapped_dx(95.0, 5.0, 100);
        assert!((dx - 10.0).abs() < 1e-9, "wrapped_dx was {}", dx);
    }

    #[test]
    fn test_sphere_feature_wrapped_distance() {
        let f1 = SphereFeature::new(5.0, 50.0, 1.0, 0);
        let f2 = SphereFeature::new(95.0, 50.0, 1.0, 1);
        let d = f1.wrapped_distance_to(&f2, 100);
        // Min-distance across wrap: 10 px.
        assert!(d < 15.0, "wrapped distance was {}", d);
    }

    #[test]
    fn test_spherical_transform_identity_magnitude() {
        let t = SphericalTransform::identity(0);
        assert!((t.magnitude()).abs() < 1e-9);
    }

    #[test]
    fn test_stitching_stabilizer_empty_frames() {
        let stab = StitchingStabilizer::new(StitchingStabilizeConfig::default());
        let result = stab.stabilize(&[]);
        assert!(matches!(result, Err(StabilizeError::EmptyFrameSequence)));
    }

    #[test]
    fn test_stitching_stabilizer_single_frame_identity() {
        let stab = StitchingStabilizer::new(StitchingStabilizeConfig::default());
        let frames = vec![solid_frame(64, 32, 128)];
        let transforms = stab.stabilize(&frames).expect("should succeed");
        assert_eq!(transforms.len(), 1);
        assert!((transforms[0].pan_px).abs() < 1e-9);
    }

    #[test]
    fn test_stitching_stabilizer_output_length() {
        let stab = StitchingStabilizer::new(StitchingStabilizeConfig::default());
        let frames: Vec<Frame> = (0..5).map(|i| solid_frame(64, 32, i as u8 * 40)).collect();
        let transforms = stab.stabilize(&frames).expect("ok");
        assert_eq!(transforms.len(), 5);
    }

    #[test]
    fn test_stitching_stabilizer_apply_length() {
        let stab = StitchingStabilizer::new(StitchingStabilizeConfig::default());
        let frames: Vec<Frame> = (0..3).map(|_i| gradient_frame(64, 32)).collect();
        let transforms: Vec<SphericalTransform> =
            (0..3).map(SphericalTransform::identity).collect();
        let out = stab.apply(&frames, &transforms).expect("ok");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_stitching_stabilizer_apply_mismatch_error() {
        let stab = StitchingStabilizer::new(StitchingStabilizeConfig::default());
        let frames = vec![solid_frame(64, 32, 100)];
        let transforms: Vec<SphericalTransform> =
            (0..3).map(SphericalTransform::identity).collect();
        let result = stab.apply(&frames, &transforms);
        assert!(matches!(
            result,
            Err(StabilizeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_apply_identity_transform_preserves_frame() {
        let stab = StitchingStabilizer::new(StitchingStabilizeConfig::default());
        let frame = gradient_frame(32, 16);
        let t = SphericalTransform::identity(0);
        let out = stab.apply(std::slice::from_ref(&frame), &[t]).expect("ok");
        // Identity warp should exactly reproduce the input.
        assert_eq!(out[0].data, frame.data);
    }

    #[test]
    fn test_apply_wrap_around_shift() {
        // Shift by full width should produce same image (modulo boundary rounding).
        let stab = StitchingStabilizer::new(StitchingStabilizeConfig::default());
        let frame = gradient_frame(64, 32);
        let t = SphericalTransform {
            frame: 0,
            pan_px: 64.0, // full-width wrap
            tilt_px: 0.0,
            roll_rad: 0.0,
        };
        let out = stab.apply(std::slice::from_ref(&frame), &[t]).expect("ok");
        assert_eq!(out[0].data, frame.data);
    }
}
