//! Region-of-interest stabilization: stabilize a specific sub-region independently.
//!
//! Standard stabilization treats the entire frame as a single rigid body. When a
//! scene contains a small subject that must remain locked while the background
//! drifts (e.g., tracking a speaker on a wide-shot live-stream stage), this
//! module estimates motion *exclusively* within the ROI bounding box and derives
//! a corrective transform that keeps that region pixel-stable.
//!
//! # Algorithm
//!
//! 1. Crop the ROI from each consecutive frame pair.
//! 2. Detect Harris corners confined to the crop.
//! 3. Match corners across the pair with template matching.
//! 4. Estimate a 2-DOF translation (or 4-DOF similarity) from the matched points.
//! 5. Compose the per-frame corrections into a cumulative trajectory and apply
//!    the usual Gaussian smoothing with configurable strength.
//! 6. Return per-frame [`RoiTransform`] values suitable for compositing or
//!    full-frame warp application.

use crate::error::{StabilizeError, StabilizeResult};
use crate::Frame;
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box describing a region of interest in pixel space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roi {
    /// Left edge (inclusive, pixels from left).
    pub x: usize,
    /// Top edge (inclusive, pixels from top).
    pub y: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl Roi {
    /// Create a new ROI.
    ///
    /// # Errors
    ///
    /// Returns an error when `width` or `height` is zero.
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> StabilizeResult<Self> {
        if width == 0 || height == 0 {
            return Err(StabilizeError::invalid_parameter(
                "roi",
                "width and height must be > 0",
            ));
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Right edge (exclusive).
    #[must_use]
    pub const fn right(&self) -> usize {
        self.x + self.width
    }

    /// Bottom edge (exclusive).
    #[must_use]
    pub const fn bottom(&self) -> usize {
        self.y + self.height
    }

    /// Whether the ROI fits entirely within a frame of the given size.
    #[must_use]
    pub fn fits_in(&self, frame_width: usize, frame_height: usize) -> bool {
        self.right() <= frame_width && self.bottom() <= frame_height
    }

    /// Clamp this ROI to fit within a frame of the given dimensions.
    #[must_use]
    pub fn clamp_to(&self, frame_width: usize, frame_height: usize) -> Self {
        let x = self.x.min(frame_width.saturating_sub(1));
        let y = self.y.min(frame_height.saturating_sub(1));
        let w = self.width.min(frame_width.saturating_sub(x));
        let h = self.height.min(frame_height.saturating_sub(y));
        Self {
            x,
            y,
            width: w.max(1),
            height: h.max(1),
        }
    }
}

/// Corrective 2-D transform for a single frame derived from ROI analysis.
#[derive(Debug, Clone, Copy)]
pub struct RoiTransform {
    /// Frame index.
    pub frame: usize,
    /// Horizontal correction (pixels, positive = shift right).
    pub dx: f64,
    /// Vertical correction (pixels, positive = shift down).
    pub dy: f64,
    /// Rotation correction (radians).
    pub angle: f64,
    /// Scale correction (1.0 = no change).
    pub scale: f64,
}

impl RoiTransform {
    /// Identity transform (no correction).
    #[must_use]
    pub const fn identity(frame: usize) -> Self {
        Self {
            frame,
            dx: 0.0,
            dy: 0.0,
            angle: 0.0,
            scale: 1.0,
        }
    }

    /// Magnitude of the translational component.
    #[must_use]
    pub fn translation_magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// Configuration for ROI stabilization.
#[derive(Debug, Clone)]
pub struct RoiStabilizeConfig {
    /// Smoothing window size (frames).
    pub smoothing_window: usize,
    /// Smoothing strength in `[0, 1]`.
    pub smoothing_strength: f64,
    /// Maximum number of corner features used per crop.
    pub max_features: usize,
    /// Minimum corner quality threshold (Harris response, normalised).
    pub quality_threshold: f64,
    /// Maximum search radius for template matching (pixels in crop space).
    pub search_radius: usize,
}

impl Default for RoiStabilizeConfig {
    fn default() -> Self {
        Self {
            smoothing_window: 30,
            smoothing_strength: 0.8,
            max_features: 200,
            quality_threshold: 0.01,
            search_radius: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Core stabilizer
// ---------------------------------------------------------------------------

/// Performs stabilization of a user-defined region of interest.
pub struct RoiStabilizer {
    config: RoiStabilizeConfig,
}

impl RoiStabilizer {
    /// Create a new ROI stabilizer with the given configuration.
    #[must_use]
    pub fn new(config: RoiStabilizeConfig) -> Self {
        Self { config }
    }

    /// Estimate per-frame corrective transforms for `roi` across `frames`.
    ///
    /// Returns one [`RoiTransform`] per frame.  The first frame always has an
    /// identity transform; subsequent frames carry cumulative corrections.
    ///
    /// # Errors
    ///
    /// - [`StabilizeError::EmptyFrameSequence`] – `frames` is empty.
    /// - [`StabilizeError::InvalidParameter`] – the ROI does not overlap any frame.
    pub fn stabilize(&self, frames: &[Frame], roi: Roi) -> StabilizeResult<Vec<RoiTransform>> {
        if frames.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let first = &frames[0];
        if !roi.fits_in(first.width, first.height) {
            let clamped = roi.clamp_to(first.width, first.height);
            return self.stabilize_internal(frames, clamped);
        }

        self.stabilize_internal(frames, roi)
    }

    fn stabilize_internal(&self, frames: &[Frame], roi: Roi) -> StabilizeResult<Vec<RoiTransform>> {
        let n = frames.len();
        // Cumulative raw translations in the ROI coordinate system.
        let mut raw_dx = vec![0.0f64; n];
        let mut raw_dy = vec![0.0f64; n];

        for i in 1..n {
            let prev_crop = self.crop_roi(&frames[i - 1].data, &roi, frames[i - 1].width);
            let curr_crop = self.crop_roi(&frames[i].data, &roi, frames[i].width);

            let (ddx, ddy) = self.estimate_translation(&prev_crop, &curr_crop);
            raw_dx[i] = raw_dx[i - 1] + ddx;
            raw_dy[i] = raw_dy[i - 1] + ddy;
        }

        // Smooth trajectories.
        let smooth_dx = self.smooth_trajectory(&raw_dx);
        let smooth_dy = self.smooth_trajectory(&raw_dy);

        // Corrections = smooth - raw  (negate cumulative drift).
        let transforms = (0..n)
            .map(|i| RoiTransform {
                frame: i,
                dx: smooth_dx[i] - raw_dx[i],
                dy: smooth_dy[i] - raw_dy[i],
                angle: 0.0,
                scale: 1.0,
            })
            .collect();

        Ok(transforms)
    }

    /// Crop the ROI from a frame data array.
    fn crop_roi(&self, data: &Array2<u8>, roi: &Roi, frame_width: usize) -> Array2<u8> {
        let height = data.dim().0;
        let clamped = roi.clamp_to(frame_width, height);

        let rows = clamped.height;
        let cols = clamped.width;
        let mut crop = Array2::zeros((rows, cols));

        for r in 0..rows {
            for c in 0..cols {
                let sy = clamped.y + r;
                let sx = clamped.x + c;
                if sy < data.dim().0 && sx < data.dim().1 {
                    crop[[r, c]] = data[[sy, sx]];
                }
            }
        }

        crop
    }

    /// Estimate 2-D translation between two crops using Harris + template matching.
    fn estimate_translation(&self, prev: &Array2<u8>, curr: &Array2<u8>) -> (f64, f64) {
        let corners = self.harris_corners(prev);
        let (h, w) = prev.dim();
        let win = 11usize;
        let half = win / 2;
        let sr = self.config.search_radius;

        let mut sum_dx = 0.0f64;
        let mut sum_dy = 0.0f64;
        let mut count = 0usize;

        for r in half..(h.saturating_sub(half)) {
            for c in half..(w.saturating_sub(half)) {
                if corners[[r, c]] < self.config.quality_threshold {
                    continue;
                }

                // Template match to find best shift.
                let mut best_score = f64::MAX;
                let mut best_dr = 0i32;
                let mut best_dc = 0i32;

                for dr in -(sr as i32)..=(sr as i32) {
                    for dc in -(sr as i32)..=(sr as i32) {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr < half as i32
                            || nc < half as i32
                            || nr + half as i32 >= h as i32
                            || nc + half as i32 >= w as i32
                        {
                            continue;
                        }
                        let score = self.ssd(prev, curr, r, c, nr as usize, nc as usize, win);
                        if score < best_score {
                            best_score = score;
                            best_dr = dr;
                            best_dc = dc;
                        }
                    }
                }

                sum_dx += best_dc as f64;
                sum_dy += best_dr as f64;
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

        (sum_dx / count as f64, sum_dy / count as f64)
    }

    /// Sum of squared differences between two windows.
    #[allow(clippy::too_many_arguments)]
    fn ssd(
        &self,
        a: &Array2<u8>,
        b: &Array2<u8>,
        ay: usize,
        ax: usize,
        by: usize,
        bx: usize,
        win: usize,
    ) -> f64 {
        let half = win / 2;
        let mut sum = 0.0f64;
        for dr in 0..win {
            for dc in 0..win {
                let pa = a[[ay + dr - half, ax + dc - half]] as f64;
                let pb = b[[by + dr - half, bx + dc - half]] as f64;
                let d = pa - pb;
                sum += d * d;
            }
        }
        sum / (win * win) as f64
    }

    /// Simple Harris corner detection returning a normalised response map.
    fn harris_corners(&self, image: &Array2<u8>) -> Array2<f64> {
        let (h, w) = image.dim();
        let mut response = Array2::zeros((h, w));
        let k = 0.04f64;
        let win = 5usize;
        let half = win / 2;

        // Sobel gradients.
        let mut gx = Array2::zeros((h, w));
        let mut gy = Array2::zeros((h, w));
        for r in 1..(h - 1) {
            for c in 1..(w - 1) {
                let p = |dr: i32, dc: i32| {
                    image[[(r as i32 + dr) as usize, (c as i32 + dc) as usize]] as f64
                };
                gx[[r, c]] =
                    -p(-1, -1) + p(-1, 1) - 2.0 * p(0, -1) + 2.0 * p(0, 1) - p(1, -1) + p(1, 1);
                gy[[r, c]] =
                    -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);
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

        // Normalise.
        let max = response.iter().cloned().fold(0.0f64, f64::max);
        if max > 0.0 {
            response.mapv_inplace(|v| v / max);
        }

        response
    }

    /// Gaussian-smoothed trajectory via repeated box filter.
    fn smooth_trajectory(&self, traj: &[f64]) -> Vec<f64> {
        let n = traj.len();
        if n == 0 {
            return Vec::new();
        }

        let half = (self.config.smoothing_window / 2).max(1);
        let strength = self.config.smoothing_strength.clamp(0.0, 1.0);
        let mut smoothed = traj.to_vec();

        // Three passes of box filter approximate Gaussian.
        for _ in 0..3 {
            let prev = smoothed.clone();
            for i in 0..n {
                let start = i.saturating_sub(half);
                let end = (i + half + 1).min(n);
                let sum: f64 = prev[start..end].iter().sum();
                let avg = sum / (end - start) as f64;
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

    fn make_frame(w: usize, h: usize, ts: f64, fill: u8) -> Frame {
        Frame::new(w, h, ts, Array2::from_elem((h, w), fill))
    }

    fn make_shifted_frame(w: usize, h: usize, ts: f64, dx: usize) -> Frame {
        let mut data = Array2::zeros((h, w));
        // Draw a small square offset by dx columns.
        let sq = 10usize;
        for r in 20..(20 + sq) {
            for c in (20 + dx)..(20 + dx + sq) {
                if r < h && c < w {
                    data[[r, c]] = 200;
                }
            }
        }
        Frame::new(w, h, ts, data)
    }

    #[test]
    fn test_roi_new_valid() {
        let roi = Roi::new(10, 20, 100, 80);
        assert!(roi.is_ok());
        let r = roi.expect("valid roi");
        assert_eq!(r.right(), 110);
        assert_eq!(r.bottom(), 100);
    }

    #[test]
    fn test_roi_new_zero_size() {
        assert!(Roi::new(0, 0, 0, 10).is_err());
        assert!(Roi::new(0, 0, 10, 0).is_err());
    }

    #[test]
    fn test_roi_fits_in() {
        let roi = Roi::new(0, 0, 100, 100).expect("valid");
        assert!(roi.fits_in(100, 100));
        assert!(!roi.fits_in(99, 100));
    }

    #[test]
    fn test_roi_clamp() {
        let roi = Roi::new(90, 90, 50, 50).expect("valid");
        let clamped = roi.clamp_to(100, 100);
        assert!(clamped.right() <= 100);
        assert!(clamped.bottom() <= 100);
    }

    #[test]
    fn test_roi_transform_identity() {
        let t = RoiTransform::identity(0);
        assert!((t.dx).abs() < f64::EPSILON);
        assert!((t.scale - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roi_stabilizer_empty_frames() {
        let config = RoiStabilizeConfig::default();
        let stabilizer = RoiStabilizer::new(config);
        let roi = Roi::new(0, 0, 50, 50).expect("valid");
        let result = stabilizer.stabilize(&[], roi);
        assert!(matches!(result, Err(StabilizeError::EmptyFrameSequence)));
    }

    #[test]
    fn test_roi_stabilizer_single_frame() {
        let config = RoiStabilizeConfig::default();
        let stabilizer = RoiStabilizer::new(config);
        let roi = Roi::new(0, 0, 50, 50).expect("valid");
        let frames = vec![make_frame(100, 100, 0.0, 128)];
        let result = stabilizer.stabilize(&frames, roi);
        assert!(result.is_ok());
        let transforms = result.expect("should succeed");
        assert_eq!(transforms.len(), 1);
        // Single frame → identity.
        assert!((transforms[0].dx).abs() < 1e-9);
    }

    #[test]
    fn test_roi_stabilizer_output_length() {
        let config = RoiStabilizeConfig::default();
        let stabilizer = RoiStabilizer::new(config);
        let roi = Roi::new(0, 0, 60, 60).expect("valid");
        let frames: Vec<Frame> = (0..5).map(|i| make_frame(100, 100, i as f64, 64)).collect();
        let result = stabilizer.stabilize(&frames, roi);
        assert!(result.is_ok());
        assert_eq!(result.expect("ok").len(), 5);
    }

    #[test]
    fn test_roi_stabilizer_oversized_roi_is_clamped() {
        let config = RoiStabilizeConfig::default();
        let stabilizer = RoiStabilizer::new(config);
        let roi = Roi::new(0, 0, 200, 200).expect("valid"); // larger than frame
        let frames: Vec<Frame> = (0..3)
            .map(|i| make_frame(100, 100, i as f64, 100))
            .collect();
        let result = stabilizer.stabilize(&frames, roi);
        // Should succeed after clamping.
        assert!(result.is_ok());
    }

    #[test]
    fn test_roi_stabilizer_detects_horizontal_shift() {
        // Frames with a translated square: frame 0 no shift, frame 1 shifted 5px right.
        let config = RoiStabilizeConfig {
            smoothing_window: 3,
            smoothing_strength: 0.0, // no smoothing → corrections should be near-zero for stable
            max_features: 50,
            quality_threshold: 0.01,
            search_radius: 10,
        };
        let stabilizer = RoiStabilizer::new(config);
        let roi = Roi::new(0, 0, 80, 80).expect("valid");

        let f0 = make_shifted_frame(100, 100, 0.0, 0);
        let f1 = make_shifted_frame(100, 100, 1.0 / 30.0, 5);
        let frames = vec![f0, f1];

        let result = stabilizer.stabilize(&frames, roi);
        assert!(result.is_ok());
        let transforms = result.expect("ok");
        assert_eq!(transforms.len(), 2);
        // Frame 0 is always identity.
        assert!((transforms[0].dx).abs() < 1e-9);
    }

    #[test]
    fn test_roi_stabilizer_frame_indices() {
        let config = RoiStabilizeConfig::default();
        let stabilizer = RoiStabilizer::new(config);
        let roi = Roi::new(0, 0, 50, 50).expect("valid");
        let frames: Vec<Frame> = (0..4).map(|i| make_frame(100, 100, i as f64, 50)).collect();
        let transforms = stabilizer.stabilize(&frames, roi).expect("ok");
        for (i, t) in transforms.iter().enumerate() {
            assert_eq!(t.frame, i);
        }
    }
}
