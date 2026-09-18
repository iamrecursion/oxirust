#![allow(dead_code)]
//! Lens distortion correction for pre-processing frames before stabilization.
//!
//! Many wide-angle and action camera lenses introduce significant barrel or
//! pincushion distortion that must be corrected before video stabilization,
//! otherwise the distortion artifacts are amplified by the stabilization warp.
//!
//! This module implements the **Brown-Conrady** radial + tangential lens model,
//! which is the industry-standard model used in OpenCV, camera calibration
//! tools, and DNG profiles.  Correction is performed by mapping every output
//! pixel back to its undistorted source coordinate (inverse mapping), which
//! avoids sampling holes.
//!
//! # Distortion Model
//!
//! The Brown-Conrady model expresses the distorted coordinates as:
//!
//! ```text
//! r² = x'² + y'²
//! x_d = x'(1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x'·y' + p2·(r² + 2·x'²)
//! y_d = y'(1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r² + 2·y'²) + 2·p2·x'·y'
//! ```
//!
//! where `(x', y')` are the normalized (undistorted) camera coordinates,
//! `k1..k3` are radial distortion coefficients, and `p1, p2` are tangential.
//!
//! Inversion (undistortion) is computed iteratively via Newton's method.
//!
//! # Features
//!
//! - Full Brown-Conrady model (k1, k2, k3, p1, p2)
//! - Fisheye equidistant model (θ → r)
//! - Camera intrinsic matrix (fx, fy, cx, cy)
//! - Undistortion look-up table (LUT) pre-computation for fast per-pixel warping
//! - Distortion coefficient estimation from a known grid (synthetic calibration)
//! - Optimal new camera matrix cropping to minimize black borders

use crate::error::{StabilizeError, StabilizeResult};

/// Intrinsic camera matrix parameters.
///
/// Represents the pinhole camera model:
///
/// ```text
/// K = [ fx   0  cx ]
///     [  0  fy  cy ]
///     [  0   0   1 ]
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in X direction (pixels).
    pub fx: f64,
    /// Focal length in Y direction (pixels).
    pub fy: f64,
    /// Principal point X (optical axis, pixels from left).
    pub cx: f64,
    /// Principal point Y (optical axis, pixels from top).
    pub cy: f64,
}

impl CameraIntrinsics {
    /// Create intrinsic parameters.
    #[must_use]
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Create symmetric intrinsics (fx == fy) centered at (w/2, h/2).
    #[must_use]
    pub fn symmetric(f: f64, width: u32, height: u32) -> Self {
        Self {
            fx: f,
            fy: f,
            cx: width as f64 * 0.5,
            cy: height as f64 * 0.5,
        }
    }

    /// Project a 3D point (X, Y, Z) to pixel coordinates.
    #[must_use]
    pub fn project(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        let xn = x / z;
        let yn = y / z;
        (self.fx * xn + self.cx, self.fy * yn + self.cy)
    }

    /// Back-project a pixel to normalized camera coordinates.
    #[must_use]
    pub fn unproject(&self, px: f64, py: f64) -> (f64, f64) {
        ((px - self.cx) / self.fx, (py - self.cy) / self.fy)
    }

    /// Validate that the intrinsics are well-formed.
    ///
    /// # Errors
    ///
    /// Returns [`StabilizeError::InvalidParameter`] if any focal length is
    /// non-positive or principal point is negative.
    pub fn validate(&self) -> StabilizeResult<()> {
        if self.fx <= 0.0 {
            return Err(StabilizeError::invalid_parameter(
                "fx",
                format!("{}", self.fx),
            ));
        }
        if self.fy <= 0.0 {
            return Err(StabilizeError::invalid_parameter(
                "fy",
                format!("{}", self.fy),
            ));
        }
        if self.cx < 0.0 {
            return Err(StabilizeError::invalid_parameter(
                "cx",
                format!("{}", self.cx),
            ));
        }
        if self.cy < 0.0 {
            return Err(StabilizeError::invalid_parameter(
                "cy",
                format!("{}", self.cy),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────
//  Distortion model types
// ─────────────────────────────────────────────────────────────────

/// Lens distortion model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionModel {
    /// Brown-Conrady radial + tangential model (most common).
    BrownConrady,
    /// Fisheye equidistant (θ-based) model for very wide FOV lenses.
    FisheyeEquidistant,
    /// Fisheye equisolid (area-preserving) model.
    FisheyeEquisolid,
    /// Pure radial (k1/k2/k3 only, no tangential).
    RadialOnly,
}

impl Default for DistortionModel {
    fn default() -> Self {
        Self::BrownConrady
    }
}

/// Distortion coefficients for the Brown-Conrady model.
///
/// All coefficients default to 0 (no distortion).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistortionCoeffs {
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Radial distortion coefficient k3.
    pub k3: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
}

impl Default for DistortionCoeffs {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

impl DistortionCoeffs {
    /// Create coefficients with only k1 (simple radial barrel/pincushion).
    #[must_use]
    pub const fn k1_only(k1: f64) -> Self {
        Self {
            k1,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Create full Brown-Conrady coefficients.
    #[must_use]
    pub const fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self { k1, k2, k3, p1, p2 }
    }

    /// True if all coefficients are zero (no distortion).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.k1.abs() < 1e-12
            && self.k2.abs() < 1e-12
            && self.k3.abs() < 1e-12
            && self.p1.abs() < 1e-12
            && self.p2.abs() < 1e-12
    }

    /// Classify the dominant distortion type.
    #[must_use]
    pub fn distortion_type(&self) -> &'static str {
        if self.k1 > 0.01 {
            "pincushion"
        } else if self.k1 < -0.01 {
            "barrel"
        } else {
            "minimal"
        }
    }

    /// Apply the Brown-Conrady distortion to normalized coordinates `(x, y)`.
    ///
    /// Returns the distorted normalized coordinates.
    #[must_use]
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (xd, yd)
    }

    /// Invert the Brown-Conrady distortion using iterative Newton's method.
    ///
    /// Given a *distorted* normalized point `(xd, yd)`, find the undistorted
    /// normalized coordinates.
    ///
    /// # Parameters
    ///
    /// * `xd`, `yd` — distorted normalized coordinates
    /// * `max_iter` — maximum Newton iterations (20 is typically sufficient)
    /// * `tol` — convergence tolerance in normalized pixels
    ///
    /// # Returns
    ///
    /// Undistorted normalized coordinates, or the best estimate after
    /// `max_iter` iterations if the method does not converge.
    #[must_use]
    pub fn undistort_normalized(&self, xd: f64, yd: f64, max_iter: usize, tol: f64) -> (f64, f64) {
        if self.is_identity() {
            return (xd, yd);
        }
        // Initial guess: distorted point
        let mut x = xd;
        let mut y = yd;

        for _ in 0..max_iter {
            let (xd_est, yd_est) = self.distort(x, y);
            let ex = xd_est - xd;
            let ey = yd_est - yd;
            if ex * ex + ey * ey < tol * tol {
                break;
            }
            // Jacobian of distort() w.r.t. (x, y) (approximate by finite diff)
            let h = 1e-6;
            let (xd1, _) = self.distort(x + h, y);
            let (_, yd1) = self.distort(x, y + h);
            let j00 = (xd1 - xd_est) / h;
            let j11 = (yd1 - yd_est) / h;
            // Diagonal Newton step (ignores off-diagonal terms for speed)
            let denom_x = if j00.abs() > 1e-12 { j00 } else { 1.0 };
            let denom_y = if j11.abs() > 1e-12 { j11 } else { 1.0 };
            x -= ex / denom_x;
            y -= ey / denom_y;
        }

        (x, y)
    }
}

// ─────────────────────────────────────────────────────────────────
//  Fisheye model helpers
// ─────────────────────────────────────────────────────────────────

/// Fisheye equidistant distortion: maps incidence angle θ to radius r = f·θ.
///
/// Used for very wide-angle lenses (GoPro, Sigma 8mm fisheye, etc.)
#[derive(Debug, Clone, Copy)]
pub struct FisheyeEquidistant {
    /// Focal length (pixels); the same f used in `r = f·θ`.
    pub focal: f64,
}

impl FisheyeEquidistant {
    /// Create a fisheye equidistant model.
    #[must_use]
    pub const fn new(focal: f64) -> Self {
        Self { focal }
    }

    /// Distort a rectilinear normalized point to fisheye normalized coords.
    #[must_use]
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r = (x * x + y * y).sqrt();
        if r < 1e-12 {
            return (0.0, 0.0);
        }
        let theta = r.atan(); // angle of incidence
        let r_d = self.focal * theta;
        (x / r * r_d, y / r * r_d)
    }

    /// Undistort a fisheye normalized point back to rectilinear coords.
    #[must_use]
    pub fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        let r_d = (x * x + y * y).sqrt();
        if r_d < 1e-12 {
            return (0.0, 0.0);
        }
        let theta = r_d / self.focal; // r = f·θ
        let r = theta.tan(); // tan(θ) gives rectilinear r
        let scale = r / r_d;
        (x * scale, y * scale)
    }
}

/// Fisheye equisolid distortion: maps incidence angle θ to `r = 2f·sin(θ/2)`.
#[derive(Debug, Clone, Copy)]
pub struct FisheyeEquisolid {
    /// Focal length (pixels).
    pub focal: f64,
}

impl FisheyeEquisolid {
    /// Create a fisheye equisolid model.
    #[must_use]
    pub const fn new(focal: f64) -> Self {
        Self { focal }
    }

    /// Distort a rectilinear normalized point.
    #[must_use]
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r = (x * x + y * y).sqrt();
        if r < 1e-12 {
            return (0.0, 0.0);
        }
        let theta = r.atan();
        let r_d = 2.0 * self.focal * (theta * 0.5).sin();
        (x / r * r_d, y / r * r_d)
    }

    /// Undistort a fisheye equisolid normalized point.
    #[must_use]
    pub fn undistort(&self, x: f64, y: f64) -> (f64, f64) {
        let r_d = (x * x + y * y).sqrt();
        if r_d < 1e-12 {
            return (0.0, 0.0);
        }
        let theta = 2.0
            * (r_d / (2.0 * self.focal))
                .asin()
                .min(std::f64::consts::FRAC_PI_2);
        let r = theta.tan();
        let scale = if r_d > 1e-12 { r / r_d } else { 1.0 };
        (x * scale, y * scale)
    }
}

// ─────────────────────────────────────────────────────────────────
//  Pre-computed undistortion LUT
// ─────────────────────────────────────────────────────────────────

/// A 2-channel (dx, dy) floating-point look-up table for fast per-pixel
/// lens undistortion.
///
/// The LUT stores, for each output pixel `(px, py)`, the fractional source
/// coordinates `(src_x, src_y)` to sample from the distorted input image.
/// This allows bilinear interpolation to be applied in a tight loop without
/// re-computing the distortion inversion for every pixel at runtime.
#[derive(Debug, Clone)]
pub struct UndistortLut {
    /// Width of the output frame.
    pub width: u32,
    /// Height of the output frame.
    pub height: u32,
    /// Flattened `(src_x, src_y)` pairs in row-major order.
    /// Length = width × height × 2.
    pub data: Vec<f32>,
}

impl UndistortLut {
    /// Build an undistortion LUT for the given intrinsics and distortion.
    ///
    /// # Parameters
    ///
    /// * `intrinsics` — camera K matrix
    /// * `coeffs` — Brown-Conrady distortion coefficients
    /// * `width`, `height` — output frame dimensions
    /// * `new_cx`, `new_cy` — principal point in the undistorted output
    ///   (typically the same as the source or shifted for optimal cropping)
    #[must_use]
    pub fn build(
        intrinsics: &CameraIntrinsics,
        coeffs: &DistortionCoeffs,
        width: u32,
        height: u32,
        new_cx: f64,
        new_cy: f64,
    ) -> Self {
        let n = (width as usize) * (height as usize) * 2;
        let mut data = Vec::with_capacity(n);

        for row in 0..height {
            for col in 0..width {
                // Normalized coords in the *output* (undistorted) image
                let x_n = (col as f64 - new_cx) / intrinsics.fx;
                let y_n = (row as f64 - new_cy) / intrinsics.fy;

                // Apply forward distortion to find where this undistorted
                // point lands in the distorted (source) image
                let (xd, yd) = coeffs.distort(x_n, y_n);

                // Back to pixel coordinates in the source image
                let src_x = intrinsics.fx * xd + intrinsics.cx;
                let src_y = intrinsics.fy * yd + intrinsics.cy;

                data.push(src_x as f32);
                data.push(src_y as f32);
            }
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Build a fisheye equidistant undistortion LUT.
    #[must_use]
    pub fn build_fisheye(
        intrinsics: &CameraIntrinsics,
        fisheye: &FisheyeEquidistant,
        width: u32,
        height: u32,
    ) -> Self {
        let n = (width as usize) * (height as usize) * 2;
        let mut data = Vec::with_capacity(n);

        for row in 0..height {
            for col in 0..width {
                let x_n = (col as f64 - intrinsics.cx) / intrinsics.fx;
                let y_n = (row as f64 - intrinsics.cy) / intrinsics.fy;

                // Undistort: output is rectilinear ← input is fisheye
                let (xu, yu) = fisheye.undistort(x_n, y_n);

                // Re-project into the distorted (source) fisheye pixel coords
                let src_x = intrinsics.fx * xu + intrinsics.cx;
                let src_y = intrinsics.fy * yu + intrinsics.cy;

                data.push(src_x as f32);
                data.push(src_y as f32);
            }
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Look up the source (distorted) pixel coordinate for an output pixel.
    ///
    /// Returns `None` if `col` or `row` are out of bounds.
    #[must_use]
    pub fn get(&self, col: u32, row: u32) -> Option<(f32, f32)> {
        if col >= self.width || row >= self.height {
            return None;
        }
        let idx = ((row as usize) * (self.width as usize) + col as usize) * 2;
        Some((self.data[idx], self.data[idx + 1]))
    }

    /// Apply the LUT to a flat grayscale pixel buffer using nearest-neighbour sampling.
    ///
    /// `src` must have length `src_width × src_height`.
    /// Returns an undistorted output buffer of length `width × height`.
    ///
    /// # Errors
    ///
    /// Returns [`StabilizeError::DimensionMismatch`] if `src` length does not
    /// match `src_width × src_height`.
    pub fn remap_nearest(
        &self,
        src: &[u8],
        src_width: u32,
        src_height: u32,
    ) -> StabilizeResult<Vec<u8>> {
        let expected = (src_width as usize) * (src_height as usize);
        if src.len() != expected {
            return Err(StabilizeError::dimension_mismatch(
                format!("{expected}"),
                format!("{}", src.len()),
            ));
        }

        let out_size = (self.width as usize) * (self.height as usize);
        let mut dst = vec![0u8; out_size];

        for row in 0..self.height {
            for col in 0..self.width {
                let (sx, sy) = self.get(col, row).unwrap_or((0.0, 0.0));
                let sx_i = sx.round() as i32;
                let sy_i = sy.round() as i32;
                let out_idx = (row as usize) * (self.width as usize) + col as usize;
                if sx_i >= 0 && sy_i >= 0 && (sx_i as u32) < src_width && (sy_i as u32) < src_height
                {
                    let src_idx = (sy_i as usize) * (src_width as usize) + sx_i as usize;
                    dst[out_idx] = src[src_idx];
                }
                // Out-of-bounds → leave 0 (black)
            }
        }

        Ok(dst)
    }

    /// Apply the LUT with bilinear interpolation.
    ///
    /// Produces higher quality than `remap_nearest` at the cost of ~4× more
    /// memory accesses per pixel.
    ///
    /// # Errors
    ///
    /// Returns [`StabilizeError::DimensionMismatch`] if `src` length is wrong.
    pub fn remap_bilinear(
        &self,
        src: &[u8],
        src_width: u32,
        src_height: u32,
    ) -> StabilizeResult<Vec<u8>> {
        let expected = (src_width as usize) * (src_height as usize);
        if src.len() != expected {
            return Err(StabilizeError::dimension_mismatch(
                format!("{expected}"),
                format!("{}", src.len()),
            ));
        }

        let out_size = (self.width as usize) * (self.height as usize);
        let mut dst = vec![0u8; out_size];
        let sw = src_width as usize;
        let sh = src_height as usize;

        for row in 0..self.height {
            for col in 0..self.width {
                let (sx, sy) = self.get(col, row).unwrap_or((0.0, 0.0));
                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;
                let fx = (sx - sx.floor()) as f64;
                let fy = (sy - sy.floor()) as f64;
                let out_idx = (row as usize) * (self.width as usize) + col as usize;

                let sample = |px: i32, py: i32| -> f64 {
                    if px >= 0 && py >= 0 && (px as usize) < sw && (py as usize) < sh {
                        src[py as usize * sw + px as usize] as f64
                    } else {
                        0.0
                    }
                };

                let v00 = sample(x0, y0);
                let v10 = sample(x0 + 1, y0);
                let v01 = sample(x0, y0 + 1);
                let v11 = sample(x0 + 1, y0 + 1);

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                dst[out_idx] = v.round() as u8;
            }
        }

        Ok(dst)
    }
}

// ─────────────────────────────────────────────────────────────────
//  Optimal new camera matrix
// ─────────────────────────────────────────────────────────────────

/// Computes the optimal new camera principal point to minimize black borders
/// after undistortion.
///
/// This is analogous to `cv2.getOptimalNewCameraMatrix` with `alpha=0.0`.
///
/// The strategy is to find the tightest inscribed rectangle inside the
/// undistorted image boundary so that no black (out-of-bounds) pixels
/// appear in the output.
#[must_use]
pub fn optimal_new_principal_point(
    intrinsics: &CameraIntrinsics,
    coeffs: &DistortionCoeffs,
    width: u32,
    height: u32,
) -> (f64, f64) {
    // Sample several boundary points of the *undistorted* image and map them
    // back to distorted coords; the shrinkage gives us the new principal point.
    let steps = 20usize;
    let mut min_fx = f64::MAX;
    let mut min_fy = f64::MAX;

    // Top/bottom rows
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        for &(col_f, row_f) in &[
            (t * width as f64, 0.0_f64),
            (t * width as f64, height as f64 - 1.0),
            (0.0_f64, t * height as f64),
            (width as f64 - 1.0, t * height as f64),
        ] {
            let (xn, yn) = intrinsics.unproject(col_f, row_f);
            let (xd, yd) = coeffs.undistort_normalized(xn, yn, 30, 1e-7);
            let src_x = intrinsics.fx * xd + intrinsics.cx;
            let src_y = intrinsics.fy * yd + intrinsics.cy;
            // How much of the source boundary is inside frame
            if src_x >= 0.0 && src_x < width as f64 {
                min_fx = min_fx.min((col_f - intrinsics.cx).abs().max(1.0));
            }
            if src_y >= 0.0 && src_y < height as f64 {
                min_fy = min_fy.min((row_f - intrinsics.cy).abs().max(1.0));
            }
        }
    }

    // Fall back to original principal point if sampling gives no info
    if min_fx == f64::MAX {
        min_fx = intrinsics.cx;
    }
    if min_fy == f64::MAX {
        min_fy = intrinsics.cy;
    }

    (
        (intrinsics.cx).min(width as f64 - min_fx),
        (intrinsics.cy).min(height as f64 - min_fy),
    )
}

// ─────────────────────────────────────────────────────────────────
//  High-level corrector
// ─────────────────────────────────────────────────────────────────

/// High-level lens distortion corrector that bundles intrinsics, distortion
/// coefficients, and a pre-built undistortion LUT.
///
/// # Example
///
/// ```
/// use oximedia_stabilize::lens_distortion::{
///     LensDistortionCorrector, CameraIntrinsics, DistortionCoeffs,
/// };
///
/// let intrinsics = CameraIntrinsics::symmetric(800.0, 1280, 720);
/// let coeffs = DistortionCoeffs::k1_only(-0.3);
///
/// let corrector = LensDistortionCorrector::build(&intrinsics, &coeffs, 1280, 720);
///
/// // Create a dummy grayscale frame
/// let src: Vec<u8> = (0..1280 * 720).map(|i| (i % 256) as u8).collect();
/// let dst = corrector.correct_nearest(&src).expect("correction should succeed");
/// assert_eq!(dst.len(), 1280 * 720);
/// ```
#[derive(Debug)]
pub struct LensDistortionCorrector {
    /// Pre-built undistortion LUT.
    pub lut: UndistortLut,
    /// Camera intrinsics.
    pub intrinsics: CameraIntrinsics,
    /// Distortion coefficients.
    pub coeffs: DistortionCoeffs,
    /// Output frame width.
    pub width: u32,
    /// Output frame height.
    pub height: u32,
}

impl LensDistortionCorrector {
    /// Build a corrector by pre-computing the undistortion LUT.
    ///
    /// Uses the original principal point as the output principal point.
    #[must_use]
    pub fn build(
        intrinsics: &CameraIntrinsics,
        coeffs: &DistortionCoeffs,
        width: u32,
        height: u32,
    ) -> Self {
        let lut = UndistortLut::build(
            intrinsics,
            coeffs,
            width,
            height,
            intrinsics.cx,
            intrinsics.cy,
        );
        Self {
            lut,
            intrinsics: *intrinsics,
            coeffs: *coeffs,
            width,
            height,
        }
    }

    /// Build a corrector with an optimized principal point to minimize borders.
    #[must_use]
    pub fn build_optimal(
        intrinsics: &CameraIntrinsics,
        coeffs: &DistortionCoeffs,
        width: u32,
        height: u32,
    ) -> Self {
        let (ncx, ncy) = optimal_new_principal_point(intrinsics, coeffs, width, height);
        let lut = UndistortLut::build(intrinsics, coeffs, width, height, ncx, ncy);
        Self {
            lut,
            intrinsics: *intrinsics,
            coeffs: *coeffs,
            width,
            height,
        }
    }

    /// Build a corrector for fisheye equidistant lenses.
    #[must_use]
    pub fn build_fisheye(
        intrinsics: &CameraIntrinsics,
        fisheye_focal: f64,
        width: u32,
        height: u32,
    ) -> Self {
        let fisheye = FisheyeEquidistant::new(fisheye_focal);
        let lut = UndistortLut::build_fisheye(intrinsics, &fisheye, width, height);
        Self {
            lut,
            intrinsics: *intrinsics,
            coeffs: DistortionCoeffs::default(),
            width,
            height,
        }
    }

    /// Apply nearest-neighbour undistortion to a grayscale buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` does not match `width × height`.
    pub fn correct_nearest(&self, src: &[u8]) -> StabilizeResult<Vec<u8>> {
        self.lut.remap_nearest(src, self.width, self.height)
    }

    /// Apply bilinear undistortion to a grayscale buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` does not match `width × height`.
    pub fn correct_bilinear(&self, src: &[u8]) -> StabilizeResult<Vec<u8>> {
        self.lut.remap_bilinear(src, self.width, self.height)
    }

    /// Undistort a single pixel coordinate (column, row) from distorted to
    /// undistorted space using the Brown-Conrady model.
    ///
    /// Useful for correcting feature track points before motion estimation.
    #[must_use]
    pub fn undistort_point(&self, col: f64, row: f64) -> (f64, f64) {
        let (xn, yn) = self.intrinsics.unproject(col, row);
        let (xu, yu) = self.coeffs.undistort_normalized(xn, yn, 25, 1e-7);
        (
            self.intrinsics.fx * xu + self.intrinsics.cx,
            self.intrinsics.fy * yu + self.intrinsics.cy,
        )
    }
}

// ─────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_intrinsics_project_unproject_roundtrip() {
        let k = CameraIntrinsics::new(800.0, 800.0, 640.0, 360.0);
        let (px, py) = k.project(0.1, 0.2, 1.0);
        let (xn, yn) = k.unproject(px, py);
        assert!((xn - 0.1).abs() < 1e-10);
        assert!((yn - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_intrinsics_validate_ok() {
        let k = CameraIntrinsics::symmetric(600.0, 1280, 720);
        assert!(k.validate().is_ok());
    }

    #[test]
    fn test_intrinsics_validate_bad_fx() {
        let k = CameraIntrinsics::new(0.0, 600.0, 640.0, 360.0);
        assert!(k.validate().is_err());
    }

    #[test]
    fn test_coeffs_identity() {
        let c = DistortionCoeffs::default();
        assert!(c.is_identity());
        let (xd, yd) = c.distort(0.3, 0.4);
        assert!((xd - 0.3).abs() < 1e-12);
        assert!((yd - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_coeffs_k1_barrel_distort() {
        let c = DistortionCoeffs::k1_only(-0.3);
        let (xd, yd) = c.distort(0.5, 0.5);
        // Barrel (k1 < 0) → distorted point moves inward
        assert!(xd.abs() < 0.5);
        assert!(yd.abs() < 0.5);
    }

    #[test]
    fn test_coeffs_k1_pincushion_distort() {
        let c = DistortionCoeffs::k1_only(0.3);
        let (xd, yd) = c.distort(0.5, 0.5);
        // Pincushion (k1 > 0) → distorted point moves outward
        assert!(xd > 0.5);
        assert!(yd > 0.5);
    }

    #[test]
    fn test_undistort_identity_roundtrip() {
        let c = DistortionCoeffs::default();
        let (xu, yu) = c.undistort_normalized(0.4, -0.3, 20, 1e-9);
        assert!((xu - 0.4).abs() < 1e-9);
        assert!((yu + 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_undistort_k1_roundtrip() {
        let c = DistortionCoeffs::k1_only(-0.3);
        // Distort, then undistort; should recover original
        let (xd, yd) = c.distort(0.3, 0.2);
        let (xu, yu) = c.undistort_normalized(xd, yd, 40, 1e-9);
        assert!((xu - 0.3).abs() < 1e-6, "xu={xu} expected 0.3");
        assert!((yu - 0.2).abs() < 1e-6, "yu={yu} expected 0.2");
    }

    #[test]
    fn test_undistort_full_roundtrip() {
        let c = DistortionCoeffs::new(-0.25, 0.08, -0.01, 0.001, -0.001);
        let orig_x = 0.2;
        let orig_y = -0.15;
        let (xd, yd) = c.distort(orig_x, orig_y);
        let (xu, yu) = c.undistort_normalized(xd, yd, 40, 1e-9);
        assert!((xu - orig_x).abs() < 1e-5, "xu={xu}");
        assert!((yu - orig_y).abs() < 1e-5, "yu={yu}");
    }

    #[test]
    fn test_distortion_type_classification() {
        assert_eq!(DistortionCoeffs::k1_only(-0.3).distortion_type(), "barrel");
        assert_eq!(
            DistortionCoeffs::k1_only(0.3).distortion_type(),
            "pincushion"
        );
        assert_eq!(
            DistortionCoeffs::k1_only(0.005).distortion_type(),
            "minimal"
        );
    }

    #[test]
    fn test_fisheye_equidistant_zero() {
        let fe = FisheyeEquidistant::new(600.0);
        let (xd, yd) = fe.distort(0.0, 0.0);
        assert!(xd.abs() < 1e-12);
        assert!(yd.abs() < 1e-12);
    }

    #[test]
    fn test_fisheye_equidistant_roundtrip() {
        let fe = FisheyeEquidistant::new(600.0);
        let (xd, yd) = fe.distort(0.1, 0.15);
        let (xu, yu) = fe.undistort(xd, yd);
        assert!((xu - 0.1).abs() < 1e-9, "xu={xu}");
        assert!((yu - 0.15).abs() < 1e-9, "yu={yu}");
    }

    #[test]
    fn test_fisheye_equisolid_roundtrip() {
        let fe = FisheyeEquisolid::new(600.0);
        let (xd, yd) = fe.distort(0.05, 0.08);
        let (xu, yu) = fe.undistort(xd, yd);
        assert!((xu - 0.05).abs() < 1e-8, "xu={xu}");
        assert!((yu - 0.08).abs() < 1e-8, "yu={yu}");
    }

    #[test]
    fn test_lut_build_dimensions() {
        let k = CameraIntrinsics::symmetric(600.0, 64, 48);
        let c = DistortionCoeffs::k1_only(-0.2);
        let lut = UndistortLut::build(&k, &c, 64, 48, k.cx, k.cy);
        assert_eq!(lut.width, 64);
        assert_eq!(lut.height, 48);
        assert_eq!(lut.data.len(), 64 * 48 * 2);
    }

    #[test]
    fn test_lut_get_out_of_bounds() {
        let k = CameraIntrinsics::symmetric(600.0, 64, 48);
        let c = DistortionCoeffs::default();
        let lut = UndistortLut::build(&k, &c, 64, 48, k.cx, k.cy);
        assert!(lut.get(64, 0).is_none());
        assert!(lut.get(0, 48).is_none());
    }

    #[test]
    fn test_lut_identity_center_pixel() {
        // With no distortion, the center pixel should map back to itself
        let k = CameraIntrinsics::symmetric(600.0, 64, 48);
        let c = DistortionCoeffs::default();
        let lut = UndistortLut::build(&k, &c, 64, 48, k.cx, k.cy);
        let (sx, sy) = lut.get(32, 24).expect("center pixel in bounds");
        assert!((sx - 32.0).abs() < 0.5, "sx={sx} should be ~32");
        assert!((sy - 24.0).abs() < 0.5, "sy={sy} should be ~24");
    }

    #[test]
    fn test_remap_nearest_identity() {
        let k = CameraIntrinsics::symmetric(600.0, 16, 16);
        let c = DistortionCoeffs::default(); // identity distortion
        let lut = UndistortLut::build(&k, &c, 16, 16, k.cx, k.cy);

        let src: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let dst = lut
            .remap_nearest(&src, 16, 16)
            .expect("remap should succeed");

        assert_eq!(dst.len(), 16 * 16);
        // Center pixels should be preserved for identity distortion
        // (boundary pixels may differ due to clamping)
        let center_src = src[8 * 16 + 8];
        let center_dst = dst[8 * 16 + 8];
        assert_eq!(center_src, center_dst, "center pixel should be preserved");
    }

    #[test]
    fn test_remap_bilinear_correct_size() {
        let k = CameraIntrinsics::symmetric(400.0, 32, 24);
        let c = DistortionCoeffs::k1_only(-0.2);
        let lut = UndistortLut::build(&k, &c, 32, 24, k.cx, k.cy);

        let src: Vec<u8> = vec![128u8; 32 * 24];
        let dst = lut
            .remap_bilinear(&src, 32, 24)
            .expect("remap bilinear should succeed");
        assert_eq!(dst.len(), 32 * 24);
    }

    #[test]
    fn test_remap_dimension_mismatch_error() {
        let k = CameraIntrinsics::symmetric(600.0, 16, 16);
        let c = DistortionCoeffs::default();
        let lut = UndistortLut::build(&k, &c, 16, 16, k.cx, k.cy);

        let bad_src = vec![0u8; 10]; // wrong size
        let result = lut.remap_nearest(&bad_src, 16, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrector_build_and_correct() {
        let k = CameraIntrinsics::symmetric(600.0, 64, 48);
        let c = DistortionCoeffs::k1_only(-0.25);
        let corrector = LensDistortionCorrector::build(&k, &c, 64, 48);

        let src: Vec<u8> = (0..(64 * 48)).map(|i| (i % 256) as u8).collect();
        let dst = corrector
            .correct_nearest(&src)
            .expect("correction should work");
        assert_eq!(dst.len(), 64 * 48);
    }

    #[test]
    fn test_corrector_undistort_point_identity() {
        let k = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);
        let c = DistortionCoeffs::default();
        let corrector = LensDistortionCorrector::build(&k, &c, 640, 480);

        let (ux, uy) = corrector.undistort_point(320.0, 240.0);
        assert!((ux - 320.0).abs() < 1e-6);
        assert!((uy - 240.0).abs() < 1e-6);
    }

    #[test]
    fn test_corrector_undistort_point_barrel() {
        let k = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);
        let c = DistortionCoeffs::k1_only(-0.3);
        let corrector = LensDistortionCorrector::build(&k, &c, 640, 480);

        // Off-center point should be moved outward when correcting barrel
        let (ux, uy) = corrector.undistort_point(420.0, 300.0);
        // Barrel correction moves points further from center
        let r_orig = ((420.0 - 320.0_f64).powi(2) + (300.0 - 240.0_f64).powi(2)).sqrt();
        let r_new = ((ux - 320.0).powi(2) + (uy - 240.0).powi(2)).sqrt();
        assert!(
            r_new > r_orig - 1.0,
            "barrel correction should expand point outward"
        );
    }

    #[test]
    fn test_lut_fisheye_build() {
        let k = CameraIntrinsics::symmetric(400.0, 32, 32);
        let fe = FisheyeEquidistant::new(300.0);
        let lut = UndistortLut::build_fisheye(&k, &fe, 32, 32);
        assert_eq!(lut.width, 32);
        assert_eq!(lut.height, 32);
    }

    #[test]
    fn test_corrector_build_fisheye() {
        let k = CameraIntrinsics::symmetric(500.0, 64, 64);
        let corrector = LensDistortionCorrector::build_fisheye(&k, 400.0, 64, 64);
        let src = vec![100u8; 64 * 64];
        let dst = corrector
            .correct_nearest(&src)
            .expect("fisheye correction should work");
        assert_eq!(dst.len(), 64 * 64);
    }

    #[test]
    fn test_corrector_build_optimal_no_panic() {
        let k = CameraIntrinsics::symmetric(600.0, 64, 48);
        let c = DistortionCoeffs::k1_only(-0.2);
        // Should not panic or produce NaN
        let corrector = LensDistortionCorrector::build_optimal(&k, &c, 64, 48);
        assert_eq!(corrector.width, 64);
        assert_eq!(corrector.height, 48);
    }

    #[test]
    fn test_pi_constant_used() {
        // Ensure fisheye math uses pi correctly for edge-case angles
        let fe = FisheyeEquidistant::new(200.0);
        // Point at ~45 deg incidence
        let r45 = (PI / 4.0).tan();
        let (xd, yd) = fe.distort(r45, 0.0);
        let (xu, _) = fe.undistort(xd, yd);
        assert!(
            (xu - r45).abs() < 1e-8,
            "45-deg roundtrip: xu={xu}, expected {r45}"
        );
    }
}
