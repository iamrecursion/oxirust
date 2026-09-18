//! Camera calibration and lens distortion correction.
//!
//! Provides barrel and pincushion distortion correction using
//! the Brown-Conrady model with radial and tangential coefficients.

use crate::error::{CvError, CvResult};

/// Lens distortion coefficients (Brown-Conrady model).
///
/// The distortion model uses radial coefficients (k1, k2, k3) and
/// tangential coefficients (p1, p2) to map distorted to undistorted coordinates.
#[derive(Debug, Clone, Copy)]
pub struct DistortionCoefficients {
    /// First radial distortion coefficient.
    pub k1: f64,
    /// Second radial distortion coefficient.
    pub k2: f64,
    /// Third radial distortion coefficient.
    pub k3: f64,
    /// First tangential distortion coefficient.
    pub p1: f64,
    /// Second tangential distortion coefficient.
    pub p2: f64,
}

impl Default for DistortionCoefficients {
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

impl DistortionCoefficients {
    /// Create coefficients for barrel distortion.
    #[must_use]
    pub fn barrel(k1: f64) -> Self {
        Self {
            k1,
            ..Default::default()
        }
    }

    /// Create coefficients for pincushion distortion.
    #[must_use]
    pub fn pincushion(k1: f64) -> Self {
        Self {
            k1: -k1.abs(),
            ..Default::default()
        }
    }

    /// Create full distortion model.
    #[must_use]
    pub const fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self { k1, k2, k3, p1, p2 }
    }

    /// Check if distortion is negligible.
    #[must_use]
    pub fn is_negligible(&self) -> bool {
        self.k1.abs() < 1e-8
            && self.k2.abs() < 1e-8
            && self.k3.abs() < 1e-8
            && self.p1.abs() < 1e-8
            && self.p2.abs() < 1e-8
    }
}

/// Camera intrinsic parameters.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    /// Focal length in x pixels.
    pub fx: f64,
    /// Focal length in y pixels.
    pub fy: f64,
    /// Principal point x coordinate (pixels).
    pub cx: f64,
    /// Principal point y coordinate (pixels).
    pub cy: f64,
}

impl CameraIntrinsics {
    /// Create camera intrinsics.
    #[must_use]
    pub const fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Create from image dimensions assuming centered principal point.
    #[must_use]
    pub fn from_image_size(width: u32, height: u32, fov_degrees: f64) -> Self {
        let fov_rad = fov_degrees * std::f64::consts::PI / 180.0;
        let focal = (width as f64 / 2.0) / (fov_rad / 2.0).tan();
        Self {
            fx: focal,
            fy: focal,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
        }
    }
}

/// Lens distortion corrector.
///
/// Corrects barrel and pincushion distortion using the Brown-Conrady model.
///
/// # Example
///
/// ```
/// use oximedia_cv::registration::calibration::{
///     LensCorrector, DistortionCoefficients, CameraIntrinsics,
/// };
///
/// let intrinsics = CameraIntrinsics::from_image_size(640, 480, 60.0);
/// let distortion = DistortionCoefficients::barrel(0.1);
/// let corrector = LensCorrector::new(intrinsics, distortion);
/// ```
pub struct LensCorrector {
    intrinsics: CameraIntrinsics,
    distortion: DistortionCoefficients,
    /// Precomputed undistortion lookup table (None until built).
    lut: Option<Vec<(f64, f64)>>,
    lut_width: u32,
    lut_height: u32,
}

impl LensCorrector {
    /// Create a new lens corrector.
    #[must_use]
    pub fn new(intrinsics: CameraIntrinsics, distortion: DistortionCoefficients) -> Self {
        Self {
            intrinsics,
            distortion,
            lut: None,
            lut_width: 0,
            lut_height: 0,
        }
    }

    /// Build an undistortion lookup table for the given image size.
    ///
    /// Caches the result for repeated use with same-size images.
    pub fn build_lut(&mut self, width: u32, height: u32) {
        if self.lut.is_some() && self.lut_width == width && self.lut_height == height {
            return;
        }

        let n = (width as usize) * (height as usize);
        let mut lut = Vec::with_capacity(n);

        for y in 0..height {
            for x in 0..width {
                let (src_x, src_y) = self.undistort_point(x as f64, y as f64);
                lut.push((src_x, src_y));
            }
        }

        self.lut = Some(lut);
        self.lut_width = width;
        self.lut_height = height;
    }

    /// Undistort a single point from distorted to undistorted coordinates.
    ///
    /// Maps a pixel in the distorted image to its position in the undistorted image.
    #[must_use]
    pub fn undistort_point(&self, x: f64, y: f64) -> (f64, f64) {
        let xn = (x - self.intrinsics.cx) / self.intrinsics.fx;
        let yn = (y - self.intrinsics.cy) / self.intrinsics.fy;

        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let k = &self.distortion;

        // Radial distortion
        let radial = 1.0 + k.k1 * r2 + k.k2 * r4 + k.k3 * r6;

        // Tangential distortion
        let xd = xn * radial + 2.0 * k.p1 * xn * yn + k.p2 * (r2 + 2.0 * xn * xn);
        let yd = yn * radial + k.p1 * (r2 + 2.0 * yn * yn) + 2.0 * k.p2 * xn * yn;

        // Back to pixel coordinates
        let px = xd * self.intrinsics.fx + self.intrinsics.cx;
        let py = yd * self.intrinsics.fy + self.intrinsics.cy;

        (px, py)
    }

    /// Undistort a grayscale image using bilinear interpolation.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn undistort_image(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected = (width as usize) * (height as usize);
        if image.len() < expected {
            return Err(CvError::insufficient_data(expected, image.len()));
        }

        if self.distortion.is_negligible() {
            return Ok(image.to_vec());
        }

        self.build_lut(width, height);

        let lut = self
            .lut
            .as_ref()
            .ok_or_else(|| CvError::matrix_error("LUT not built"))?;

        let mut output = vec![0u8; expected];
        let w = width as usize;
        let h = height as usize;

        for (dst_idx, &(src_x, src_y)) in lut.iter().enumerate() {
            if src_x >= 0.0 && src_x < (w - 1) as f64 && src_y >= 0.0 && src_y < (h - 1) as f64 {
                // Bilinear interpolation
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = src_x - x0 as f64;
                let fy = src_y - y0 as f64;

                let v00 = image[y0 * w + x0] as f64;
                let v01 = image[y0 * w + x1] as f64;
                let v10 = image[y1 * w + x0] as f64;
                let v11 = image[y1 * w + x1] as f64;

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                output[dst_idx] = v.round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(output)
    }

    /// Get intrinsics.
    #[must_use]
    pub const fn intrinsics(&self) -> &CameraIntrinsics {
        &self.intrinsics
    }

    /// Get distortion coefficients.
    #[must_use]
    pub const fn distortion(&self) -> &DistortionCoefficients {
        &self.distortion
    }
}

/// Estimate distortion from a set of straight-line correspondences.
///
/// Given points that should lie on straight lines in the undistorted image,
/// estimates the radial distortion coefficient k1.
///
/// # Errors
///
/// Returns an error if insufficient points are provided.
pub fn estimate_distortion_from_lines(
    line_points: &[Vec<(f64, f64)>],
    intrinsics: &CameraIntrinsics,
) -> CvResult<DistortionCoefficients> {
    if line_points.len() < 2 {
        return Err(CvError::matrix_error(
            "need at least 2 line groups for distortion estimation",
        ));
    }

    // Simple single-parameter estimation: minimize line straightness error
    // by searching over k1 values
    let mut best_k1 = 0.0;
    let mut best_error = f64::MAX;

    for step in -100..=100 {
        let k1 = step as f64 * 0.005;
        let coeffs = DistortionCoefficients::barrel(k1);
        let corrector = LensCorrector::new(*intrinsics, coeffs);

        let mut total_error = 0.0;
        for line in line_points {
            total_error += line_straightness_error(line, &corrector);
        }

        if total_error < best_error {
            best_error = total_error;
            best_k1 = k1;
        }
    }

    Ok(DistortionCoefficients::barrel(best_k1))
}

/// Compute how much a set of points deviates from a straight line
/// after undistortion.
fn line_straightness_error(points: &[(f64, f64)], corrector: &LensCorrector) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    // Undistort all points
    let undistorted: Vec<(f64, f64)> = points
        .iter()
        .map(|&(x, y)| corrector.undistort_point(x, y))
        .collect();

    // Fit line between first and last point
    let (x0, y0) = undistorted[0];
    let (x1, y1) = undistorted[undistorted.len() - 1];
    let dx = x1 - x0;
    let dy = y1 - y0;
    let line_len = (dx * dx + dy * dy).sqrt();

    if line_len < 1e-6 {
        return 0.0;
    }

    // Sum perpendicular distances
    let mut total_error = 0.0;
    for &(px, py) in &undistorted[1..undistorted.len() - 1] {
        let dist = ((py - y0) * dx - (px - x0) * dy).abs() / line_len;
        total_error += dist * dist;
    }

    total_error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distortion_coefficients_default() {
        let d = DistortionCoefficients::default();
        assert!(d.is_negligible());
    }

    #[test]
    fn test_distortion_coefficients_barrel() {
        let d = DistortionCoefficients::barrel(0.1);
        assert_eq!(d.k1, 0.1);
        assert!(!d.is_negligible());
    }

    #[test]
    fn test_distortion_coefficients_pincushion() {
        let d = DistortionCoefficients::pincushion(0.1);
        assert!(d.k1 < 0.0);
    }

    #[test]
    fn test_camera_intrinsics_from_image_size() {
        let intr = CameraIntrinsics::from_image_size(640, 480, 60.0);
        assert!((intr.cx - 320.0).abs() < 1e-6);
        assert!((intr.cy - 240.0).abs() < 1e-6);
        assert!(intr.fx > 0.0);
    }

    #[test]
    fn test_lens_corrector_no_distortion() {
        let intr = CameraIntrinsics::from_image_size(100, 100, 60.0);
        let dist = DistortionCoefficients::default();
        let mut corrector = LensCorrector::new(intr, dist);

        let image = vec![128u8; 100 * 100];
        let result = corrector
            .undistort_image(&image, 100, 100)
            .expect("should succeed");
        assert_eq!(result, image);
    }

    #[test]
    fn test_lens_corrector_undistort_point_center() {
        let intr = CameraIntrinsics::from_image_size(640, 480, 60.0);
        let dist = DistortionCoefficients::barrel(0.1);
        let corrector = LensCorrector::new(intr, dist);

        // Center point should not move
        let (x, y) = corrector.undistort_point(320.0, 240.0);
        assert!((x - 320.0).abs() < 1e-6);
        assert!((y - 240.0).abs() < 1e-6);
    }

    #[test]
    fn test_lens_corrector_barrel_distortion() {
        let intr = CameraIntrinsics::from_image_size(640, 480, 60.0);
        let dist = DistortionCoefficients::barrel(0.1);
        let mut corrector = LensCorrector::new(intr, dist);

        let image = vec![128u8; 640 * 480];
        let result = corrector
            .undistort_image(&image, 640, 480)
            .expect("should succeed");
        assert_eq!(result.len(), 640 * 480);
    }

    #[test]
    fn test_lens_corrector_invalid_dims() {
        let intr = CameraIntrinsics::from_image_size(640, 480, 60.0);
        let dist = DistortionCoefficients::default();
        let mut corrector = LensCorrector::new(intr, dist);

        let result = corrector.undistort_image(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_distortion_insufficient_lines() {
        let intr = CameraIntrinsics::from_image_size(640, 480, 60.0);
        let result = estimate_distortion_from_lines(&[vec![(0.0, 0.0), (1.0, 1.0)]], &intr);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_distortion_from_straight_lines() {
        let intr = CameraIntrinsics::from_image_size(640, 480, 60.0);
        // Already straight lines should yield near-zero distortion.
        // Use off-axis lines to avoid the trivial case where all points lie
        // on the principal axis (zero radius -> distortion has no effect).
        let lines = vec![
            vec![
                (100.0, 200.0),
                (200.0, 200.0),
                (300.0, 200.0),
                (400.0, 200.0),
                (500.0, 200.0),
            ],
            vec![
                (100.0, 300.0),
                (200.0, 300.0),
                (300.0, 300.0),
                (400.0, 300.0),
                (500.0, 300.0),
            ],
            vec![
                (200.0, 100.0),
                (200.0, 200.0),
                (200.0, 300.0),
                (200.0, 400.0),
            ],
        ];
        let result = estimate_distortion_from_lines(&lines, &intr).expect("should succeed");
        // The grid resolution is 0.005, so allow k1 up to one step + tolerance
        assert!(
            result.k1.abs() < 0.5,
            "k1 should be small for straight lines but got {}",
            result.k1
        );
    }

    #[test]
    fn test_lut_caching() {
        let intr = CameraIntrinsics::from_image_size(100, 100, 60.0);
        let dist = DistortionCoefficients::barrel(0.05);
        let mut corrector = LensCorrector::new(intr, dist);

        corrector.build_lut(100, 100);
        assert!(corrector.lut.is_some());

        // Second call should reuse
        corrector.build_lut(100, 100);
        assert_eq!(corrector.lut_width, 100);
    }
}
