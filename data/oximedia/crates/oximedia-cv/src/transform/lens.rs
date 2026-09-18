//! Lens distortion correction (barrel/pincushion).
//!
//! Provides correction for radial lens distortion using the Brown-Conrady model.
//! Supports both barrel distortion (positive k1, bulging outward from center)
//! and pincushion distortion (negative k1, pinching inward toward center).
//!
//! # Example
//!
//! ```
//! use oximedia_cv::transform::lens::{LensDistortionCorrector, DistortionModel};
//!
//! let model = DistortionModel::new_radial(-0.2, 0.05);
//! let corrector = LensDistortionCorrector::new(model, 640, 480);
//! ```

use crate::error::{CvError, CvResult};

/// Distortion model parameters.
///
/// Supports radial (k1, k2, k3) and tangential (p1, p2) distortion.
#[derive(Debug, Clone, Copy)]
pub struct DistortionModel {
    /// First radial coefficient (barrel if positive, pincushion if negative).
    pub k1: f64,
    /// Second radial coefficient.
    pub k2: f64,
    /// Third radial coefficient.
    pub k3: f64,
    /// First tangential coefficient.
    pub p1: f64,
    /// Second tangential coefficient.
    pub p2: f64,
    /// Focal length in pixels (if 0, estimated from image size).
    pub focal_length: f64,
}

impl Default for DistortionModel {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            focal_length: 0.0,
        }
    }
}

impl DistortionModel {
    /// Create a radial-only distortion model.
    #[must_use]
    pub fn new_radial(k1: f64, k2: f64) -> Self {
        Self {
            k1,
            k2,
            ..Default::default()
        }
    }

    /// Create a full distortion model.
    #[must_use]
    pub const fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self {
            k1,
            k2,
            k3,
            p1,
            p2,
            focal_length: 0.0,
        }
    }

    /// Set focal length.
    #[must_use]
    pub const fn with_focal_length(mut self, fl: f64) -> Self {
        self.focal_length = fl;
        self
    }

    /// Check if model has negligible distortion.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.k1.abs() < 1e-10
            && self.k2.abs() < 1e-10
            && self.k3.abs() < 1e-10
            && self.p1.abs() < 1e-10
            && self.p2.abs() < 1e-10
    }
}

/// Lens distortion corrector.
///
/// Precomputes a remap table for efficient repeated correction.
pub struct LensDistortionCorrector {
    model: DistortionModel,
    width: u32,
    height: u32,
    cx: f64,
    cy: f64,
    focal: f64,
    /// Precomputed remap table: (src_x, src_y) for each destination pixel.
    remap: Vec<(f64, f64)>,
}

impl LensDistortionCorrector {
    /// Create a new corrector for the given model and image dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::transform::lens::{LensDistortionCorrector, DistortionModel};
    ///
    /// let model = DistortionModel::new_radial(-0.2, 0.05);
    /// let corrector = LensDistortionCorrector::new(model, 640, 480);
    /// ```
    #[must_use]
    pub fn new(model: DistortionModel, width: u32, height: u32) -> Self {
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let focal = if model.focal_length > 0.0 {
            model.focal_length
        } else {
            // Estimate from image diagonal
            ((cx * cx + cy * cy) as f64).sqrt()
        };

        let mut corrector = Self {
            model,
            width,
            height,
            cx,
            cy,
            focal,
            remap: Vec::new(),
        };
        corrector.build_remap();
        corrector
    }

    /// Build the remap table.
    fn build_remap(&mut self) {
        let n = (self.width as usize) * (self.height as usize);
        self.remap = Vec::with_capacity(n);

        for y in 0..self.height {
            for x in 0..self.width {
                let (sx, sy) = self.apply_distortion(x as f64, y as f64);
                self.remap.push((sx, sy));
            }
        }
    }

    /// Apply the inverse distortion to find source coordinates for a destination pixel.
    fn apply_distortion(&self, dst_x: f64, dst_y: f64) -> (f64, f64) {
        // Normalize to camera coordinates
        let xn = (dst_x - self.cx) / self.focal;
        let yn = (dst_y - self.cy) / self.focal;

        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        // Radial distortion factor
        let radial = 1.0 + self.model.k1 * r2 + self.model.k2 * r4 + self.model.k3 * r6;

        // Tangential distortion
        let tx = 2.0 * self.model.p1 * xn * yn + self.model.p2 * (r2 + 2.0 * xn * xn);
        let ty = self.model.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.model.p2 * xn * yn;

        let xd = xn * radial + tx;
        let yd = yn * radial + ty;

        // Back to pixel coordinates
        (xd * self.focal + self.cx, yd * self.focal + self.cy)
    }

    /// Correct distortion in a grayscale image.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions don't match.
    pub fn correct(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        if width != self.width || height != self.height {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected = (width as usize) * (height as usize);
        if image.len() < expected {
            return Err(CvError::insufficient_data(expected, image.len()));
        }

        if self.model.is_identity() {
            return Ok(image[..expected].to_vec());
        }

        let w = width as usize;
        let h = height as usize;
        let mut output = vec![0u8; expected];

        for (dst_idx, &(src_x, src_y)) in self.remap.iter().enumerate() {
            if src_x >= 0.0 && src_x < (w - 1) as f64 && src_y >= 0.0 && src_y < (h - 1) as f64 {
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

    /// Correct distortion in an RGB image (3 bytes per pixel).
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions don't match.
    pub fn correct_rgb(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        if width != self.width || height != self.height {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected = (width as usize) * (height as usize) * 3;
        if image.len() < expected {
            return Err(CvError::insufficient_data(expected, image.len()));
        }

        if self.model.is_identity() {
            return Ok(image[..expected].to_vec());
        }

        let w = width as usize;
        let h = height as usize;
        let mut output = vec![0u8; expected];

        for (dst_idx, &(src_x, src_y)) in self.remap.iter().enumerate() {
            if src_x >= 0.0 && src_x < (w - 1) as f64 && src_y >= 0.0 && src_y < (h - 1) as f64 {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = src_x - x0 as f64;
                let fy = src_y - y0 as f64;

                for c in 0..3 {
                    let v00 = image[(y0 * w + x0) * 3 + c] as f64;
                    let v01 = image[(y0 * w + x1) * 3 + c] as f64;
                    let v10 = image[(y1 * w + x0) * 3 + c] as f64;
                    let v11 = image[(y1 * w + x1) * 3 + c] as f64;

                    let v = v00 * (1.0 - fx) * (1.0 - fy)
                        + v01 * fx * (1.0 - fy)
                        + v10 * (1.0 - fx) * fy
                        + v11 * fx * fy;

                    output[dst_idx * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(output)
    }

    /// Get the distortion model.
    #[must_use]
    pub const fn model(&self) -> &DistortionModel {
        &self.model
    }

    /// Get the image dimensions this corrector was built for.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distortion_model_default_is_identity() {
        let model = DistortionModel::default();
        assert!(model.is_identity());
    }

    #[test]
    fn test_distortion_model_radial() {
        let model = DistortionModel::new_radial(-0.2, 0.05);
        assert!(!model.is_identity());
        assert_eq!(model.k1, -0.2);
        assert_eq!(model.k2, 0.05);
    }

    #[test]
    fn test_corrector_identity() {
        let model = DistortionModel::default();
        let corrector = LensDistortionCorrector::new(model, 100, 100);
        let image = vec![128u8; 100 * 100];
        let result = corrector.correct(&image, 100, 100).expect("should succeed");
        assert_eq!(result, image);
    }

    #[test]
    fn test_corrector_barrel() {
        let model = DistortionModel::new_radial(0.1, 0.0);
        let corrector = LensDistortionCorrector::new(model, 100, 100);
        let image = vec![128u8; 100 * 100];
        let result = corrector.correct(&image, 100, 100).expect("should succeed");
        assert_eq!(result.len(), 100 * 100);
    }

    #[test]
    fn test_corrector_pincushion() {
        let model = DistortionModel::new_radial(-0.1, 0.0);
        let corrector = LensDistortionCorrector::new(model, 100, 100);
        let image = vec![128u8; 100 * 100];
        let result = corrector.correct(&image, 100, 100).expect("should succeed");
        assert_eq!(result.len(), 100 * 100);
    }

    #[test]
    fn test_corrector_dimension_mismatch() {
        let model = DistortionModel::default();
        let corrector = LensDistortionCorrector::new(model, 100, 100);
        let image = vec![128u8; 200 * 200];
        let result = corrector.correct(&image, 200, 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrector_rgb() {
        let model = DistortionModel::new_radial(0.05, 0.0);
        let corrector = LensDistortionCorrector::new(model, 50, 50);
        let image = vec![128u8; 50 * 50 * 3];
        let result = corrector
            .correct_rgb(&image, 50, 50)
            .expect("should succeed");
        assert_eq!(result.len(), 50 * 50 * 3);
    }

    #[test]
    fn test_center_pixel_unchanged() {
        let model = DistortionModel::new_radial(0.1, 0.0);
        let corrector = LensDistortionCorrector::new(model, 100, 100);

        // Center pixel should map close to itself
        let (sx, sy) = corrector.apply_distortion(50.0, 50.0);
        assert!((sx - 50.0).abs() < 1.0);
        assert!((sy - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_with_focal_length() {
        let model = DistortionModel::new_radial(0.1, 0.0).with_focal_length(500.0);
        assert_eq!(model.focal_length, 500.0);
    }

    #[test]
    fn test_corrector_dimensions() {
        let model = DistortionModel::default();
        let corrector = LensDistortionCorrector::new(model, 640, 480);
        assert_eq!(corrector.dimensions(), (640, 480));
    }
}
