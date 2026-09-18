//! Lens distortion correction.
//!
//! This module provides lens distortion modeling and correction:
//!
//! - Brown-Conrady radial distortion model
//! - Tangential distortion
//! - Fisheye lens model
//! - Camera calibration
//! - Real-time undistortion

use crate::{AlignError, AlignResult, Point2D};
use nalgebra::{Matrix3, Vector2};

/// Camera intrinsic parameters
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    /// Focal length X (pixels)
    pub fx: f64,
    /// Focal length Y (pixels)
    pub fy: f64,
    /// Principal point X (pixels)
    pub cx: f64,
    /// Principal point Y (pixels)
    pub cy: f64,
}

impl CameraIntrinsics {
    /// Create new camera intrinsics
    #[must_use]
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Create from image dimensions (assuming centered principal point)
    #[must_use]
    pub fn from_image_size(width: usize, height: usize, fov_degrees: f64) -> Self {
        let fov_rad = fov_degrees.to_radians();
        let fx = (width as f64 / 2.0) / (fov_rad / 2.0).tan();
        let fy = fx; // Square pixels
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;

        Self { fx, fy, cx, cy }
    }

    /// Get camera matrix K
    #[must_use]
    pub fn to_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    /// Convert pixel to normalized coordinates
    #[must_use]
    pub fn pixel_to_normalized(&self, pixel: &Point2D) -> Vector2<f64> {
        Vector2::new((pixel.x - self.cx) / self.fx, (pixel.y - self.cy) / self.fy)
    }

    /// Convert normalized to pixel coordinates
    #[must_use]
    pub fn normalized_to_pixel(&self, normalized: &Vector2<f64>) -> Point2D {
        Point2D::new(
            normalized[0] * self.fx + self.cx,
            normalized[1] * self.fy + self.cy,
        )
    }
}

/// Brown-Conrady distortion model
#[derive(Debug, Clone)]
pub struct BrownConradyDistortion {
    /// Radial distortion coefficients [k1, k2, k3]
    pub radial: [f64; 3],
    /// Tangential distortion coefficients [p1, p2]
    pub tangential: [f64; 2],
}

impl Default for BrownConradyDistortion {
    fn default() -> Self {
        Self {
            radial: [0.0, 0.0, 0.0],
            tangential: [0.0, 0.0],
        }
    }
}

impl BrownConradyDistortion {
    /// Create new distortion model
    #[must_use]
    pub fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self {
            radial: [k1, k2, k3],
            tangential: [p1, p2],
        }
    }

    /// Apply distortion to normalized coordinates
    #[must_use]
    pub fn distort(&self, point: &Vector2<f64>) -> Vector2<f64> {
        let x = point[0];
        let y = point[1];
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let k1 = self.radial[0];
        let k2 = self.radial[1];
        let k3 = self.radial[2];
        let p1 = self.tangential[0];
        let p2 = self.tangential[1];

        // Radial distortion
        let radial_distortion = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        // Tangential distortion
        let x_tangential = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let y_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        Vector2::new(
            x * radial_distortion + x_tangential,
            y * radial_distortion + y_tangential,
        )
    }

    /// Remove distortion from normalized coordinates (iterative)
    #[must_use]
    pub fn undistort(&self, distorted: &Vector2<f64>) -> Vector2<f64> {
        let mut undistorted = *distorted;

        // Iterative refinement (typically converges in 5-10 iterations)
        for _ in 0..10 {
            let distorted_estimate = self.distort(&undistorted);
            let error = distorted - distorted_estimate;

            undistorted += error;

            // Check convergence
            if error.norm() < 1e-8 {
                break;
            }
        }

        undistorted
    }
}

/// Fisheye distortion model
#[derive(Debug, Clone)]
pub struct FisheyeDistortion {
    /// Fisheye distortion coefficients [k1, k2, k3, k4]
    pub coefficients: [f64; 4],
}

impl Default for FisheyeDistortion {
    fn default() -> Self {
        Self {
            coefficients: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

impl FisheyeDistortion {
    /// Create new fisheye distortion model
    #[must_use]
    pub fn new(k1: f64, k2: f64, k3: f64, k4: f64) -> Self {
        Self {
            coefficients: [k1, k2, k3, k4],
        }
    }

    /// Apply fisheye distortion
    #[must_use]
    pub fn distort(&self, point: &Vector2<f64>) -> Vector2<f64> {
        let x = point[0];
        let y = point[1];
        let r = (x * x + y * y).sqrt();

        if r < 1e-10 {
            return *point;
        }

        let theta = r.atan();
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta6 * theta2;

        let k1 = self.coefficients[0];
        let k2 = self.coefficients[1];
        let k3 = self.coefficients[2];
        let k4 = self.coefficients[3];

        let theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

        let scale = theta_d / r;

        Vector2::new(x * scale, y * scale)
    }

    /// Remove fisheye distortion
    #[must_use]
    pub fn undistort(&self, distorted: &Vector2<f64>) -> Vector2<f64> {
        let x = distorted[0];
        let y = distorted[1];
        let r = (x * x + y * y).sqrt();

        if r < 1e-10 {
            return *distorted;
        }

        // Iterative solution for theta
        let mut theta = r;
        for _ in 0..10 {
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta6 * theta2;

            let k1 = self.coefficients[0];
            let k2 = self.coefficients[1];
            let k3 = self.coefficients[2];
            let k4 = self.coefficients[3];

            let theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            let error = theta_d - r;

            if error.abs() < 1e-8 {
                break;
            }

            // Newton's method derivative
            let derivative =
                1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8;

            theta -= error / derivative;
        }

        let scale = theta.tan() / r;
        Vector2::new(x * scale, y * scale)
    }
}

/// Complete camera model with intrinsics and distortion
pub struct CameraModel {
    /// Camera intrinsics
    pub intrinsics: CameraIntrinsics,
    /// Distortion model
    pub distortion: DistortionModel,
}

/// Distortion model variants
#[derive(Debug, Clone)]
pub enum DistortionModel {
    /// No distortion
    None,
    /// Brown-Conrady model
    BrownConrady(BrownConradyDistortion),
    /// Fisheye model
    Fisheye(FisheyeDistortion),
}

impl CameraModel {
    /// Create new camera model
    #[must_use]
    pub fn new(intrinsics: CameraIntrinsics, distortion: DistortionModel) -> Self {
        Self {
            intrinsics,
            distortion,
        }
    }

    /// Project 3D point to pixel coordinates
    #[must_use]
    pub fn project(&self, point_3d: &nalgebra::Vector3<f64>) -> Point2D {
        // Normalize by Z
        let normalized = Vector2::new(point_3d[0] / point_3d[2], point_3d[1] / point_3d[2]);

        // Apply distortion
        let distorted = match &self.distortion {
            DistortionModel::None => normalized,
            DistortionModel::BrownConrady(d) => d.distort(&normalized),
            DistortionModel::Fisheye(d) => d.distort(&normalized),
        };

        // Convert to pixels
        self.intrinsics.normalized_to_pixel(&distorted)
    }

    /// Unproject pixel to normalized ray direction
    #[must_use]
    pub fn unproject(&self, pixel: &Point2D) -> Vector2<f64> {
        // Convert to normalized coordinates
        let distorted = self.intrinsics.pixel_to_normalized(pixel);

        // Remove distortion
        match &self.distortion {
            DistortionModel::None => distorted,
            DistortionModel::BrownConrady(d) => d.undistort(&distorted),
            DistortionModel::Fisheye(d) => d.undistort(&distorted),
        }
    }
}

/// Image undistorter with precomputed lookup tables
pub struct ImageUndistorter {
    /// Camera model
    pub camera: CameraModel,
    /// Output width
    pub width: usize,
    /// Output height
    pub height: usize,
    /// Precomputed undistortion map (x coordinates)
    map_x: Vec<f32>,
    /// Precomputed undistortion map (y coordinates)
    map_y: Vec<f32>,
}

impl ImageUndistorter {
    /// Create new undistorter with precomputed maps
    #[must_use]
    pub fn new(camera: CameraModel, width: usize, height: usize) -> Self {
        let mut map_x = vec![0.0; width * height];
        let mut map_y = vec![0.0; width * height];

        // Precompute undistortion map
        for y in 0..height {
            for x in 0..width {
                let pixel = Point2D::new(x as f64, y as f64);
                let undistorted = camera.unproject(&pixel);
                let source = camera.intrinsics.normalized_to_pixel(&undistorted);

                let idx = y * width + x;
                map_x[idx] = source.x as f32;
                map_y[idx] = source.y as f32;
            }
        }

        Self {
            camera,
            width,
            height,
            map_x,
            map_y,
        }
    }

    /// Undistort an image using bilinear interpolation
    ///
    /// # Errors
    /// Returns error if image size doesn't match
    pub fn undistort(&self, input: &[u8], channels: usize) -> AlignResult<Vec<u8>> {
        let expected_size = self.width * self.height * channels;
        if input.len() != expected_size {
            return Err(AlignError::InvalidConfig(format!(
                "Input size {} doesn't match expected {}",
                input.len(),
                expected_size
            )));
        }

        let mut output = vec![0u8; expected_size];

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let src_x = self.map_x[idx];
                let src_y = self.map_y[idx];

                // Bilinear interpolation
                let x0 = src_x.floor() as isize;
                let y0 = src_y.floor() as isize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;

                // Check bounds
                if x0 >= 0 && x1 < self.width as isize && y0 >= 0 && y1 < self.height as isize {
                    for c in 0..channels {
                        let i00 = ((y0 as usize) * self.width + (x0 as usize)) * channels + c;
                        let i10 = ((y0 as usize) * self.width + (x1 as usize)) * channels + c;
                        let i01 = ((y1 as usize) * self.width + (x0 as usize)) * channels + c;
                        let i11 = ((y1 as usize) * self.width + (x1 as usize)) * channels + c;

                        let v00 = f32::from(input[i00]);
                        let v10 = f32::from(input[i10]);
                        let v01 = f32::from(input[i01]);
                        let v11 = f32::from(input[i11]);

                        let v0 = v00 * (1.0 - dx) + v10 * dx;
                        let v1 = v01 * (1.0 - dx) + v11 * dx;
                        let v = v0 * (1.0 - dy) + v1 * dy;

                        output[idx * channels + c] = v.round() as u8;
                    }
                }
            }
        }

        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fisheye equidistant projection model
// ─────────────────────────────────────────────────────────────────────────────

/// Fisheye equidistant projection model.
///
/// In the equidistant model the distorted radius is simply `r_d = f * θ`,
/// where `θ = atan(r)` and `f` is a scale factor.  No polynomial coefficients
/// are needed; the `scale` field corresponds to the focal-length equivalent.
#[derive(Debug, Clone)]
pub struct FisheyeEquidistant {
    /// Scale factor (equivalent focal length in the distorted image plane)
    pub scale: f64,
}

impl FisheyeEquidistant {
    /// Create a new equidistant fisheye model.
    #[must_use]
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }

    /// Apply equidistant fisheye projection to a normalised point.
    ///
    /// Returns the distorted point in normalised coordinates.
    #[must_use]
    pub fn distort(&self, point: &Vector2<f64>) -> Vector2<f64> {
        let x = point[0];
        let y = point[1];
        let r = (x * x + y * y).sqrt();

        if r < 1e-10 {
            return *point;
        }

        let theta = r.atan();
        let r_d = self.scale * theta;
        let scale = r_d / r;

        Vector2::new(x * scale, y * scale)
    }

    /// Invert the equidistant projection.
    ///
    /// Given a distorted normalised point, recover the undistorted point.
    #[must_use]
    pub fn undistort(&self, distorted: &Vector2<f64>) -> Vector2<f64> {
        let x = distorted[0];
        let y = distorted[1];
        let r_d = (x * x + y * y).sqrt();

        if r_d < 1e-10 {
            return *distorted;
        }

        // θ = r_d / scale  →  r = tan(θ)
        let theta = r_d / self.scale;
        let r = theta.tan();
        let scale = r / r_d;

        Vector2::new(x * scale, y * scale)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stereographic projection model
// ─────────────────────────────────────────────────────────────────────────────

/// Stereographic fisheye projection model.
///
/// In the stereographic model the distorted radius is
/// `r_d = 2 * f * tan(θ / 2)`, where `θ = atan(r)` and `f` is the scale
/// factor.  This projection preserves circles (conformal mapping).
#[derive(Debug, Clone)]
pub struct StereographicProjection {
    /// Scale factor
    pub scale: f64,
}

impl StereographicProjection {
    /// Create a new stereographic projection model.
    #[must_use]
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }

    /// Apply the stereographic projection to a normalised point.
    #[must_use]
    pub fn distort(&self, point: &Vector2<f64>) -> Vector2<f64> {
        let x = point[0];
        let y = point[1];
        let r = (x * x + y * y).sqrt();

        if r < 1e-10 {
            return *point;
        }

        let theta = r.atan();
        let r_d = 2.0 * self.scale * (theta / 2.0).tan();
        let scale = r_d / r;

        Vector2::new(x * scale, y * scale)
    }

    /// Invert the stereographic projection.
    #[must_use]
    pub fn undistort(&self, distorted: &Vector2<f64>) -> Vector2<f64> {
        let x = distorted[0];
        let y = distorted[1];
        let r_d = (x * x + y * y).sqrt();

        if r_d < 1e-10 {
            return *distorted;
        }

        // r_d = 2*scale*tan(θ/2)  →  θ/2 = atan(r_d / (2*scale))
        let theta = 2.0 * (r_d / (2.0 * self.scale)).atan();
        let r = theta.tan();
        let scale = r / r_d;

        Vector2::new(x * scale, y * scale)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended distortion-model enum (includes new models)
// ─────────────────────────────────────────────────────────────────────────────

/// Extended set of distortion model variants (includes projection models).
#[derive(Debug, Clone)]
pub enum ExtendedDistortionModel {
    /// No distortion
    None,
    /// Brown-Conrady radial + tangential distortion
    BrownConrady(BrownConradyDistortion),
    /// Fisheye polynomial model (OpenCV-style)
    Fisheye(FisheyeDistortion),
    /// Fisheye equidistant projection
    FisheyeEquidistant(FisheyeEquidistant),
    /// Stereographic (conformal) fisheye projection
    Stereographic(StereographicProjection),
}

impl ExtendedDistortionModel {
    /// Apply distortion to a normalised coordinate.
    #[must_use]
    pub fn distort(&self, point: &Vector2<f64>) -> Vector2<f64> {
        match self {
            Self::None => *point,
            Self::BrownConrady(m) => m.distort(point),
            Self::Fisheye(m) => m.distort(point),
            Self::FisheyeEquidistant(m) => m.distort(point),
            Self::Stereographic(m) => m.distort(point),
        }
    }

    /// Remove distortion from a normalised coordinate.
    #[must_use]
    pub fn undistort(&self, point: &Vector2<f64>) -> Vector2<f64> {
        match self {
            Self::None => *point,
            Self::BrownConrady(m) => m.undistort(point),
            Self::Fisheye(m) => m.undistort(point),
            Self::FisheyeEquidistant(m) => m.undistort(point),
            Self::Stereographic(m) => m.undistort(point),
        }
    }
}

/// Camera calibration using checkerboard pattern
pub struct CameraCalibrator {
    /// Checkerboard width (interior corners)
    pub board_width: usize,
    /// Checkerboard height (interior corners)
    pub board_height: usize,
    /// Square size in real-world units
    pub square_size: f64,
}

impl CameraCalibrator {
    /// Create new calibrator
    #[must_use]
    pub fn new(board_width: usize, board_height: usize, square_size: f64) -> Self {
        Self {
            board_width,
            board_height,
            square_size,
        }
    }

    /// Calibrate camera from multiple views of checkerboard
    ///
    /// # Errors
    /// Returns error if calibration fails
    #[allow(dead_code)]
    pub fn calibrate(
        &self,
        image_points: &[Vec<Point2D>],
        image_width: usize,
        image_height: usize,
    ) -> AlignResult<CameraModel> {
        if image_points.is_empty() {
            return Err(AlignError::InsufficientData(
                "Need at least one image for calibration".to_string(),
            ));
        }

        // Generate object points (3D world coordinates)
        let _object_points = self.generate_object_points();

        // Initial guess for intrinsics
        let intrinsics = CameraIntrinsics::from_image_size(image_width, image_height, 60.0);

        // Simplified calibration: return default model
        // In production, this would use iterative optimization
        Ok(CameraModel::new(
            intrinsics,
            DistortionModel::BrownConrady(BrownConradyDistortion::default()),
        ))
    }

    /// Generate 3D object points for checkerboard
    fn generate_object_points(&self) -> Vec<nalgebra::Vector3<f64>> {
        let mut points = Vec::new();
        for y in 0..self.board_height {
            for x in 0..self.board_width {
                points.push(nalgebra::Vector3::new(
                    x as f64 * self.square_size,
                    y as f64 * self.square_size,
                    0.0,
                ));
            }
        }
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── New projection model tests ────────────────────────────────────────

    #[test]
    fn test_fisheye_equidistant_identity_at_origin() {
        let model = FisheyeEquidistant::new(1.0);
        let origin = Vector2::new(0.0, 0.0);
        let distorted = model.distort(&origin);
        assert!((distorted[0]).abs() < 1e-10);
        assert!((distorted[1]).abs() < 1e-10);
    }

    #[test]
    fn test_fisheye_equidistant_roundtrip() {
        let model = FisheyeEquidistant::new(1.0);
        let point = Vector2::new(0.4, 0.3);
        let distorted = model.distort(&point);
        let recovered = model.undistort(&distorted);
        assert!((recovered[0] - point[0]).abs() < 1e-8);
        assert!((recovered[1] - point[1]).abs() < 1e-8);
    }

    #[test]
    fn test_fisheye_equidistant_scale_effect() {
        // Higher scale → larger distorted radius for the same input
        let m1 = FisheyeEquidistant::new(1.0);
        let m2 = FisheyeEquidistant::new(2.0);
        let point = Vector2::new(0.5, 0.5);
        let d1 = m1.distort(&point);
        let d2 = m2.distort(&point);
        let r1 = (d1[0] * d1[0] + d1[1] * d1[1]).sqrt();
        let r2 = (d2[0] * d2[0] + d2[1] * d2[1]).sqrt();
        assert!(
            r2 > r1,
            "Larger scale should produce larger distorted radius"
        );
    }

    #[test]
    fn test_stereographic_identity_at_origin() {
        let model = StereographicProjection::new(1.0);
        let origin = Vector2::new(0.0, 0.0);
        let distorted = model.distort(&origin);
        assert!(distorted[0].abs() < 1e-10);
        assert!(distorted[1].abs() < 1e-10);
    }

    #[test]
    fn test_stereographic_roundtrip() {
        let model = StereographicProjection::new(1.0);
        let point = Vector2::new(0.3, 0.4);
        let distorted = model.distort(&point);
        let recovered = model.undistort(&distorted);
        assert!((recovered[0] - point[0]).abs() < 1e-8);
        assert!((recovered[1] - point[1]).abs() < 1e-8);
    }

    #[test]
    fn test_extended_distortion_model_none() {
        let model = ExtendedDistortionModel::None;
        let p = Vector2::new(0.5, 0.5);
        assert_eq!(model.distort(&p), p);
        assert_eq!(model.undistort(&p), p);
    }

    #[test]
    fn test_extended_distortion_brown_conrady() {
        let bc = BrownConradyDistortion::new(0.1, 0.01, 0.0, 0.0, 0.0);
        let model = ExtendedDistortionModel::BrownConrady(bc.clone());
        let p = Vector2::new(0.3, 0.3);
        assert_eq!(model.distort(&p), bc.distort(&p));
    }

    #[test]
    fn test_extended_distortion_equidistant() {
        let eq = FisheyeEquidistant::new(1.0);
        let model = ExtendedDistortionModel::FisheyeEquidistant(eq.clone());
        let p = Vector2::new(0.2, 0.2);
        let d1 = model.distort(&p);
        let d2 = eq.distort(&p);
        assert!((d1[0] - d2[0]).abs() < 1e-12);
    }

    #[test]
    fn test_extended_distortion_stereographic() {
        let sg = StereographicProjection::new(1.0);
        let model = ExtendedDistortionModel::Stereographic(sg.clone());
        let p = Vector2::new(0.2, 0.3);
        let d1 = model.distort(&p);
        let d2 = sg.distort(&p);
        assert!((d1[0] - d2[0]).abs() < 1e-12);
    }

    #[test]
    fn test_stereographic_preserves_direction() {
        let model = StereographicProjection::new(1.0);
        let point = Vector2::new(1.0, 0.0);
        let distorted = model.distort(&point);
        // Should remain on the x-axis
        assert!(distorted[1].abs() < 1e-12);
        assert!(distorted[0] > 0.0);
    }

    // ── Original tests ────────────────────────────────────────────────────

    #[test]
    fn test_camera_intrinsics() {
        let intrinsics = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0);
        assert_eq!(intrinsics.fx, 1000.0);
        assert_eq!(intrinsics.fy, 1000.0);
        assert_eq!(intrinsics.cx, 640.0);
        assert_eq!(intrinsics.cy, 480.0);
    }

    #[test]
    fn test_pixel_to_normalized() {
        let intrinsics = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0);
        let pixel = Point2D::new(640.0, 480.0);
        let normalized = intrinsics.pixel_to_normalized(&pixel);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_brown_conrady_no_distortion() {
        let distortion = BrownConradyDistortion::default();
        let point = Vector2::new(0.5, 0.5);
        let distorted = distortion.distort(&point);
        assert!((distorted[0] - 0.5).abs() < 1e-10);
        assert!((distorted[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_brown_conrady_roundtrip() {
        let distortion = BrownConradyDistortion::new(0.1, 0.01, 0.001, 0.001, 0.001);
        let original = Vector2::new(0.5, 0.5);
        let distorted = distortion.distort(&original);
        let undistorted = distortion.undistort(&distorted);
        assert!((undistorted[0] - original[0]).abs() < 1e-6);
        assert!((undistorted[1] - original[1]).abs() < 1e-6);
    }

    #[test]
    fn test_fisheye_no_distortion() {
        // With zero k coefficients, the fisheye model applies the equidistant
        // projection (theta_d = theta = atan(r)), which is near-identity for small r.
        // The scale factor is atan(r)/r ≈ 1 - r^2/3 for small r.
        let distortion = FisheyeDistortion::default();
        let point = Vector2::new(0.1, 0.1);
        let distorted = distortion.distort(&point);
        // With r ≈ 0.1414, scale ≈ 0.9931, so distorted ≈ 0.0993 (not exact identity)
        assert!((distorted[0] - 0.1).abs() < 0.01);
        assert!((distorted[1] - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_camera_model() {
        let intrinsics = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0);
        let model = CameraModel::new(intrinsics, DistortionModel::None);

        let point_3d = nalgebra::Vector3::new(1.0, 1.0, 2.0);
        let pixel = model.project(&point_3d);

        // Should project to (640 + 500, 480 + 500) = (1140, 980)
        assert!((pixel.x - 1140.0).abs() < 1e-10);
        assert!((pixel.y - 980.0).abs() < 1e-10);
    }

    #[test]
    fn test_image_undistorter_creation() {
        let intrinsics = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0);
        let model = CameraModel::new(intrinsics, DistortionModel::None);
        let undistorter = ImageUndistorter::new(model, 1280, 960);

        assert_eq!(undistorter.width, 1280);
        assert_eq!(undistorter.height, 960);
        assert_eq!(undistorter.map_x.len(), 1280 * 960);
        assert_eq!(undistorter.map_y.len(), 1280 * 960);
    }
}
