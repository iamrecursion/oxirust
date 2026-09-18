//! Geometric transformations for image alignment.
//!
//! This module provides tools for applying geometric transformations:
//!
//! - Image warping
//! - Bilinear and bicubic interpolation
//! - Transformation composition
//! - Region-based transformations

use crate::spatial::{AffineTransform, Homography};
use crate::{AlignError, AlignResult, Point2D};
use nalgebra::Matrix3;

/// Interpolation method for image warping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Nearest neighbor (fastest)
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation (highest quality)
    Bicubic,
}

/// Image warper for applying transformations
pub struct ImageWarper {
    /// Interpolation method
    pub interpolation: InterpolationMethod,
    /// Border handling mode
    pub border_mode: BorderMode,
}

/// Border handling mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    /// Constant border (fill with specified value)
    Constant(u8),
    /// Replicate edge pixels
    Replicate,
    /// Reflect border
    Reflect,
    /// Wrap around
    Wrap,
}

impl Default for ImageWarper {
    fn default() -> Self {
        Self {
            interpolation: InterpolationMethod::Bilinear,
            border_mode: BorderMode::Constant(0),
        }
    }
}

impl ImageWarper {
    /// Create a new image warper
    #[must_use]
    pub fn new(interpolation: InterpolationMethod, border_mode: BorderMode) -> Self {
        Self {
            interpolation,
            border_mode,
        }
    }

    /// Warp image using homography
    ///
    /// # Errors
    /// Returns error if warping fails
    pub fn warp_homography(
        &self,
        input: &[u8],
        width: usize,
        height: usize,
        homography: &Homography,
        output_width: usize,
        output_height: usize,
    ) -> AlignResult<Vec<u8>> {
        if input.len() != width * height * 3 {
            return Err(AlignError::InvalidConfig("Input size mismatch".to_string()));
        }

        let mut output = vec![0u8; output_width * output_height * 3];

        // Invert homography for backward mapping
        let inv_h = homography.inverse()?;

        for y in 0..output_height {
            for x in 0..output_width {
                let dst = Point2D::new(x as f64, y as f64);
                let src = inv_h.transform(&dst);

                let pixel = self.sample_pixel(input, width, height, src.x as f32, src.y as f32);

                let idx = (y * output_width + x) * 3;
                output[idx..idx + 3].copy_from_slice(&pixel);
            }
        }

        Ok(output)
    }

    /// Warp image using affine transform
    ///
    /// # Errors
    /// Returns error if warping fails
    pub fn warp_affine(
        &self,
        input: &[u8],
        width: usize,
        height: usize,
        transform: &AffineTransform,
        output_width: usize,
        output_height: usize,
    ) -> AlignResult<Vec<u8>> {
        if input.len() != width * height * 3 {
            return Err(AlignError::InvalidConfig("Input size mismatch".to_string()));
        }

        let mut output = vec![0u8; output_width * output_height * 3];

        // Compute inverse transform for backward mapping
        let inv = self.invert_affine(transform)?;

        for y in 0..output_height {
            for x in 0..output_width {
                let dst = Point2D::new(x as f64, y as f64);
                let src = inv.transform(&dst);

                let pixel = self.sample_pixel(input, width, height, src.x as f32, src.y as f32);

                let idx = (y * output_width + x) * 3;
                output[idx..idx + 3].copy_from_slice(&pixel);
            }
        }

        Ok(output)
    }

    /// Sample pixel with interpolation
    fn sample_pixel(&self, image: &[u8], width: usize, height: usize, x: f32, y: f32) -> [u8; 3] {
        match self.interpolation {
            InterpolationMethod::Nearest => self.sample_nearest(image, width, height, x, y),
            InterpolationMethod::Bilinear => self.sample_bilinear(image, width, height, x, y),
            InterpolationMethod::Bicubic => self.sample_bicubic(image, width, height, x, y),
        }
    }

    /// Nearest neighbor sampling
    fn sample_nearest(&self, image: &[u8], width: usize, height: usize, x: f32, y: f32) -> [u8; 3] {
        let xi = x.round() as isize;
        let yi = y.round() as isize;

        if xi >= 0 && xi < width as isize && yi >= 0 && yi < height as isize {
            let idx = (yi as usize * width + xi as usize) * 3;
            if idx + 2 < image.len() {
                return [image[idx], image[idx + 1], image[idx + 2]];
            }
        }

        self.get_border_value()
    }

    /// Bilinear interpolation
    fn sample_bilinear(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        x: f32,
        y: f32,
    ) -> [u8; 3] {
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let p00 = self.get_pixel(image, width, height, x0, y0);
        let p10 = self.get_pixel(image, width, height, x1, y0);
        let p01 = self.get_pixel(image, width, height, x0, y1);
        let p11 = self.get_pixel(image, width, height, x1, y1);

        let mut result = [0u8; 3];
        for c in 0..3 {
            let v0 = f32::from(p00[c]) * (1.0 - dx) + f32::from(p10[c]) * dx;
            let v1 = f32::from(p01[c]) * (1.0 - dx) + f32::from(p11[c]) * dx;
            let v = v0 * (1.0 - dy) + v1 * dy;
            result[c] = v.round().clamp(0.0, 255.0) as u8;
        }

        result
    }

    /// Bicubic interpolation
    fn sample_bicubic(&self, image: &[u8], width: usize, height: usize, x: f32, y: f32) -> [u8; 3] {
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;

        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let mut result = [0u8; 3];

        for c in 0..3 {
            let mut value = 0.0f32;

            // Bicubic kernel is 4x4
            for j in -1..=2 {
                for i in -1..=2 {
                    let pixel = self.get_pixel(image, width, height, x0 + i, y0 + j);
                    let wx = Self::cubic_weight(i as f32 - dx);
                    let wy = Self::cubic_weight(j as f32 - dy);
                    value += f32::from(pixel[c]) * wx * wy;
                }
            }

            result[c] = value.round().clamp(0.0, 255.0) as u8;
        }

        result
    }

    /// Cubic interpolation weight (Mitchell-Netravali filter)
    fn cubic_weight(x: f32) -> f32 {
        let x = x.abs();
        if x < 1.0 {
            (1.5 * x - 2.5) * x * x + 1.0
        } else if x < 2.0 {
            ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
        } else {
            0.0
        }
    }

    /// Get pixel with border handling
    fn get_pixel(&self, image: &[u8], width: usize, height: usize, x: isize, y: isize) -> [u8; 3] {
        let (x_clamped, y_clamped) = self.apply_border_mode(x, y, width, height);

        if x_clamped >= 0
            && x_clamped < width as isize
            && y_clamped >= 0
            && y_clamped < height as isize
        {
            let idx = (y_clamped as usize * width + x_clamped as usize) * 3;
            if idx + 2 < image.len() {
                return [image[idx], image[idx + 1], image[idx + 2]];
            }
        }

        self.get_border_value()
    }

    /// Apply border mode
    fn apply_border_mode(&self, x: isize, y: isize, width: usize, height: usize) -> (isize, isize) {
        match self.border_mode {
            BorderMode::Constant(_) => (x, y),
            BorderMode::Replicate => (
                x.clamp(0, width as isize - 1),
                y.clamp(0, height as isize - 1),
            ),
            BorderMode::Reflect => (
                Self::reflect_coord(x, width),
                Self::reflect_coord(y, height),
            ),
            BorderMode::Wrap => (
                ((x % width as isize + width as isize) % width as isize),
                ((y % height as isize + height as isize) % height as isize),
            ),
        }
    }

    /// Reflect coordinate
    fn reflect_coord(x: isize, size: usize) -> isize {
        let size = size as isize;
        if x < 0 {
            -x - 1
        } else if x >= size {
            2 * size - x - 1
        } else {
            x
        }
    }

    /// Get border value
    fn get_border_value(&self) -> [u8; 3] {
        match self.border_mode {
            BorderMode::Constant(v) => [v, v, v],
            _ => [0, 0, 0],
        }
    }

    /// Invert affine transform
    fn invert_affine(&self, transform: &AffineTransform) -> AlignResult<AffineTransform> {
        let a = transform.matrix[(0, 0)];
        let b = transform.matrix[(0, 1)];
        let c = transform.matrix[(1, 0)];
        let d = transform.matrix[(1, 1)];
        let tx = transform.matrix[(0, 2)];
        let ty = transform.matrix[(1, 2)];

        let det = a * d - b * c;

        if det.abs() < 1e-10 {
            return Err(AlignError::NumericalError("Singular matrix".to_string()));
        }

        let inv_det = 1.0 / det;

        let inv_matrix = nalgebra::Matrix2x3::new(
            d * inv_det,
            -b * inv_det,
            (b * ty - d * tx) * inv_det,
            -c * inv_det,
            a * inv_det,
            (c * tx - a * ty) * inv_det,
        );

        Ok(AffineTransform::new(inv_matrix))
    }
}

/// Transformation builder for composing multiple transforms
pub struct TransformBuilder {
    /// Accumulated transformation matrix
    matrix: Matrix3<f64>,
}

impl Default for TransformBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformBuilder {
    /// Create a new transform builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            matrix: Matrix3::identity(),
        }
    }

    /// Add translation
    #[must_use]
    pub fn translate(mut self, tx: f64, ty: f64) -> Self {
        let t = Matrix3::new(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0);
        self.matrix = t * self.matrix;
        self
    }

    /// Add rotation (angle in radians)
    #[must_use]
    pub fn rotate(mut self, angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let r = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
        self.matrix = r * self.matrix;
        self
    }

    /// Add scale
    #[must_use]
    pub fn scale(mut self, sx: f64, sy: f64) -> Self {
        let s = Matrix3::new(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0);
        self.matrix = s * self.matrix;
        self
    }

    /// Add shear
    #[must_use]
    pub fn shear(mut self, shx: f64, shy: f64) -> Self {
        let sh = Matrix3::new(1.0, shx, 0.0, shy, 1.0, 0.0, 0.0, 0.0, 1.0);
        self.matrix = sh * self.matrix;
        self
    }

    /// Build homography
    #[must_use]
    pub fn build(self) -> Homography {
        Homography::new(self.matrix)
    }
}

/// Mesh warper for non-rigid transformations
pub struct MeshWarper {
    /// Grid width
    pub grid_width: usize,
    /// Grid height
    pub grid_height: usize,
    /// Control points
    control_points: Vec<Vec<Point2D>>,
}

impl MeshWarper {
    /// Create a new mesh warper
    #[must_use]
    pub fn new(grid_width: usize, grid_height: usize) -> Self {
        let mut control_points = Vec::new();

        for _y in 0..=grid_height {
            let mut row = Vec::new();
            for _x in 0..=grid_width {
                row.push(Point2D::new(0.0, 0.0));
            }
            control_points.push(row);
        }

        Self {
            grid_width,
            grid_height,
            control_points,
        }
    }

    /// Set control point
    pub fn set_control_point(&mut self, x: usize, y: usize, point: Point2D) {
        if y < self.control_points.len() && x < self.control_points[y].len() {
            self.control_points[y][x] = point;
        }
    }

    /// Initialize regular grid
    pub fn init_regular_grid(&mut self, width: usize, height: usize) {
        let dx = width as f64 / self.grid_width as f64;
        let dy = height as f64 / self.grid_height as f64;

        for y in 0..=self.grid_height {
            for x in 0..=self.grid_width {
                self.control_points[y][x] = Point2D::new(x as f64 * dx, y as f64 * dy);
            }
        }
    }

    /// Warp image using mesh
    ///
    /// # Errors
    /// Returns error if warping fails
    pub fn warp(&self, input: &[u8], width: usize, height: usize) -> AlignResult<Vec<u8>> {
        if input.len() != width * height * 3 {
            return Err(AlignError::InvalidConfig("Input size mismatch".to_string()));
        }

        let mut output = vec![0u8; width * height * 3];
        let warper = ImageWarper::default();

        let dx = width as f64 / self.grid_width as f64;
        let dy = height as f64 / self.grid_height as f64;

        for y in 0..height {
            for x in 0..width {
                // Find grid cell
                let gx = (x as f64 / dx).floor() as usize;
                let gy = (y as f64 / dy).floor() as usize;

                if gx < self.grid_width && gy < self.grid_height {
                    // Bilinear interpolation within cell
                    let tx = (x as f64 - gx as f64 * dx) / dx;
                    let ty = (y as f64 - gy as f64 * dy) / dy;

                    let p00 = &self.control_points[gy][gx];
                    let p10 = &self.control_points[gy][gx + 1];
                    let p01 = &self.control_points[gy + 1][gx];
                    let p11 = &self.control_points[gy + 1][gx + 1];

                    let src_x = p00.x * (1.0 - tx) * (1.0 - ty)
                        + p10.x * tx * (1.0 - ty)
                        + p01.x * (1.0 - tx) * ty
                        + p11.x * tx * ty;

                    let src_y = p00.y * (1.0 - tx) * (1.0 - ty)
                        + p10.y * tx * (1.0 - ty)
                        + p01.y * (1.0 - tx) * ty
                        + p11.y * tx * ty;

                    let pixel =
                        warper.sample_pixel(input, width, height, src_x as f32, src_y as f32);

                    let idx = (y * width + x) * 3;
                    output[idx..idx + 3].copy_from_slice(&pixel);
                }
            }
        }

        Ok(output)
    }
}

/// Perspective quad warper
pub struct QuadWarper;

impl QuadWarper {
    /// Warp a quadrilateral region to rectangle
    ///
    /// # Errors
    /// Returns error if warping fails
    pub fn warp_quad(
        input: &[u8],
        width: usize,
        height: usize,
        src_quad: &[Point2D; 4],
        dst_width: usize,
        dst_height: usize,
    ) -> AlignResult<Vec<u8>> {
        // Build homography from quad to rectangle
        let dst_quad = [
            Point2D::new(0.0, 0.0),
            Point2D::new(dst_width as f64, 0.0),
            Point2D::new(dst_width as f64, dst_height as f64),
            Point2D::new(0.0, dst_height as f64),
        ];

        let homography = Self::compute_quad_to_quad_homography(src_quad, &dst_quad)?;

        let warper = ImageWarper::default();
        warper.warp_homography(input, width, height, &homography, dst_width, dst_height)
    }

    /// Compute homography from quad to quad
    fn compute_quad_to_quad_homography(
        src: &[Point2D; 4],
        dst: &[Point2D; 4],
    ) -> AlignResult<Homography> {
        // Build system of equations for DLT
        let mut a = nalgebra::DMatrix::zeros(8, 9);

        for i in 0..4 {
            let x = src[i].x;
            let y = src[i].y;
            let xp = dst[i].x;
            let yp = dst[i].y;

            a[(i * 2, 0)] = -x;
            a[(i * 2, 1)] = -y;
            a[(i * 2, 2)] = -1.0;
            a[(i * 2, 6)] = xp * x;
            a[(i * 2, 7)] = xp * y;
            a[(i * 2, 8)] = xp;

            a[(i * 2 + 1, 3)] = -x;
            a[(i * 2 + 1, 4)] = -y;
            a[(i * 2 + 1, 5)] = -1.0;
            a[(i * 2 + 1, 6)] = yp * x;
            a[(i * 2 + 1, 7)] = yp * y;
            a[(i * 2 + 1, 8)] = yp;
        }

        let svd = a.svd(true, true);
        let v = svd
            .v_t
            .ok_or_else(|| AlignError::NumericalError("SVD failed".to_string()))?;

        let h_vec = v.row(8);

        if h_vec[8].abs() < 1e-10 {
            return Err(AlignError::NumericalError(
                "Degenerate solution".to_string(),
            ));
        }

        let scale = h_vec[8];
        let matrix = Matrix3::new(
            h_vec[0] / scale,
            h_vec[1] / scale,
            h_vec[2] / scale,
            h_vec[3] / scale,
            h_vec[4] / scale,
            h_vec[5] / scale,
            h_vec[6] / scale,
            h_vec[7] / scale,
            1.0,
        );

        Ok(Homography::new(matrix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_method() {
        assert_eq!(InterpolationMethod::Nearest, InterpolationMethod::Nearest);
        assert_ne!(InterpolationMethod::Nearest, InterpolationMethod::Bilinear);
    }

    #[test]
    fn test_border_mode() {
        let mode = BorderMode::Constant(128);
        match mode {
            BorderMode::Constant(v) => assert_eq!(v, 128),
            _ => panic!("Wrong border mode"),
        }
    }

    #[test]
    fn test_image_warper_creation() {
        let warper = ImageWarper::default();
        assert_eq!(warper.interpolation, InterpolationMethod::Bilinear);
    }

    #[test]
    fn test_cubic_weight() {
        let w = ImageWarper::cubic_weight(0.0);
        assert!((w - 1.0).abs() < 1e-6);

        let w = ImageWarper::cubic_weight(2.0);
        assert!(w.abs() < 1e-6);
    }

    #[test]
    fn test_transform_builder() {
        let transform = TransformBuilder::new()
            .translate(10.0, 20.0)
            .rotate(std::f64::consts::PI / 4.0)
            .scale(2.0, 2.0)
            .build();

        let point = Point2D::new(0.0, 0.0);
        let transformed = transform.transform(&point);
        assert!(transformed.x.is_finite());
        assert!(transformed.y.is_finite());
    }

    #[test]
    fn test_mesh_warper_creation() {
        let warper = MeshWarper::new(10, 10);
        assert_eq!(warper.grid_width, 10);
        assert_eq!(warper.grid_height, 10);
    }

    #[test]
    fn test_mesh_warper_control_points() {
        let mut warper = MeshWarper::new(2, 2);
        warper.set_control_point(1, 1, Point2D::new(100.0, 100.0));
        assert_eq!(warper.control_points[1][1].x, 100.0);
        assert_eq!(warper.control_points[1][1].y, 100.0);
    }

    #[test]
    fn test_mesh_warper_regular_grid() {
        let mut warper = MeshWarper::new(4, 4);
        warper.init_regular_grid(400, 400);
        assert_eq!(warper.control_points[0][0].x, 0.0);
        assert_eq!(warper.control_points[4][4].x, 400.0);
        assert_eq!(warper.control_points[4][4].y, 400.0);
    }
}
