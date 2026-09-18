//! Lens distortion compensation
//!
//! Implements Brown-Conrady (radial + tangential) and fisheye (equidistant)
//! distortion models used in virtual production camera tracking pipelines.
//! These models correct the optical distortion inherent in real camera lenses
//! so that LED wall content can be rendered with matching perspective.

use super::LensParameters;

/// Distortion model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DistortionModel {
    /// Brown-Conrady model: radial (k1..k6) + tangential (p1, p2)
    /// This is the standard OpenCV/photogrammetry model.
    BrownConrady,
    /// Equidistant fisheye model: r_d = theta * (1 + k1*theta^2 + k2*theta^4 + ...)
    /// Used for ultra-wide-angle lenses (>120 degree FOV).
    Fisheye,
    /// Division model: a simpler 1-parameter model for quick undistortion.
    /// x_u = x_d / (1 + k1 * r_d^2)
    Division,
}

/// Extended lens distortion parameters beyond the basic `LensParameters`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistortionParams {
    /// Distortion model to use
    pub model: DistortionModel,
    /// Radial distortion coefficients k1..k6.
    /// Brown-Conrady uses up to 6; fisheye typically uses 4.
    pub radial: Vec<f64>,
    /// Tangential distortion coefficients (p1, p2).
    /// Only used by Brown-Conrady model.
    pub tangential: [f64; 2],
    /// Principal point offset from image center (cx, cy) in normalised coords.
    /// (0, 0) means the optical axis passes through the image center.
    pub principal_point: [f64; 2],
    /// Focal length in pixels (fx, fy). Used for pixel <-> normalised conversion.
    pub focal_px: [f64; 2],
    /// Image dimensions (width, height) in pixels.
    pub image_size: [usize; 2],
}

impl DistortionParams {
    /// Create Brown-Conrady parameters from basic lens parameters.
    ///
    /// Maps `LensParameters` radial/tangential coefficients into a full
    /// `DistortionParams` suitable for pixel-level correction.
    #[must_use]
    pub fn from_lens_params(
        params: &LensParameters,
        image_width: usize,
        image_height: usize,
    ) -> Self {
        // Compute focal length in pixels from mm and sensor size
        let fx = params.focal_length * (image_width as f64) / params.sensor_width;
        let fy = params.focal_length * (image_height as f64) / params.sensor_height;

        Self {
            model: DistortionModel::BrownConrady,
            radial: params.radial_distortion.clone(),
            tangential: if params.tangential_distortion.len() >= 2 {
                [
                    params.tangential_distortion[0],
                    params.tangential_distortion[1],
                ]
            } else {
                [0.0, 0.0]
            },
            principal_point: [0.0, 0.0],
            focal_px: [fx, fy],
            image_size: [image_width, image_height],
        }
    }

    /// Create a zero-distortion (identity) set of parameters.
    #[must_use]
    pub fn identity(image_width: usize, image_height: usize) -> Self {
        let fx = image_width as f64;
        let fy = image_height as f64;
        Self {
            model: DistortionModel::BrownConrady,
            radial: vec![0.0; 3],
            tangential: [0.0, 0.0],
            principal_point: [0.0, 0.0],
            focal_px: [fx, fy],
            image_size: [image_width, image_height],
        }
    }

    /// Create a fisheye distortion parameter set.
    #[must_use]
    pub fn fisheye(
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
        fx: f64,
        fy: f64,
        image_width: usize,
        image_height: usize,
    ) -> Self {
        Self {
            model: DistortionModel::Fisheye,
            radial: vec![k1, k2, k3, k4],
            tangential: [0.0, 0.0], // not used in fisheye
            principal_point: [0.0, 0.0],
            focal_px: [fx, fy],
            image_size: [image_width, image_height],
        }
    }

    /// Create a Division model parameter set.
    #[must_use]
    pub fn division(k1: f64, fx: f64, fy: f64, image_width: usize, image_height: usize) -> Self {
        Self {
            model: DistortionModel::Division,
            radial: vec![k1],
            tangential: [0.0, 0.0],
            principal_point: [0.0, 0.0],
            focal_px: [fx, fy],
            image_size: [image_width, image_height],
        }
    }
}

/// Distortion corrector that can undistort and redistort pixel coordinates.
///
/// Supports Brown-Conrady, fisheye (equidistant), and division models.
/// All operations work in normalised image coordinates centered at the
/// principal point, then convert back to pixel space.
pub struct DistortionCorrector {
    params: DistortionParams,
    /// Pre-computed half-image-size for normalisation
    half_w: f64,
    half_h: f64,
}

impl DistortionCorrector {
    /// Create new distortion corrector from basic lens parameters.
    #[must_use]
    pub fn new(params: LensParameters) -> Self {
        let dp = DistortionParams::from_lens_params(&params, 1920, 1080);
        Self::from_distortion_params(dp)
    }

    /// Create from full distortion parameters.
    #[must_use]
    pub fn from_distortion_params(params: DistortionParams) -> Self {
        let half_w = params.image_size[0] as f64 * 0.5;
        let half_h = params.image_size[1] as f64 * 0.5;
        Self {
            params,
            half_w,
            half_h,
        }
    }

    /// Correct (undistort) pixel coordinates.
    ///
    /// Takes a distorted pixel position `(px, py)` and returns the
    /// corresponding undistorted pixel position.
    #[must_use]
    pub fn correct(&self, px: f64, py: f64) -> (f64, f64) {
        match self.params.model {
            DistortionModel::BrownConrady => self.undistort_brown_conrady(px, py),
            DistortionModel::Fisheye => self.undistort_fisheye(px, py),
            DistortionModel::Division => self.undistort_division(px, py),
        }
    }

    /// Apply distortion to undistorted pixel coordinates (forward model).
    ///
    /// Takes an ideal (undistorted) pixel position and returns the
    /// position where that point actually appears in the distorted image.
    #[must_use]
    pub fn distort(&self, px: f64, py: f64) -> (f64, f64) {
        match self.params.model {
            DistortionModel::BrownConrady => self.distort_brown_conrady(px, py),
            DistortionModel::Fisheye => self.distort_fisheye(px, py),
            DistortionModel::Division => self.distort_division(px, py),
        }
    }

    /// Get the underlying parameters.
    #[must_use]
    pub fn params(&self) -> &DistortionParams {
        &self.params
    }

    // -----------------------------------------------------------------------
    // Pixel <-> normalised coordinate conversion
    // -----------------------------------------------------------------------

    /// Convert pixel coords to normalised coords centered at principal point.
    fn pixel_to_norm(&self, px: f64, py: f64) -> (f64, f64) {
        let cx = self.half_w + self.params.principal_point[0] * self.half_w;
        let cy = self.half_h + self.params.principal_point[1] * self.half_h;
        let x = (px - cx) / self.params.focal_px[0];
        let y = (py - cy) / self.params.focal_px[1];
        (x, y)
    }

    /// Convert normalised coords back to pixel coords.
    fn norm_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        let cx = self.half_w + self.params.principal_point[0] * self.half_w;
        let cy = self.half_h + self.params.principal_point[1] * self.half_h;
        let px = x * self.params.focal_px[0] + cx;
        let py = y * self.params.focal_px[1] + cy;
        (px, py)
    }

    // -----------------------------------------------------------------------
    // Brown-Conrady model
    // -----------------------------------------------------------------------

    /// Forward Brown-Conrady distortion: undistorted -> distorted (normalised).
    ///
    /// Standard model:
    ///   r^2 = x^2 + y^2
    ///   radial = (1 + k1*r^2 + k2*r^4 + k3*r^6) / (1 + k4*r^2 + k5*r^4 + k6*r^6)
    ///   x_d = x * radial + 2*p1*x*y + p2*(r^2 + 2*x^2)
    ///   y_d = y * radial + p1*(r^2 + 2*y^2) + 2*p2*x*y
    fn brown_conrady_forward(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let k = &self.params.radial;
        let k1 = k.first().copied().unwrap_or(0.0);
        let k2 = k.get(1).copied().unwrap_or(0.0);
        let k3 = k.get(2).copied().unwrap_or(0.0);
        let k4 = k.get(3).copied().unwrap_or(0.0);
        let k5 = k.get(4).copied().unwrap_or(0.0);
        let k6 = k.get(5).copied().unwrap_or(0.0);

        let numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
        let radial = if denominator.abs() > 1e-15 {
            numerator / denominator
        } else {
            numerator
        };

        let p1 = self.params.tangential[0];
        let p2 = self.params.tangential[1];

        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        (xd, yd)
    }

    /// Undistort Brown-Conrady: distorted pixel -> undistorted pixel.
    ///
    /// Uses iterative Newton-Raphson to invert the forward distortion model.
    fn undistort_brown_conrady(&self, px: f64, py: f64) -> (f64, f64) {
        let (xd, yd) = self.pixel_to_norm(px, py);

        // Iterative refinement: start from distorted point as initial guess
        let mut xu = xd;
        let mut yu = yd;
        let max_iterations = 20;
        let tolerance = 1e-12;

        for _ in 0..max_iterations {
            let (xd_est, yd_est) = self.brown_conrady_forward(xu, yu);
            let dx = xd - xd_est;
            let dy = yd - yd_est;

            if dx * dx + dy * dy < tolerance {
                break;
            }

            xu += dx;
            yu += dy;
        }

        self.norm_to_pixel(xu, yu)
    }

    /// Distort Brown-Conrady: undistorted pixel -> distorted pixel.
    fn distort_brown_conrady(&self, px: f64, py: f64) -> (f64, f64) {
        let (xu, yu) = self.pixel_to_norm(px, py);
        let (xd, yd) = self.brown_conrady_forward(xu, yu);
        self.norm_to_pixel(xd, yd)
    }

    // -----------------------------------------------------------------------
    // Fisheye (equidistant) model
    // -----------------------------------------------------------------------

    /// Undistort fisheye: distorted pixel -> undistorted pixel.
    ///
    /// Equidistant fisheye: r_d = f * theta * (1 + k1*theta^2 + k2*theta^4 + ...)
    /// We invert theta from r_d, then compute the pinhole projection.
    fn undistort_fisheye(&self, px: f64, py: f64) -> (f64, f64) {
        let (xd, yd) = self.pixel_to_norm(px, py);
        let rd = (xd * xd + yd * yd).sqrt();

        if rd < 1e-15 {
            return (px, py);
        }

        // theta_d = rd (already normalised by focal length)
        // Solve: rd = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        // Using Newton-Raphson
        let k = &self.params.radial;
        let k1 = k.first().copied().unwrap_or(0.0);
        let k2 = k.get(1).copied().unwrap_or(0.0);
        let k3 = k.get(2).copied().unwrap_or(0.0);
        let k4 = k.get(3).copied().unwrap_or(0.0);

        let mut theta = rd; // initial guess
        let max_iterations = 20;
        let tolerance = 1e-12;

        for _ in 0..max_iterations {
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;

            let f_theta =
                theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - rd;
            let f_prime =
                1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8;

            if f_prime.abs() < 1e-15 {
                break;
            }

            let delta = f_theta / f_prime;
            theta -= delta;

            if delta.abs() < tolerance {
                break;
            }
        }

        // Convert from fisheye angle back to pinhole projection
        let tan_theta = theta.tan();
        if !tan_theta.is_finite() {
            return (px, py);
        }
        let scale = if rd > 1e-15 { tan_theta / rd } else { 1.0 };
        let xu = xd * scale;
        let yu = yd * scale;

        self.norm_to_pixel(xu, yu)
    }

    /// Distort fisheye: undistorted pixel -> distorted pixel.
    fn distort_fisheye(&self, px: f64, py: f64) -> (f64, f64) {
        let (xu, yu) = self.pixel_to_norm(px, py);
        let ru = (xu * xu + yu * yu).sqrt();

        if ru < 1e-15 {
            return (px, py);
        }

        // theta = atan(ru) for pinhole -> fisheye angle conversion
        let theta = ru.atan();
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;

        let k = &self.params.radial;
        let k1 = k.first().copied().unwrap_or(0.0);
        let k2 = k.get(1).copied().unwrap_or(0.0);
        let k3 = k.get(2).copied().unwrap_or(0.0);
        let k4 = k.get(3).copied().unwrap_or(0.0);

        let rd = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        let scale = if ru > 1e-15 { rd / ru } else { 1.0 };

        let xd = xu * scale;
        let yd = yu * scale;

        self.norm_to_pixel(xd, yd)
    }

    // -----------------------------------------------------------------------
    // Division model
    // -----------------------------------------------------------------------

    /// Undistort division model: distorted pixel -> undistorted pixel.
    ///
    /// x_u = x_d / (1 + k1 * r_d^2)
    fn undistort_division(&self, px: f64, py: f64) -> (f64, f64) {
        let (xd, yd) = self.pixel_to_norm(px, py);
        let rd2 = xd * xd + yd * yd;
        let k1 = self.params.radial.first().copied().unwrap_or(0.0);
        let denom = 1.0 + k1 * rd2;
        if denom.abs() < 1e-15 {
            return (px, py);
        }
        let xu = xd / denom;
        let yu = yd / denom;
        self.norm_to_pixel(xu, yu)
    }

    /// Distort division model: undistorted pixel -> distorted pixel.
    ///
    /// Inverts x_u = x_d / (1 + k1*r_d^2) iteratively.
    fn distort_division(&self, px: f64, py: f64) -> (f64, f64) {
        let (xu, yu) = self.pixel_to_norm(px, py);
        let k1 = self.params.radial.first().copied().unwrap_or(0.0);

        // Iterative: start from undistorted as guess for distorted
        let mut xd = xu;
        let mut yd = yu;
        let max_iterations = 20;
        let tolerance = 1e-12;

        for _ in 0..max_iterations {
            let rd2 = xd * xd + yd * yd;
            let denom = 1.0 + k1 * rd2;
            if denom.abs() < 1e-15 {
                break;
            }
            let xu_est = xd / denom;
            let yu_est = yd / denom;
            let dx = xu - xu_est;
            let dy = yu - yu_est;

            if dx * dx + dy * dy < tolerance {
                break;
            }

            xd += dx;
            yd += dy;
        }

        self.norm_to_pixel(xd, yd)
    }

    /// Build an undistortion lookup table for the full image.
    ///
    /// Returns a Vec of (src_x, src_y) pairs in pixel coords, one per output pixel,
    /// in row-major order. To remap an image, sample the input at the
    /// returned coordinates for each output pixel.
    #[must_use]
    pub fn build_undistort_map(&self) -> Vec<(f64, f64)> {
        let w = self.params.image_size[0];
        let h = self.params.image_size[1];
        let mut map = Vec::with_capacity(w * h);

        for row in 0..h {
            for col in 0..w {
                // For each output (undistorted) pixel, find where it maps in distorted image
                let (src_x, src_y) = self.distort(col as f64, row as f64);
                map.push((src_x, src_y));
            }
        }

        map
    }

    /// Compute the maximum distortion magnitude across a grid of sample points.
    ///
    /// Returns the maximum pixel displacement between distorted and undistorted
    /// coordinates, useful for evaluating lens quality.
    #[must_use]
    pub fn max_distortion(&self) -> f64 {
        let w = self.params.image_size[0];
        let h = self.params.image_size[1];
        let step = 32;
        let mut max_dist = 0.0_f64;

        let mut row = 0;
        while row < h {
            let mut col = 0;
            while col < w {
                let (ux, uy) = self.correct(col as f64, row as f64);
                let dx = ux - col as f64;
                let dy = uy - row as f64;
                let dist = (dx * dx + dy * dy).sqrt();
                max_dist = max_dist.max(dist);
                col += step;
            }
            row += step;
        }

        max_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brown_conrady_zero_distortion_is_identity() {
        let params = DistortionParams::identity(1920, 1080);
        let corrector = DistortionCorrector::from_distortion_params(params);
        let (ux, uy) = corrector.correct(960.0, 540.0);
        assert!((ux - 960.0).abs() < 1e-6, "center should be unchanged");
        assert!((uy - 540.0).abs() < 1e-6, "center should be unchanged");

        // Off-center point
        let (ux2, uy2) = corrector.correct(100.0, 200.0);
        assert!((ux2 - 100.0).abs() < 1e-6);
        assert!((uy2 - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_brown_conrady_roundtrip() {
        let params = DistortionParams {
            model: DistortionModel::BrownConrady,
            radial: vec![-0.28, 0.09, -0.01],
            tangential: [0.001, -0.0005],
            principal_point: [0.0, 0.0],
            focal_px: [1000.0, 1000.0],
            image_size: [1920, 1080],
        };
        let corrector = DistortionCorrector::from_distortion_params(params);

        // Start with an undistorted point, distort it, then undistort
        let px = 800.0;
        let py = 400.0;
        let (dx, dy) = corrector.distort(px, py);
        let (ux, uy) = corrector.correct(dx, dy);

        assert!((ux - px).abs() < 0.01, "roundtrip x failed: {ux} vs {px}");
        assert!((uy - py).abs() < 0.01, "roundtrip y failed: {uy} vs {py}");
    }

    #[test]
    fn test_brown_conrady_barrel_distortion() {
        // Negative k1 = barrel distortion: points move toward center
        let params = DistortionParams {
            model: DistortionModel::BrownConrady,
            radial: vec![-0.3, 0.0, 0.0],
            tangential: [0.0, 0.0],
            principal_point: [0.0, 0.0],
            focal_px: [1000.0, 1000.0],
            image_size: [1920, 1080],
        };
        let corrector = DistortionCorrector::from_distortion_params(params);

        // A corner point should be pushed inward by barrel distortion
        let (dx, dy) = corrector.distort(1600.0, 800.0);
        let dist_orig = ((1600.0 - 960.0_f64).powi(2) + (800.0 - 540.0_f64).powi(2)).sqrt();
        let dist_distorted = ((dx - 960.0).powi(2) + (dy - 540.0).powi(2)).sqrt();
        assert!(
            dist_distorted < dist_orig,
            "barrel distortion should move points inward: {dist_distorted} >= {dist_orig}"
        );
    }

    #[test]
    fn test_brown_conrady_pincushion_distortion() {
        // Positive k1 = pincushion distortion: points move away from center
        let params = DistortionParams {
            model: DistortionModel::BrownConrady,
            radial: vec![0.3, 0.0, 0.0],
            tangential: [0.0, 0.0],
            principal_point: [0.0, 0.0],
            focal_px: [1000.0, 1000.0],
            image_size: [1920, 1080],
        };
        let corrector = DistortionCorrector::from_distortion_params(params);

        let (dx, dy) = corrector.distort(1200.0, 700.0);
        let dist_orig = ((1200.0 - 960.0_f64).powi(2) + (700.0 - 540.0_f64).powi(2)).sqrt();
        let dist_distorted = ((dx - 960.0).powi(2) + (dy - 540.0).powi(2)).sqrt();
        assert!(
            dist_distorted > dist_orig,
            "pincushion should move points outward"
        );
    }

    #[test]
    fn test_brown_conrady_tangential_effect() {
        // Tangential distortion should shift points asymmetrically
        let params_no_tang = DistortionParams {
            model: DistortionModel::BrownConrady,
            radial: vec![-0.1, 0.0, 0.0],
            tangential: [0.0, 0.0],
            principal_point: [0.0, 0.0],
            focal_px: [1000.0, 1000.0],
            image_size: [1920, 1080],
        };
        let params_with_tang = DistortionParams {
            model: DistortionModel::BrownConrady,
            radial: vec![-0.1, 0.0, 0.0],
            tangential: [0.005, 0.003],
            principal_point: [0.0, 0.0],
            focal_px: [1000.0, 1000.0],
            image_size: [1920, 1080],
        };

        let c1 = DistortionCorrector::from_distortion_params(params_no_tang);
        let c2 = DistortionCorrector::from_distortion_params(params_with_tang);

        let (x1, y1) = c1.distort(1200.0, 700.0);
        let (x2, y2) = c2.distort(1200.0, 700.0);

        // Tangential should cause a measurable shift
        let shift = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        assert!(
            shift > 0.1,
            "tangential distortion should cause shift: {shift}"
        );
    }

    #[test]
    fn test_fisheye_zero_distortion_roundtrip() {
        let params = DistortionParams::fisheye(0.0, 0.0, 0.0, 0.0, 500.0, 500.0, 1024, 1024);
        let corrector = DistortionCorrector::from_distortion_params(params);

        let px = 600.0;
        let py = 400.0;
        let (dx, dy) = corrector.distort(px, py);
        let (ux, uy) = corrector.correct(dx, dy);

        assert!((ux - px).abs() < 0.1, "fisheye roundtrip x: {ux} vs {px}");
        assert!((uy - py).abs() < 0.1, "fisheye roundtrip y: {uy} vs {py}");
    }

    #[test]
    fn test_fisheye_with_distortion_roundtrip() {
        let params =
            DistortionParams::fisheye(0.05, -0.01, 0.002, -0.001, 500.0, 500.0, 1024, 1024);
        let corrector = DistortionCorrector::from_distortion_params(params);

        let px = 600.0;
        let py = 400.0;
        let (dx, dy) = corrector.distort(px, py);
        let (ux, uy) = corrector.correct(dx, dy);

        assert!((ux - px).abs() < 0.5, "fisheye roundtrip x: {ux} vs {px}");
        assert!((uy - py).abs() < 0.5, "fisheye roundtrip y: {uy} vs {py}");
    }

    #[test]
    fn test_fisheye_center_unchanged() {
        let params = DistortionParams::fisheye(0.1, -0.05, 0.01, 0.0, 500.0, 500.0, 1024, 1024);
        let corrector = DistortionCorrector::from_distortion_params(params);
        let (ux, uy) = corrector.correct(512.0, 512.0);
        assert!((ux - 512.0).abs() < 1e-6, "center should stay fixed");
        assert!((uy - 512.0).abs() < 1e-6, "center should stay fixed");
    }

    #[test]
    fn test_division_model_zero_distortion() {
        let params = DistortionParams::division(0.0, 1000.0, 1000.0, 1920, 1080);
        let corrector = DistortionCorrector::from_distortion_params(params);

        let (ux, uy) = corrector.correct(800.0, 300.0);
        assert!((ux - 800.0).abs() < 1e-6);
        assert!((uy - 300.0).abs() < 1e-6);
    }

    #[test]
    fn test_division_model_roundtrip() {
        let params = DistortionParams::division(-0.2, 1000.0, 1000.0, 1920, 1080);
        let corrector = DistortionCorrector::from_distortion_params(params);

        let px = 1200.0;
        let py = 700.0;
        let (dx, dy) = corrector.distort(px, py);
        let (ux, uy) = corrector.correct(dx, dy);

        assert!((ux - px).abs() < 0.1, "division roundtrip x: {ux} vs {px}");
        assert!((uy - py).abs() < 0.1, "division roundtrip y: {uy} vs {py}");
    }

    #[test]
    fn test_max_distortion_zero_for_identity() {
        let params = DistortionParams::identity(640, 480);
        let corrector = DistortionCorrector::from_distortion_params(params);
        let max_d = corrector.max_distortion();
        assert!(
            max_d < 1e-6,
            "identity should have zero distortion: {max_d}"
        );
    }

    #[test]
    fn test_max_distortion_nonzero_for_barrel() {
        let params = DistortionParams {
            model: DistortionModel::BrownConrady,
            radial: vec![-0.3, 0.1, 0.0],
            tangential: [0.0, 0.0],
            principal_point: [0.0, 0.0],
            focal_px: [500.0, 500.0],
            image_size: [640, 480],
        };
        let corrector = DistortionCorrector::from_distortion_params(params);
        let max_d = corrector.max_distortion();
        assert!(
            max_d > 1.0,
            "barrel distortion should have measurable displacement: {max_d}"
        );
    }

    #[test]
    fn test_build_undistort_map_size() {
        let params = DistortionParams::identity(64, 48);
        let corrector = DistortionCorrector::from_distortion_params(params);
        let map = corrector.build_undistort_map();
        assert_eq!(map.len(), 64 * 48);
    }

    #[test]
    fn test_from_lens_params() {
        let lens = LensParameters::new(50.0, 36.0, 24.0);
        let dp = DistortionParams::from_lens_params(&lens, 1920, 1080);
        assert_eq!(dp.model, DistortionModel::BrownConrady);
        // fx = 50 * 1920 / 36 = 2666.67
        let expected_fx = 50.0 * 1920.0 / 36.0;
        assert!((dp.focal_px[0] - expected_fx).abs() < 0.01);
    }

    #[test]
    fn test_principal_point_offset() {
        // With principal point shifted, center pixel should not map to itself
        let params = DistortionParams {
            model: DistortionModel::BrownConrady,
            radial: vec![0.0],
            tangential: [0.0, 0.0],
            principal_point: [0.05, -0.03], // 5% right, 3% up
            focal_px: [1000.0, 1000.0],
            image_size: [1920, 1080],
        };
        let corrector = DistortionCorrector::from_distortion_params(params);
        // The optical center is at (960 + 0.05*960, 540 - 0.03*540)
        // Points at the optical center should be identity
        let cx = 960.0 + 0.05 * 960.0;
        let cy = 540.0 + (-0.03) * 540.0;
        let (ux, uy) = corrector.correct(cx, cy);
        assert!((ux - cx).abs() < 1e-6, "optical center should be fixed");
        assert!((uy - cy).abs() < 1e-6, "optical center should be fixed");
    }

    #[test]
    fn test_rational_model_k4_k5_k6() {
        // Test the rational polynomial extension (denominator coefficients)
        let params = DistortionParams {
            model: DistortionModel::BrownConrady,
            radial: vec![-0.2, 0.05, -0.01, 0.1, -0.02, 0.005],
            tangential: [0.0, 0.0],
            principal_point: [0.0, 0.0],
            focal_px: [1000.0, 1000.0],
            image_size: [1920, 1080],
        };
        let corrector = DistortionCorrector::from_distortion_params(params);

        let px = 1100.0;
        let py = 650.0;
        let (dx, dy) = corrector.distort(px, py);
        let (ux, uy) = corrector.correct(dx, dy);
        assert!(
            (ux - px).abs() < 0.1,
            "rational model roundtrip x: {ux} vs {px}"
        );
        assert!(
            (uy - py).abs() < 0.1,
            "rational model roundtrip y: {uy} vs {py}"
        );
    }
}
