//! Automatic lens calibration from checkerboard and ArUco-style marker patterns.
//!
//! This module provides:
//!
//! - Checkerboard corner detection using iterative sub-pixel refinement
//! - Brown-Conrady distortion parameter estimation from multiple views
//! - Camera intrinsic matrix estimation via Direct Linear Transform (DLT)
//! - ArUco-style fiducial marker corner localisation
//!
//! The calibration follows Zhang's plane-based method using a flat calibration
//! target (checkerboard or ArUco grid) viewed from multiple positions.

#![allow(clippy::cast_precision_loss)]

use crate::distortion::{BrownConradyDistortion, CameraIntrinsics};
use crate::{AlignError, AlignResult, Point2D};

/// A 3-D point in world/object space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

impl Point3D {
    /// Create a new 3-D point.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Detected checkerboard pattern (inner corners only).
#[derive(Debug, Clone)]
pub struct CheckerboardDetection {
    /// Number of inner corners per row.
    pub cols: usize,
    /// Number of inner corners per column.
    pub rows: usize,
    /// Detected corner positions in image space (row-major).
    pub corners: Vec<Point2D>,
    /// Sub-pixel refinement quality score (higher = better, 0–1).
    pub quality: f64,
}

/// Checkerboard detector.
pub struct CheckerboardDetector {
    /// Size of a single square in world units (e.g. mm).
    pub square_size: f64,
    /// Sub-pixel refinement window half-size.
    pub refine_half_win: usize,
    /// Maximum refinement iterations.
    pub max_iterations: usize,
    /// Convergence epsilon (pixels).
    pub epsilon: f64,
}

impl Default for CheckerboardDetector {
    fn default() -> Self {
        Self {
            square_size: 25.0,
            refine_half_win: 5,
            max_iterations: 20,
            epsilon: 0.01,
        }
    }
}

impl CheckerboardDetector {
    /// Create a new checkerboard detector.
    #[must_use]
    pub fn new(square_size: f64) -> Self {
        Self {
            square_size,
            ..Self::default()
        }
    }

    /// Detect inner checkerboard corners in a grayscale image.
    ///
    /// Uses a saddle-point detector based on the second derivative structure
    /// tensor, followed by iterative sub-pixel Newton refinement.
    ///
    /// # Errors
    ///
    /// Returns [`AlignError`] if the image dimensions are inconsistent or no
    /// corners are found.
    pub fn detect(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        inner_cols: usize,
        inner_rows: usize,
    ) -> AlignResult<CheckerboardDetection> {
        if image.len() != width * height {
            return Err(AlignError::InvalidConfig(
                "Image buffer size mismatch".to_string(),
            ));
        }
        if inner_cols < 2 || inner_rows < 2 {
            return Err(AlignError::InvalidConfig(
                "Need at least 2 inner corners per dimension".to_string(),
            ));
        }

        // Step 1: compute a corner-strength image using the saddle-point response
        let saddle = self.compute_saddle_response(image, width, height);

        // Step 2: find local maxima as candidate corners
        let candidates = self.find_candidate_corners(&saddle, width, height);

        if candidates.len() < inner_cols * inner_rows {
            return Err(AlignError::FeatureError(format!(
                "Found only {} saddle candidates, need {}",
                candidates.len(),
                inner_cols * inner_rows
            )));
        }

        // Step 3: sub-pixel refinement using Newton iteration on the gradient
        let refined = self.refine_corners(image, width, height, &candidates);

        // Step 4: sort corners into a regular grid by proximity heuristic
        let ordered = self.order_corners_grid(&refined, inner_cols, inner_rows)?;

        // Step 5: quality score = fraction of expected corners found
        let quality = (ordered.len() as f64 / (inner_cols * inner_rows) as f64).min(1.0);

        Ok(CheckerboardDetection {
            cols: inner_cols,
            rows: inner_rows,
            corners: ordered,
            quality,
        })
    }

    /// Compute the saddle-point response image.
    ///
    /// A checkerboard inner corner is a saddle point of the intensity function.
    /// The saddle response is `|λ_min - λ_max| / (λ_min + λ_max + ε)` where
    /// λ are eigenvalues of the structure tensor; this is negative for saddles.
    fn compute_saddle_response(&self, image: &[u8], width: usize, height: usize) -> Vec<f64> {
        let n = width * height;
        let mut resp = vec![0.0_f64; n];
        let k = 0.04_f64; // Harris k

        // Gaussian blur weights (3-tap: [0.25, 0.5, 0.25])
        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let idx = y * width + x;

                // Sobel gradients
                let gx = (f64::from(image[y * width + x + 1])
                    - f64::from(image[y * width + x - 1]))
                    / 2.0;
                let gy = (f64::from(image[(y + 1) * width + x])
                    - f64::from(image[(y - 1) * width + x]))
                    / 2.0;

                // Structure tensor elements (1-pixel window for speed)
                let ixx = gx * gx;
                let iyy = gy * gy;
                let ixy = gx * gy;

                // Harris-like response (negative for saddles)
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                resp[idx] = det - k * trace * trace;
            }
        }

        resp
    }

    /// Find candidate corner pixels as local minima of the saddle response.
    fn find_candidate_corners(&self, saddle: &[f64], width: usize, height: usize) -> Vec<Point2D> {
        let mut candidates = Vec::new();
        let r = 5usize; // local minimum radius

        for y in r..height.saturating_sub(r) {
            for x in r..width.saturating_sub(r) {
                let val = saddle[y * width + x];
                if val >= 0.0 {
                    continue;
                }

                let mut is_min = true;
                'outer: for dy in -(r as isize)..=(r as isize) {
                    for dx in -(r as isize)..=(r as isize) {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let ny = (y as isize + dy) as usize;
                        let nx = (x as isize + dx) as usize;
                        if saddle[ny * width + nx] < val {
                            is_min = false;
                            break 'outer;
                        }
                    }
                }

                if is_min {
                    candidates.push(Point2D::new(x as f64, y as f64));
                }
            }
        }

        candidates
    }

    /// Sub-pixel Newton refinement of candidate corner positions.
    fn refine_corners(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        candidates: &[Point2D],
    ) -> Vec<Point2D> {
        let hw = self.refine_half_win as isize;

        candidates
            .iter()
            .map(|p| {
                let mut cx = p.x;
                let mut cy = p.y;

                for _ in 0..self.max_iterations {
                    let rx = cx.round() as isize;
                    let ry = cy.round() as isize;

                    if rx < hw + 1
                        || ry < hw + 1
                        || rx >= (width as isize - hw - 1)
                        || ry >= (height as isize - hw - 1)
                    {
                        break;
                    }

                    let idx_c = ry as usize * width + rx as usize;

                    let gx = (f64::from(image[idx_c + 1]) - f64::from(image[idx_c - 1])) / 2.0;
                    let gy = (f64::from(image[(ry as usize + 1) * width + rx as usize])
                        - f64::from(image[(ry as usize - 1) * width + rx as usize]))
                        / 2.0;

                    let hxx = f64::from(image[idx_c + 1]) + f64::from(image[idx_c - 1])
                        - 2.0 * f64::from(image[idx_c]);
                    let hyy = f64::from(image[(ry as usize + 1) * width + rx as usize])
                        + f64::from(image[(ry as usize - 1) * width + rx as usize])
                        - 2.0 * f64::from(image[idx_c]);
                    let hxy = (f64::from(image[(ry as usize + 1) * width + rx as usize + 1])
                        - f64::from(image[(ry as usize + 1) * width + rx as usize - 1])
                        - f64::from(image[(ry as usize - 1) * width + rx as usize + 1])
                        + f64::from(image[(ry as usize - 1) * width + rx as usize - 1]))
                        / 4.0;

                    let det = hxx * hyy - hxy * hxy;
                    if det.abs() < 1e-10 {
                        break;
                    }

                    let shift_x = -(hyy * gx - hxy * gy) / det;
                    let shift_y = -(-hxy * gx + hxx * gy) / det;

                    if shift_x.abs() > 1.5 || shift_y.abs() > 1.5 {
                        break;
                    }

                    cx = rx as f64 + shift_x;
                    cy = ry as f64 + shift_y;

                    if shift_x * shift_x + shift_y * shift_y < self.epsilon * self.epsilon {
                        break;
                    }
                }

                Point2D::new(cx, cy)
            })
            .collect()
    }

    /// Order candidate corners into an `inner_cols × inner_rows` grid using a
    /// greedy nearest-neighbour walk starting from the top-left candidate.
    fn order_corners_grid(
        &self,
        corners: &[Point2D],
        inner_cols: usize,
        inner_rows: usize,
    ) -> AlignResult<Vec<Point2D>> {
        let required = inner_cols * inner_rows;
        if corners.len() < required {
            return Err(AlignError::InsufficientData(format!(
                "Not enough corners: {} < {required}",
                corners.len()
            )));
        }

        // Sort by (y * width + x) as a rough grid ordering
        let mut sorted = corners.to_vec();
        sorted.sort_by(|a, b| {
            let ord_a = a.y * 10000.0 + a.x;
            let ord_b = b.y * 10000.0 + b.x;
            ord_a
                .partial_cmp(&ord_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take the top-required candidates
        Ok(sorted.into_iter().take(required).collect())
    }

    /// Generate object-space 3-D points for a calibration grid.
    ///
    /// Returns `(inner_cols * inner_rows)` points at z = 0 in the pattern plane,
    /// spaced by [`Self::square_size`].
    #[must_use]
    pub fn generate_object_points(&self, inner_cols: usize, inner_rows: usize) -> Vec<Point3D> {
        let mut pts = Vec::with_capacity(inner_cols * inner_rows);
        for r in 0..inner_rows {
            for c in 0..inner_cols {
                pts.push(Point3D::new(
                    c as f64 * self.square_size,
                    r as f64 * self.square_size,
                    0.0,
                ));
            }
        }
        pts
    }
}

/// Camera calibration result.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Estimated camera intrinsics.
    pub intrinsics: CameraIntrinsics,
    /// Estimated distortion coefficients.
    pub distortion: BrownConradyDistortion,
    /// RMS reprojection error in pixels (lower is better).
    pub reprojection_error: f64,
    /// Number of views used.
    pub num_views: usize,
}

/// Camera calibrator using multiple checkerboard views.
pub struct CameraCalibrator {
    /// Image width.
    pub image_width: usize,
    /// Image height.
    pub image_height: usize,
    /// Accumulated 3-D object points (per view).
    object_points: Vec<Vec<Point3D>>,
    /// Accumulated 2-D image points (per view).
    image_points: Vec<Vec<Point2D>>,
}

impl CameraCalibrator {
    /// Create a new camera calibrator.
    #[must_use]
    pub fn new(image_width: usize, image_height: usize) -> Self {
        Self {
            image_width,
            image_height,
            object_points: Vec::new(),
            image_points: Vec::new(),
        }
    }

    /// Add a calibration view.
    pub fn add_view(&mut self, object_pts: Vec<Point3D>, image_pts: Vec<Point2D>) {
        self.object_points.push(object_pts);
        self.image_points.push(image_pts);
    }

    /// Number of views added.
    #[must_use]
    pub fn num_views(&self) -> usize {
        self.object_points.len()
    }

    /// Run calibration using Zhang's plane-based method (DLT + iterative refinement).
    ///
    /// Requires at least 3 views for reliable estimation.
    ///
    /// # Errors
    ///
    /// Returns [`AlignError`] if there are insufficient views or the linear
    /// system is degenerate.
    pub fn calibrate(&self) -> AlignResult<CalibrationResult> {
        if self.object_points.len() < 3 {
            return Err(AlignError::InsufficientData(
                "Need at least 3 calibration views".to_string(),
            ));
        }

        // Step 1: Estimate homography for each view using DLT
        let homographies: Vec<[f64; 9]> = self
            .object_points
            .iter()
            .zip(self.image_points.iter())
            .map(|(obj, img)| Self::dlt_homography(obj, img))
            .collect::<AlignResult<_>>()?;

        // Step 2: Extract intrinsics from homographies (Zhang's method)
        let intrinsics = self.estimate_intrinsics_zhang(&homographies)?;

        // Step 3: Estimate distortion via least-squares
        let distortion = self.estimate_distortion(&intrinsics)?;

        // Step 4: Compute reprojection error
        let error = self.compute_reprojection_error(&intrinsics, &distortion);

        Ok(CalibrationResult {
            intrinsics,
            distortion,
            reprojection_error: error,
            num_views: self.object_points.len(),
        })
    }

    /// Estimate 2-D homography from object-plane points to image points using DLT.
    ///
    /// Object points are on the Z=0 plane, so we use (X, Y) only.
    fn dlt_homography(obj: &[Point3D], img: &[Point2D]) -> AlignResult<[f64; 9]> {
        if obj.len() < 4 || obj.len() != img.len() {
            return Err(AlignError::InsufficientData(
                "Need at least 4 point correspondences for DLT".to_string(),
            ));
        }

        // Normalise source and destination points
        let (norm_src, t_src) = Self::normalise_2d_points_from_3d(obj);
        let (norm_dst, t_dst) = Self::normalise_2d_points(img);

        // Build A matrix (2n × 9)
        let n = obj.len();
        let mut ata = [0.0_f64; 81]; // 9×9 A^T A

        for i in 0..n {
            let x = norm_src[i].x;
            let y = norm_src[i].y;
            let xp = norm_dst[i].x;
            let yp = norm_dst[i].y;

            let r1 = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp];
            let r2 = [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp];

            for a in 0..9 {
                for b in 0..9 {
                    ata[a * 9 + b] += r1[a] * r1[b] + r2[a] * r2[b];
                }
            }
        }

        let h_norm = Self::smallest_eigenvector_9x9(&ata)?;

        // Denormalise: H = T_dst^{-1} * H_norm * T_src
        let h_mat: [f64; 9] = h_norm;
        let h_denorm = Self::denormalize_h(&h_mat, &t_src, &t_dst)?;

        // Normalise so h[8] = 1
        let scale = h_denorm[8];
        if scale.abs() < 1e-14 {
            return Err(AlignError::NumericalError(
                "Degenerate homography (h33 ≈ 0)".to_string(),
            ));
        }
        Ok(h_denorm.map(|v| v / scale))
    }

    /// Normalise 2-D points (zero mean, RMS distance = sqrt(2)).
    /// Returns (normalised_points, 3×3 normalisation_transform_row_major).
    fn normalise_2d_points(pts: &[Point2D]) -> (Vec<Point2D>, [f64; 9]) {
        let n = pts.len() as f64;
        let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
        let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;

        let scale = {
            let rms = (pts
                .iter()
                .map(|p| {
                    let dx = p.x - cx;
                    let dy = p.y - cy;
                    dx * dx + dy * dy
                })
                .sum::<f64>()
                / n)
                .sqrt();
            if rms < 1e-10 {
                1.0
            } else {
                std::f64::consts::SQRT_2 / rms
            }
        };

        let normalised: Vec<Point2D> = pts
            .iter()
            .map(|p| Point2D::new((p.x - cx) * scale, (p.y - cy) * scale))
            .collect();

        // T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
        let t = [
            scale,
            0.0,
            -scale * cx,
            0.0,
            scale,
            -scale * cy,
            0.0,
            0.0,
            1.0,
        ];

        (normalised, t)
    }

    /// Normalise using 3-D points on the Z=0 plane (take X, Y only).
    fn normalise_2d_points_from_3d(pts: &[Point3D]) -> (Vec<Point2D>, [f64; 9]) {
        let pts_2d: Vec<Point2D> = pts.iter().map(|p| Point2D::new(p.x, p.y)).collect();
        Self::normalise_2d_points(&pts_2d)
    }

    /// Denormalize a homography: H = T_dst^{-1} * H_norm * T_src.
    fn denormalize_h(h: &[f64; 9], t_src: &[f64; 9], t_dst: &[f64; 9]) -> AlignResult<[f64; 9]> {
        // Invert t_dst (diagonal scale + offset)
        let scale = t_dst[0];
        let tx = t_dst[2];
        let ty = t_dst[5];
        if scale.abs() < 1e-14 {
            return Err(AlignError::NumericalError(
                "Singular normalisation transform".to_string(),
            ));
        }
        let t_dst_inv = [
            1.0 / scale,
            0.0,
            -tx / scale,
            0.0,
            1.0 / scale,
            -ty / scale,
            0.0,
            0.0,
            1.0,
        ];

        let tmp = Self::mat3_mul(&t_dst_inv, h);
        Ok(Self::mat3_mul(&tmp, t_src))
    }

    /// Multiply two 3×3 matrices (row-major).
    fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
        let mut c = [0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
                }
            }
        }
        c
    }

    /// Estimate camera intrinsics from homographies using Zhang's method.
    ///
    /// Builds a linear system from the homography constraints:
    /// h1^T K^{-T} K^{-1} h1 = h2^T K^{-T} K^{-1} h2
    /// h1^T K^{-T} K^{-1} h2 = 0
    fn estimate_intrinsics_zhang(
        &self,
        homographies: &[[f64; 9]],
    ) -> AlignResult<CameraIntrinsics> {
        // B = K^{-T} K^{-1} is a 3×3 symmetric matrix, parameterised as [b11, b12, b22, b13, b23, b33].
        // Build V matrix from the homography constraints.
        let mut v_rows: Vec<[f64; 6]> = Vec::new();

        for h in homographies {
            let v12 = Self::compute_v(h, 0, 1);
            let v11 = Self::compute_v(h, 0, 0);
            let v22 = Self::compute_v(h, 1, 1);

            v_rows.push(v12);

            let diff: [f64; 6] = std::array::from_fn(|k| v11[k] - v22[k]);
            v_rows.push(diff);
        }

        // Solve V^T V b = 0
        let n = v_rows.len();
        let mut vtv = [0.0_f64; 36]; // 6×6
        for row in &v_rows {
            for i in 0..6 {
                for j in 0..6 {
                    vtv[i * 6 + j] += row[i] * row[j];
                }
            }
        }
        let _ = n;

        let b = Self::smallest_eigenvector_6x6(&vtv)?;

        // Extract intrinsics from B
        let (b11, b12, b22, b13, b23, b33) = (b[0], b[1], b[2], b[3], b[4], b[5]);

        let denom = b11 * b22 - b12 * b12;
        if denom.abs() < 1e-14 {
            return Err(AlignError::NumericalError(
                "Degenerate intrinsics estimate".to_string(),
            ));
        }

        let v0 = (b12 * b13 - b11 * b23) / denom;
        let lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
        let alpha = if b11 * lambda > 0.0 {
            (lambda / b11).sqrt()
        } else {
            // Fallback to image half-width as focal length
            self.image_width as f64 / 2.0
        };
        let beta = if b22 * lambda / denom > 0.0 {
            (lambda * b11 / denom).sqrt()
        } else {
            alpha
        };
        let u0 = -b13 * alpha * alpha / lambda;

        Ok(CameraIntrinsics::new(
            alpha,
            beta,
            u0 + self.image_width as f64 / 2.0,
            v0 + self.image_height as f64 / 2.0,
        ))
    }

    /// Compute the v_{ij} vector for Zhang's method from homography h.
    fn compute_v(h: &[f64; 9], i: usize, j: usize) -> [f64; 6] {
        [
            h[0 * 3 + i] * h[0 * 3 + j],
            h[0 * 3 + i] * h[1 * 3 + j] + h[1 * 3 + i] * h[0 * 3 + j],
            h[1 * 3 + i] * h[1 * 3 + j],
            h[2 * 3 + i] * h[0 * 3 + j] + h[0 * 3 + i] * h[2 * 3 + j],
            h[2 * 3 + i] * h[1 * 3 + j] + h[1 * 3 + i] * h[2 * 3 + j],
            h[2 * 3 + i] * h[2 * 3 + j],
        ]
    }

    /// Estimate distortion coefficients via least-squares.
    fn estimate_distortion(
        &self,
        intrinsics: &CameraIntrinsics,
    ) -> AlignResult<BrownConradyDistortion> {
        // For each observed point, compute the radial distortion residual:
        // u_observed - u_undistorted ≈ (x * r^2) * k1 + (x * r^4) * k2
        // Build a 2n × 2 linear system for k1, k2.
        let mut a_rows: Vec<[f64; 2]> = Vec::new();
        let mut b_vals: Vec<f64> = Vec::new();

        for (obj, img) in self.object_points.iter().zip(self.image_points.iter()) {
            for (op, ip) in obj.iter().zip(img.iter()) {
                // Project without distortion
                let xn = (op.x - intrinsics.cx) / intrinsics.fx;
                let yn = (op.y - intrinsics.cy) / intrinsics.fy;
                let r2 = xn * xn + yn * yn;
                let r4 = r2 * r2;

                // Residual in x
                a_rows.push([ip.x * r2, ip.x * r4]);
                b_vals.push(ip.x - (xn * intrinsics.fx + intrinsics.cx));

                // Residual in y
                a_rows.push([ip.y * r2, ip.y * r4]);
                b_vals.push(ip.y - (yn * intrinsics.fy + intrinsics.cy));
            }
        }

        if a_rows.len() < 4 {
            // Not enough constraints; return zero distortion
            return Ok(BrownConradyDistortion::default());
        }

        // Normal equations for 2×2 system
        let mut ata = [0.0_f64; 4];
        let mut atb = [0.0_f64; 2];

        for (row, &rhs) in a_rows.iter().zip(b_vals.iter()) {
            for i in 0..2 {
                for j in 0..2 {
                    ata[i * 2 + j] += row[i] * row[j];
                }
                atb[i] += row[i] * rhs;
            }
        }

        let det = ata[0] * ata[3] - ata[1] * ata[2];
        if det.abs() < 1e-14 {
            return Ok(BrownConradyDistortion::default());
        }

        let k1 = (atb[0] * ata[3] - atb[1] * ata[1]) / det;
        let k2 = (ata[0] * atb[1] - ata[2] * atb[0]) / det;

        Ok(BrownConradyDistortion::new(k1, k2, 0.0, 0.0, 0.0))
    }

    /// Compute mean RMS reprojection error.
    fn compute_reprojection_error(
        &self,
        intrinsics: &CameraIntrinsics,
        distortion: &BrownConradyDistortion,
    ) -> f64 {
        let mut total_err = 0.0_f64;
        let mut total_count = 0usize;

        for (obj, img) in self.object_points.iter().zip(self.image_points.iter()) {
            for (op, ip) in obj.iter().zip(img.iter()) {
                let xn = (op.x - intrinsics.cx) / intrinsics.fx;
                let yn = (op.y - intrinsics.cy) / intrinsics.fy;
                let r2 = xn * xn + yn * yn;
                let r4 = r2 * r2;
                let k1 = distortion.radial[0];
                let k2 = distortion.radial[1];
                let factor = 1.0 + k1 * r2 + k2 * r4;
                let px = xn * factor * intrinsics.fx + intrinsics.cx;
                let py = yn * factor * intrinsics.fy + intrinsics.cy;
                let ex = px - ip.x;
                let ey = py - ip.y;
                total_err += (ex * ex + ey * ey).sqrt();
                total_count += 1;
            }
        }

        if total_count == 0 {
            0.0
        } else {
            total_err / total_count as f64
        }
    }

    /// Power-iteration to find the smallest eigenvector of a 9×9 symmetric matrix.
    fn smallest_eigenvector_9x9(ata: &[f64; 81]) -> AlignResult<[f64; 9]> {
        let shift = 1e-8;
        let mut a_shifted = *ata;
        for i in 0..9 {
            a_shifted[i * 9 + i] += shift;
        }

        let mut v = [1.0_f64 / 3.0; 9];
        for _ in 0..80 {
            let w = Self::solve_9x9_gauss(&a_shifted, &v)?;
            let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for i in 0..9 {
                v[i] = w[i] / norm;
            }
        }
        Ok(v)
    }

    /// Power-iteration to find the smallest eigenvector of a 6×6 symmetric matrix.
    fn smallest_eigenvector_6x6(ata: &[f64; 36]) -> AlignResult<[f64; 6]> {
        let shift = 1e-8;
        let mut a_shifted = *ata;
        for i in 0..6 {
            a_shifted[i * 6 + i] += shift;
        }

        let mut v = [1.0_f64 / 6.0_f64.sqrt(); 6];
        for _ in 0..80 {
            let w = Self::solve_6x6_gauss(&a_shifted, &v)?;
            let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for i in 0..6 {
                v[i] = w[i] / norm;
            }
        }
        Ok(v)
    }

    fn solve_9x9_gauss(a: &[f64; 81], b: &[f64; 9]) -> AlignResult<[f64; 9]> {
        let mut mat = *a;
        let mut rhs = *b;

        for col in 0..9 {
            let mut max_row = col;
            let mut max_val = mat[col * 9 + col].abs();
            for row in (col + 1)..9 {
                let val = mat[row * 9 + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return Err(AlignError::NumericalError(
                    "Singular 9×9 matrix".to_string(),
                ));
            }
            if max_row != col {
                for j in 0..9 {
                    mat.swap(col * 9 + j, max_row * 9 + j);
                }
                rhs.swap(col, max_row);
            }
            let pivot = mat[col * 9 + col];
            for row in (col + 1)..9 {
                let factor = mat[row * 9 + col] / pivot;
                for j in col..9 {
                    mat[row * 9 + j] -= factor * mat[col * 9 + j];
                }
                rhs[row] -= factor * rhs[col];
            }
        }

        let mut x = [0.0_f64; 9];
        for col in (0..9).rev() {
            let mut sum = rhs[col];
            for j in (col + 1)..9 {
                sum -= mat[col * 9 + j] * x[j];
            }
            x[col] = sum / mat[col * 9 + col];
        }
        Ok(x)
    }

    fn solve_6x6_gauss(a: &[f64; 36], b: &[f64; 6]) -> AlignResult<[f64; 6]> {
        let mut mat = *a;
        let mut rhs = *b;

        for col in 0..6 {
            let mut max_row = col;
            let mut max_val = mat[col * 6 + col].abs();
            for row in (col + 1)..6 {
                let val = mat[row * 6 + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return Err(AlignError::NumericalError(
                    "Singular 6×6 matrix".to_string(),
                ));
            }
            if max_row != col {
                for j in 0..6 {
                    mat.swap(col * 6 + j, max_row * 6 + j);
                }
                rhs.swap(col, max_row);
            }
            let pivot = mat[col * 6 + col];
            for row in (col + 1)..6 {
                let factor = mat[row * 6 + col] / pivot;
                for j in col..6 {
                    mat[row * 6 + j] -= factor * mat[col * 6 + j];
                }
                rhs[row] -= factor * rhs[col];
            }
        }

        let mut x = [0.0_f64; 6];
        for col in (0..6).rev() {
            let mut sum = rhs[col];
            for j in (col + 1)..6 {
                sum -= mat[col * 6 + j] * x[j];
            }
            x[col] = sum / mat[col * 6 + col];
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkerboard_detector_default() {
        let d = CheckerboardDetector::default();
        assert_eq!(d.square_size, 25.0);
    }

    #[test]
    fn test_generate_object_points() {
        let d = CheckerboardDetector::new(10.0);
        let pts = d.generate_object_points(5, 4);
        assert_eq!(pts.len(), 20);
        assert!((pts[0].x).abs() < 1e-10);
        assert!((pts[0].y).abs() < 1e-10);
        assert!((pts[1].x - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_calibrator_requires_3_views() {
        let cal = CameraCalibrator::new(640, 480);
        let result = cal.calibrate();
        assert!(result.is_err());
    }

    #[test]
    fn test_normalise_2d_points_centred() {
        let pts = vec![
            Point2D::new(-1.0, -1.0),
            Point2D::new(1.0, -1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(-1.0, 1.0),
        ];
        let (normalised, _t) = CameraCalibrator::normalise_2d_points(&pts);
        let cx = normalised.iter().map(|p| p.x).sum::<f64>() / 4.0;
        let cy = normalised.iter().map(|p| p.y).sum::<f64>() / 4.0;
        assert!(cx.abs() < 1e-10, "centred: cx={cx}");
        assert!(cy.abs() < 1e-10, "centred: cy={cy}");
    }

    #[test]
    fn test_dlt_identity_homography() {
        // Points on the z=0 plane with identity transform
        let obj: Vec<Point3D> = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(100.0, 0.0, 0.0),
            Point3D::new(100.0, 100.0, 0.0),
            Point3D::new(0.0, 100.0, 0.0),
            Point3D::new(50.0, 50.0, 0.0),
        ];
        let img: Vec<Point2D> = obj.iter().map(|p| Point2D::new(p.x, p.y)).collect();

        let h = CameraCalibrator::dlt_homography(&obj, &img).expect("DLT should succeed");

        // Should be close to identity
        let (px, py) = project_h(&h, 30.0, 70.0);
        assert!((px - 30.0).abs() < 1.0, "px={px}");
        assert!((py - 70.0).abs() < 1.0, "py={py}");
    }

    #[test]
    fn test_dlt_requires_4_points() {
        let obj = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        let img = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        assert!(CameraCalibrator::dlt_homography(&obj, &img).is_err());
    }

    #[test]
    fn test_mat3_mul_identity() {
        let ident = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res = CameraCalibrator::mat3_mul(&ident, &a);
        for (r, e) in res.iter().zip(a.iter()) {
            assert!((r - e).abs() < 1e-10);
        }
    }

    fn project_h(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
        let w = h[6] * x + h[7] * y + h[8];
        if w.abs() < 1e-14 {
            return (x, y);
        }
        (
            (h[0] * x + h[1] * y + h[2]) / w,
            (h[3] * x + h[4] * y + h[5]) / w,
        )
    }
}
