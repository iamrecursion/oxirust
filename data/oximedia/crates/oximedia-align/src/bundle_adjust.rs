//! Bundle adjustment for multi-camera calibration.
//!
//! Implements a Levenberg-Marquardt-style optimizer for globally consistent
//! multi-camera calibration by minimizing reprojection error across all
//! camera views simultaneously.
//!
//! # Algorithm
//!
//! Given N cameras and M 3D points observed across multiple views:
//!
//! 1. Initialize camera parameters (rotation, translation, focal length) and
//!    3D point positions from an initial estimate.
//! 2. Compute the Jacobian of the reprojection error with respect to all
//!    parameters.
//! 3. Solve the normal equations `(J^T J + lambda * diag(J^T J)) * delta = -J^T r`
//!    where `r` is the residual vector.
//! 4. Update parameters and adjust `lambda` based on error reduction.
//! 5. Repeat until convergence.
//!
//! # References
//!
//! - Triggs, B. et al. "Bundle Adjustment — A Modern Synthesis" (2000).
//! - Levenberg, K. "A Method for the Solution of Certain Non-Linear Problems
//!   in Least Squares" (1944).

#![allow(clippy::cast_precision_loss)]

use crate::{AlignError, AlignResult};

/// Configuration for bundle adjustment.
#[derive(Debug, Clone)]
pub struct BundleAdjustConfig {
    /// Maximum number of Levenberg-Marquardt iterations.
    pub max_iterations: usize,
    /// Initial damping factor (lambda).
    pub initial_lambda: f64,
    /// Factor to increase lambda on failed step.
    pub lambda_up_factor: f64,
    /// Factor to decrease lambda on successful step.
    pub lambda_down_factor: f64,
    /// Convergence threshold on parameter update norm.
    pub param_tolerance: f64,
    /// Convergence threshold on error reduction.
    pub error_tolerance: f64,
}

impl Default for BundleAdjustConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            initial_lambda: 1e-3,
            lambda_up_factor: 10.0,
            lambda_down_factor: 0.1,
            param_tolerance: 1e-8,
            error_tolerance: 1e-10,
        }
    }
}

/// A 2D observation: which camera saw which 3D point, and at what pixel location.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Camera index.
    pub camera_idx: usize,
    /// 3D point index.
    pub point_idx: usize,
    /// Observed pixel x coordinate.
    pub pixel_x: f64,
    /// Observed pixel y coordinate.
    pub pixel_y: f64,
}

impl Observation {
    /// Create a new observation.
    #[must_use]
    pub fn new(camera_idx: usize, point_idx: usize, pixel_x: f64, pixel_y: f64) -> Self {
        Self {
            camera_idx,
            point_idx,
            pixel_x,
            pixel_y,
        }
    }
}

/// Camera parameters: 6 extrinsics (3 rotation + 3 translation) + 1 focal length.
/// Rotation is stored as a Rodrigues vector (axis * angle).
#[derive(Debug, Clone)]
pub struct CameraParams {
    /// Rodrigues rotation vector (3 elements: rx, ry, rz).
    pub rotation: [f64; 3],
    /// Translation vector (3 elements: tx, ty, tz).
    pub translation: [f64; 3],
    /// Focal length.
    pub focal_length: f64,
}

impl CameraParams {
    /// Create a new camera parameter set.
    #[must_use]
    pub fn new(rotation: [f64; 3], translation: [f64; 3], focal_length: f64) -> Self {
        Self {
            rotation,
            translation,
            focal_length,
        }
    }

    /// Create an identity camera (no rotation, no translation, unit focal length).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: [0.0; 3],
            translation: [0.0; 3],
            focal_length: 1.0,
        }
    }
}

/// A 3D point.
#[derive(Debug, Clone)]
pub struct Point3D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

impl Point3D {
    /// Create a new 3D point.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Result of bundle adjustment.
#[derive(Debug, Clone)]
pub struct BundleAdjustResult {
    /// Optimized camera parameters.
    pub cameras: Vec<CameraParams>,
    /// Optimized 3D points.
    pub points: Vec<Point3D>,
    /// Final total reprojection error (sum of squared residuals).
    pub final_error: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the optimization converged.
    pub converged: bool,
}

/// Bundle adjuster using Levenberg-Marquardt optimization.
pub struct BundleAdjuster {
    /// Configuration.
    pub config: BundleAdjustConfig,
}

impl Default for BundleAdjuster {
    fn default() -> Self {
        Self {
            config: BundleAdjustConfig::default(),
        }
    }
}

impl BundleAdjuster {
    /// Create a new bundle adjuster.
    #[must_use]
    pub fn new(config: BundleAdjustConfig) -> Self {
        Self { config }
    }

    /// Run bundle adjustment.
    ///
    /// # Arguments
    ///
    /// * `cameras` - Initial camera parameters.
    /// * `points` - Initial 3D point positions.
    /// * `observations` - 2D observations linking cameras to points.
    ///
    /// # Errors
    ///
    /// Returns an error if there are insufficient observations or if the
    /// optimization fails numerically.
    pub fn optimize(
        &self,
        cameras: &[CameraParams],
        points: &[Point3D],
        observations: &[Observation],
    ) -> AlignResult<BundleAdjustResult> {
        if cameras.is_empty() {
            return Err(AlignError::InsufficientData(
                "Need at least one camera".to_string(),
            ));
        }
        if points.is_empty() {
            return Err(AlignError::InsufficientData(
                "Need at least one 3D point".to_string(),
            ));
        }
        if observations.is_empty() {
            return Err(AlignError::InsufficientData(
                "Need at least one observation".to_string(),
            ));
        }

        // Flatten parameters into a single vector
        // Each camera: 7 params (3 rotation + 3 translation + 1 focal)
        // Each point: 3 params (x, y, z)
        let num_cam_params = cameras.len() * 7;
        let num_point_params = points.len() * 3;
        let total_params = num_cam_params + num_point_params;
        let num_residuals = observations.len() * 2;

        let mut params = vec![0.0_f64; total_params];

        // Pack camera parameters
        for (i, cam) in cameras.iter().enumerate() {
            let base = i * 7;
            params[base] = cam.rotation[0];
            params[base + 1] = cam.rotation[1];
            params[base + 2] = cam.rotation[2];
            params[base + 3] = cam.translation[0];
            params[base + 4] = cam.translation[1];
            params[base + 5] = cam.translation[2];
            params[base + 6] = cam.focal_length;
        }

        // Pack 3D points
        for (i, pt) in points.iter().enumerate() {
            let base = num_cam_params + i * 3;
            params[base] = pt.x;
            params[base + 1] = pt.y;
            params[base + 2] = pt.z;
        }

        let mut lambda = self.config.initial_lambda;
        let mut current_error = self.compute_total_error(&params, cameras.len(), observations)?;
        let mut converged = false;
        let mut iter = 0;

        for iteration in 0..self.config.max_iterations {
            iter = iteration + 1;

            // Compute Jacobian and residuals
            let (jacobian, residuals) = self.compute_jacobian_and_residuals(
                &params,
                cameras.len(),
                points.len(),
                observations,
            )?;

            // Compute J^T * J and J^T * r
            let jtj = self.compute_jtj(&jacobian, total_params, num_residuals);
            let jtr = self.compute_jtr(&jacobian, &residuals, total_params, num_residuals);

            // Solve (J^T J + lambda * diag(J^T J)) * delta = -J^T r
            let delta = self.solve_normal_equations(&jtj, &jtr, total_params, lambda)?;

            // Check convergence on parameter update
            let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            let param_norm: f64 = params.iter().map(|p| p * p).sum::<f64>().sqrt().max(1.0);

            if delta_norm / param_norm < self.config.param_tolerance {
                converged = true;
                break;
            }

            // Trial update
            let trial_params: Vec<f64> = params.iter().zip(&delta).map(|(p, d)| p + d).collect();

            let trial_error =
                self.compute_total_error(&trial_params, cameras.len(), observations)?;

            if trial_error < current_error {
                // Accept step, reduce lambda
                let error_reduction = (current_error - trial_error) / current_error.max(1e-15);
                params = trial_params;
                current_error = trial_error;
                lambda *= self.config.lambda_down_factor;
                lambda = lambda.max(1e-12);

                if error_reduction < self.config.error_tolerance {
                    converged = true;
                    break;
                }
            } else {
                // Reject step, increase lambda
                lambda *= self.config.lambda_up_factor;
                lambda = lambda.min(1e10);
            }
        }

        // Unpack results
        let mut opt_cameras = Vec::with_capacity(cameras.len());
        for i in 0..cameras.len() {
            let base = i * 7;
            opt_cameras.push(CameraParams::new(
                [params[base], params[base + 1], params[base + 2]],
                [params[base + 3], params[base + 4], params[base + 5]],
                params[base + 6],
            ));
        }

        let mut opt_points = Vec::with_capacity(points.len());
        for i in 0..points.len() {
            let base = num_cam_params + i * 3;
            opt_points.push(Point3D::new(
                params[base],
                params[base + 1],
                params[base + 2],
            ));
        }

        Ok(BundleAdjustResult {
            cameras: opt_cameras,
            points: opt_points,
            final_error: current_error,
            iterations: iter,
            converged,
        })
    }

    /// Project a 3D point through a camera, returning the 2D pixel position.
    fn project(params: &[f64], cam_idx: usize, point_params: &[f64; 3]) -> (f64, f64) {
        let base = cam_idx * 7;
        let rx = params[base];
        let ry = params[base + 1];
        let rz = params[base + 2];
        let tx = params[base + 3];
        let ty = params[base + 4];
        let tz = params[base + 5];
        let f = params[base + 6];

        let px = point_params[0];
        let py = point_params[1];
        let pz = point_params[2];

        // Rodrigues rotation
        let theta = (rx * rx + ry * ry + rz * rz).sqrt();
        let (r00, r01, r02, r10, r11, r12, r20, r21, r22) = if theta < 1e-10 {
            // Near identity
            (1.0, -rz, ry, rz, 1.0, -rx, -ry, rx, 1.0)
        } else {
            let c = theta.cos();
            let s = theta.sin();
            let t = 1.0 - c;
            let kx = rx / theta;
            let ky = ry / theta;
            let kz = rz / theta;
            (
                t * kx * kx + c,
                t * kx * ky - s * kz,
                t * kx * kz + s * ky,
                t * kx * ky + s * kz,
                t * ky * ky + c,
                t * ky * kz - s * kx,
                t * kx * kz - s * ky,
                t * ky * kz + s * kx,
                t * kz * kz + c,
            )
        };

        // Apply rotation and translation
        let cx = r00 * px + r01 * py + r02 * pz + tx;
        let cy = r10 * px + r11 * py + r12 * pz + ty;
        let cz = r20 * px + r21 * py + r22 * pz + tz;

        // Perspective projection
        if cz.abs() < 1e-10 {
            return (0.0, 0.0);
        }

        let proj_x = f * cx / cz;
        let proj_y = f * cy / cz;

        (proj_x, proj_y)
    }

    /// Compute total reprojection error.
    fn compute_total_error(
        &self,
        params: &[f64],
        num_cameras: usize,
        observations: &[Observation],
    ) -> AlignResult<f64> {
        let num_cam_params = num_cameras * 7;
        let mut total = 0.0_f64;

        for obs in observations {
            let pt_base = num_cam_params + obs.point_idx * 3;
            if pt_base + 2 >= params.len() {
                return Err(AlignError::InvalidConfig(
                    "Point index out of range".to_string(),
                ));
            }

            let point = [params[pt_base], params[pt_base + 1], params[pt_base + 2]];
            let (px, py) = Self::project(params, obs.camera_idx, &point);

            let rx = px - obs.pixel_x;
            let ry = py - obs.pixel_y;
            total += rx * rx + ry * ry;
        }

        Ok(total)
    }

    /// Compute Jacobian matrix and residual vector via finite differences.
    fn compute_jacobian_and_residuals(
        &self,
        params: &[f64],
        num_cameras: usize,
        num_points: usize,
        observations: &[Observation],
    ) -> AlignResult<(Vec<f64>, Vec<f64>)> {
        let num_cam_params = num_cameras * 7;
        let total_params = num_cam_params + num_points * 3;
        let num_residuals = observations.len() * 2;

        let mut jacobian = vec![0.0_f64; num_residuals * total_params];
        let mut residuals = vec![0.0_f64; num_residuals];

        let epsilon = 1e-7;

        for (obs_idx, obs) in observations.iter().enumerate() {
            let pt_base = num_cam_params + obs.point_idx * 3;
            let point = [params[pt_base], params[pt_base + 1], params[pt_base + 2]];
            let (px, py) = Self::project(params, obs.camera_idx, &point);

            let res_base = obs_idx * 2;
            residuals[res_base] = px - obs.pixel_x;
            residuals[res_base + 1] = py - obs.pixel_y;

            // Jacobian w.r.t. camera parameters
            let cam_base = obs.camera_idx * 7;
            for p in 0..7 {
                let param_idx = cam_base + p;
                let mut params_plus = params.to_vec();
                params_plus[param_idx] += epsilon;

                let pt_p = [
                    params_plus[pt_base],
                    params_plus[pt_base + 1],
                    params_plus[pt_base + 2],
                ];
                let (px_plus, py_plus) = Self::project(&params_plus, obs.camera_idx, &pt_p);

                jacobian[res_base * total_params + param_idx] = (px_plus - px) / epsilon;
                jacobian[(res_base + 1) * total_params + param_idx] = (py_plus - py) / epsilon;
            }

            // Jacobian w.r.t. point parameters
            for p in 0..3 {
                let param_idx = pt_base + p;
                let mut point_plus = point;
                point_plus[p] += epsilon;

                let (px_plus, py_plus) = Self::project(params, obs.camera_idx, &point_plus);

                jacobian[res_base * total_params + param_idx] = (px_plus - px) / epsilon;
                jacobian[(res_base + 1) * total_params + param_idx] = (py_plus - py) / epsilon;
            }
        }

        Ok((jacobian, residuals))
    }

    /// Compute J^T * J.
    fn compute_jtj(&self, j: &[f64], n_params: usize, n_residuals: usize) -> Vec<f64> {
        let mut jtj = vec![0.0_f64; n_params * n_params];

        for r in 0..n_residuals {
            for i in 0..n_params {
                let ji = j[r * n_params + i];
                if ji.abs() < 1e-15 {
                    continue;
                }
                for k in i..n_params {
                    let jk = j[r * n_params + k];
                    if jk.abs() < 1e-15 {
                        continue;
                    }
                    let val = ji * jk;
                    jtj[i * n_params + k] += val;
                    if i != k {
                        jtj[k * n_params + i] += val;
                    }
                }
            }
        }

        jtj
    }

    /// Compute -J^T * r.
    fn compute_jtr(&self, j: &[f64], r: &[f64], n_params: usize, n_residuals: usize) -> Vec<f64> {
        let mut jtr = vec![0.0_f64; n_params];

        for res in 0..n_residuals {
            let rv = r[res];
            if rv.abs() < 1e-15 {
                continue;
            }
            for p in 0..n_params {
                jtr[p] -= j[res * n_params + p] * rv;
            }
        }

        jtr
    }

    /// Solve the damped normal equations using Cholesky-like diagonal decomposition.
    fn solve_normal_equations(
        &self,
        jtj: &[f64],
        jtr: &[f64],
        n: usize,
        lambda: f64,
    ) -> AlignResult<Vec<f64>> {
        // Add damping to diagonal
        let mut a = jtj.to_vec();
        for i in 0..n {
            a[i * n + i] += lambda * a[i * n + i].max(1e-6);
        }

        // Gaussian elimination with partial pivoting
        let mut b = jtr.to_vec();

        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = a[col * n + col].abs();
            for row in (col + 1)..n {
                let val = a[row * n + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-14 {
                // Skip near-zero columns
                continue;
            }

            // Swap rows
            if max_row != col {
                for j in 0..n {
                    a.swap(col * n + j, max_row * n + j);
                }
                b.swap(col, max_row);
            }

            // Eliminate
            let pivot = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot;
                for j in col..n {
                    a[row * n + j] -= factor * a[col * n + j];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back substitution
        let mut x = vec![0.0_f64; n];
        for col in (0..n).rev() {
            if a[col * n + col].abs() < 1e-14 {
                continue;
            }
            let mut sum = b[col];
            for j in (col + 1)..n {
                sum -= a[col * n + j] * x[j];
            }
            x[col] = sum / a[col * n + col];
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BundleAdjustConfig::default();
        assert_eq!(config.max_iterations, 50);
        assert!((config.initial_lambda - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_camera_params_identity() {
        let cam = CameraParams::identity();
        assert_eq!(cam.rotation, [0.0; 3]);
        assert_eq!(cam.translation, [0.0; 3]);
        assert!((cam.focal_length - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_creation() {
        let pt = Point3D::new(1.0, 2.0, 3.0);
        assert!((pt.x - 1.0).abs() < 1e-10);
        assert!((pt.y - 2.0).abs() < 1e-10);
        assert!((pt.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_observation_creation() {
        let obs = Observation::new(0, 1, 100.0, 200.0);
        assert_eq!(obs.camera_idx, 0);
        assert_eq!(obs.point_idx, 1);
    }

    #[test]
    fn test_empty_cameras_error() {
        let ba = BundleAdjuster::default();
        let result = ba.optimize(&[], &[Point3D::new(0.0, 0.0, 1.0)], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_points_error() {
        let ba = BundleAdjuster::default();
        let result = ba.optimize(&[CameraParams::identity()], &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_observations_error() {
        let ba = BundleAdjuster::default();
        let result = ba.optimize(
            &[CameraParams::identity()],
            &[Point3D::new(0.0, 0.0, 1.0)],
            &[],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_projection_identity_camera() {
        // Camera at origin looking down +z, focal=1.0
        let params = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let point = [1.0, 2.0, 5.0];
        let (px, py) = BundleAdjuster::project(&params, 0, &point);
        // Expected: f * x/z = 1.0 * 1.0/5.0 = 0.2
        assert!((px - 0.2).abs() < 1e-6, "px={px}");
        assert!((py - 0.4).abs() < 1e-6, "py={py}");
    }

    #[test]
    fn test_projection_with_translation() {
        // Camera translated by (1, 0, 0), looking down +z
        let params = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let point = [1.0, 0.0, 5.0];
        let (px, _py) = BundleAdjuster::project(&params, 0, &point);
        // Expected: f * (x + tx) / z = 1.0 * (1.0 + 1.0) / 5.0 = 0.4
        assert!((px - 0.4).abs() < 1e-6, "px={px}");
    }

    #[test]
    fn test_simple_optimization() {
        // Set up a simple scenario:
        // One camera at origin, looking down +z with focal=100
        // A few points in front of the camera
        let cameras = vec![CameraParams::new([0.0; 3], [0.0; 3], 100.0)];

        let points = vec![
            Point3D::new(0.5, 0.5, 5.0),
            Point3D::new(-0.5, 0.3, 4.0),
            Point3D::new(0.0, -0.5, 6.0),
        ];

        // Generate synthetic observations
        let params: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
        let mut observations = Vec::new();
        for (i, pt) in points.iter().enumerate() {
            let point = [pt.x, pt.y, pt.z];
            let (px, py) = BundleAdjuster::project(&params, 0, &point);
            observations.push(Observation::new(0, i, px, py));
        }

        let ba = BundleAdjuster::new(BundleAdjustConfig {
            max_iterations: 20,
            ..BundleAdjustConfig::default()
        });

        let result = ba
            .optimize(&cameras, &points, &observations)
            .expect("should succeed");

        // Error should be very small (observations are consistent with initial params)
        assert!(
            result.final_error < 1.0,
            "final_error={}",
            result.final_error
        );
    }

    #[test]
    fn test_optimization_with_perturbation() {
        // Camera with known params
        let true_cameras = vec![CameraParams::new([0.0; 3], [0.0; 3], 100.0)];
        let true_points = vec![
            Point3D::new(1.0, 0.0, 5.0),
            Point3D::new(-1.0, 0.0, 5.0),
            Point3D::new(0.0, 1.0, 5.0),
            Point3D::new(0.0, -1.0, 5.0),
        ];

        // Generate synthetic observations from true params
        let true_params: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
        let mut observations = Vec::new();
        for (i, pt) in true_points.iter().enumerate() {
            let point = [pt.x, pt.y, pt.z];
            let (px, py) = BundleAdjuster::project(&true_params, 0, &point);
            observations.push(Observation::new(0, i, px, py));
        }

        // Perturb the initial points slightly
        let perturbed_points = vec![
            Point3D::new(1.1, 0.1, 5.1),
            Point3D::new(-0.9, 0.1, 4.9),
            Point3D::new(0.1, 1.1, 5.1),
            Point3D::new(0.1, -0.9, 4.9),
        ];

        let ba = BundleAdjuster::new(BundleAdjustConfig {
            max_iterations: 30,
            ..BundleAdjustConfig::default()
        });

        let result = ba
            .optimize(&true_cameras, &perturbed_points, &observations)
            .expect("should succeed");

        // The optimizer should reduce the error
        assert!(
            result.final_error < 10.0,
            "final_error={}",
            result.final_error
        );
    }

    #[test]
    fn test_two_camera_optimization() {
        // Two cameras looking at the same point
        let cameras = vec![
            CameraParams::new([0.0; 3], [0.0, 0.0, 0.0], 100.0),
            CameraParams::new([0.0; 3], [2.0, 0.0, 0.0], 100.0),
        ];
        let points = vec![Point3D::new(1.0, 0.0, 5.0)];

        let params1: Vec<f64> = vec![
            // cam 0
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, // cam 1
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 100.0,
        ];
        let point = [1.0, 0.0, 5.0];

        let (px0, py0) = BundleAdjuster::project(&params1, 0, &point);
        let (px1, py1) = BundleAdjuster::project(&params1, 1, &point);

        let observations = vec![
            Observation::new(0, 0, px0, py0),
            Observation::new(1, 0, px1, py1),
        ];

        let ba = BundleAdjuster::default();
        let result = ba
            .optimize(&cameras, &points, &observations)
            .expect("should succeed");

        assert!(
            result.final_error < 1.0,
            "final_error={}",
            result.final_error
        );
        assert_eq!(result.cameras.len(), 2);
        assert_eq!(result.points.len(), 1);
    }

    #[test]
    fn test_bundle_adjust_result_fields() {
        let result = BundleAdjustResult {
            cameras: vec![CameraParams::identity()],
            points: vec![Point3D::new(0.0, 0.0, 1.0)],
            final_error: 0.1,
            iterations: 5,
            converged: true,
        };
        assert!(result.converged);
        assert_eq!(result.iterations, 5);
    }
}
