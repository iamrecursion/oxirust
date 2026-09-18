//! Gaussian Process kernels for Bayesian Optimization.

use scirs2_core::ndarray::{Array1, Array2};

/// Gaussian Process kernel for Bayesian Optimization.
#[derive(Debug, Clone, Copy)]
pub enum GpKernel {
    /// Radial Basis Function (RBF) / Squared Exponential kernel.
    /// K(x, x') = sigma^2 * exp(-||x - x'||^2 / (2 * l^2))
    Rbf {
        /// Signal variance (output scale).
        sigma: f64,
        /// Length scale (input scale).
        length_scale: f64,
    },
    /// Matern kernel with nu = 3/2.
    /// K(x, x') = sigma^2 * (1 + sqrt(3) * r / l) * exp(-sqrt(3) * r / l)
    Matern32 {
        /// Signal variance.
        sigma: f64,
        /// Length scale.
        length_scale: f64,
    },
}

impl Default for GpKernel {
    fn default() -> Self {
        Self::Rbf {
            sigma: 1.0,
            length_scale: 1.0,
        }
    }
}

impl GpKernel {
    /// Compute kernel matrix K(X, X').
    pub(super) fn compute_kernel(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));
        for i in 0..n1 {
            for j in 0..n2 {
                let x1_row = x1.row(i);
                let x2_row = x2.row(j);
                let dist_sq = x1_row
                    .iter()
                    .zip(x2_row.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                k[[i, j]] = match self {
                    Self::Rbf {
                        sigma,
                        length_scale,
                    } => sigma.powi(2) * (-dist_sq / (2.0 * length_scale.powi(2))).exp(),
                    Self::Matern32 {
                        sigma,
                        length_scale,
                    } => {
                        let r = dist_sq.sqrt();
                        let sqrt3_r_l = (3.0_f64).sqrt() * r / length_scale;
                        sigma.powi(2) * (1.0 + sqrt3_r_l) * (-sqrt3_r_l).exp()
                    }
                };
            }
        }
        k
    }

    /// Compute kernel vector k(X, x).
    pub(super) fn compute_kernel_vector(
        &self,
        x_train: &Array2<f64>,
        x_test: &Array1<f64>,
    ) -> Array1<f64> {
        let n = x_train.nrows();
        let mut k = Array1::zeros(n);
        for i in 0..n {
            let x_train_row = x_train.row(i);
            let dist_sq = x_train_row
                .iter()
                .zip(x_test.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
            k[i] = match self {
                Self::Rbf {
                    sigma,
                    length_scale,
                } => sigma.powi(2) * (-dist_sq / (2.0 * length_scale.powi(2))).exp(),
                Self::Matern32 {
                    sigma,
                    length_scale,
                } => {
                    let r = dist_sq.sqrt();
                    let sqrt3_r_l = (3.0_f64).sqrt() * r / length_scale;
                    sigma.powi(2) * (1.0 + sqrt3_r_l) * (-sqrt3_r_l).exp()
                }
            };
        }
        k
    }
}
