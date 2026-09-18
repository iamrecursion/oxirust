//! Gaussian Process regressor for Bayesian Optimization.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{s, Array1, Array2};

use super::kernel::GpKernel;

/// Gaussian Process regressor for Bayesian Optimization.
///
/// Provides probabilistic predictions with uncertainty estimates.
#[derive(Debug)]
pub struct GaussianProcess {
    /// Kernel function.
    kernel: GpKernel,
    /// Noise variance (observation noise).
    noise_variance: f64,
    /// Training inputs (normalized to [0, 1]).
    x_train: Option<Array2<f64>>,
    /// Training outputs (standardized).
    y_train: Option<Array1<f64>>,
    /// Mean of training outputs (for standardization).
    y_mean: f64,
    /// Std of training outputs (for standardization).
    y_std: f64,
    /// Cholesky decomposition of K + sigma^2 * I (cached for efficiency).
    l_matrix: Option<Array2<f64>>,
    /// Alpha = L^T \ (L \ y) (cached).
    alpha: Option<Array1<f64>>,
}

impl GaussianProcess {
    /// Create a new Gaussian Process.
    pub fn new(kernel: GpKernel, noise_variance: f64) -> Self {
        Self {
            kernel,
            noise_variance,
            x_train: None,
            y_train: None,
            y_mean: 0.0,
            y_std: 1.0,
            l_matrix: None,
            alpha: None,
        }
    }

    /// Fit the GP to training data.
    pub fn fit(&mut self, x: Array2<f64>, y: Array1<f64>) -> TrainResult<()> {
        if x.nrows() != y.len() {
            return Err(TrainError::InvalidParameter(
                "X and y must have same number of samples".to_string(),
            ));
        }
        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = y.std(0.0).max(1e-8);
        let y_standardized = (&y - y_mean) / y_std;
        let k = self.kernel.compute_kernel(&x, &x);
        let mut k_noisy = k;
        for i in 0..k_noisy.nrows() {
            k_noisy[[i, i]] += self.noise_variance;
        }
        let l = self.cholesky(&k_noisy)?;
        let alpha_prime = self.forward_substitution(&l, &y_standardized)?;
        let alpha = self.backward_substitution(&l, &alpha_prime)?;
        self.x_train = Some(x);
        self.y_train = Some(y_standardized);
        self.y_mean = y_mean;
        self.y_std = y_std;
        self.l_matrix = Some(l);
        self.alpha = Some(alpha);
        Ok(())
    }

    /// Predict mean and standard deviation at test points.
    pub fn predict(&self, x_test: &Array2<f64>) -> TrainResult<(Array1<f64>, Array1<f64>)> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| TrainError::InvalidParameter("GP not fitted".to_string()))?;
        let l_matrix = self
            .l_matrix
            .as_ref()
            .expect("l_matrix must be set after fitting");
        let alpha = self
            .alpha
            .as_ref()
            .expect("alpha must be set after fitting");
        let n_test = x_test.nrows();
        let mut means = Array1::zeros(n_test);
        let mut stds = Array1::zeros(n_test);
        for i in 0..n_test {
            let x = x_test.row(i).to_owned();
            let k_star = self.kernel.compute_kernel_vector(x_train, &x);
            let mean_standardized = k_star.dot(alpha);
            means[i] = mean_standardized * self.y_std + self.y_mean;
            let k_star_star = self
                .kernel
                .compute_kernel_vector(&x_test.slice(s![i..i + 1, ..]).to_owned(), &x)[0];
            let v = self
                .forward_substitution(l_matrix, &k_star)
                .unwrap_or_else(|_| Array1::zeros(k_star.len()));
            let variance_standardized = k_star_star - v.dot(&v);
            stds[i] = (variance_standardized.max(1e-10) * self.y_std.powi(2)).sqrt();
        }
        Ok((means, stds))
    }

    /// Cholesky decomposition: K = L * L^T.
    fn cholesky(&self, k: &Array2<f64>) -> TrainResult<Array2<f64>> {
        let n = k.nrows();
        let mut l = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k_idx in 0..j {
                    sum += l[[i, k_idx]] * l[[j, k_idx]];
                }
                if i == j {
                    let val = k[[i, i]] - sum;
                    if val <= 0.0 {
                        l[[i, j]] = (k[[i, i]] - sum + 1e-6).sqrt();
                    } else {
                        l[[i, j]] = val.sqrt();
                    }
                } else {
                    l[[i, j]] = (k[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
        Ok(l)
    }

    /// Forward substitution: solve L * x = b.
    fn forward_substitution(&self, l: &Array2<f64>, b: &Array1<f64>) -> TrainResult<Array1<f64>> {
        let n = l.nrows();
        let mut x = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * x[j];
            }
            x[i] = (b[i] - sum) / l[[i, i]];
        }
        Ok(x)
    }

    /// Backward substitution: solve L^T * x = b.
    fn backward_substitution(&self, l: &Array2<f64>, b: &Array1<f64>) -> TrainResult<Array1<f64>> {
        let n = l.nrows();
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[[j, i]] * x[j];
            }
            x[i] = (b[i] - sum) / l[[i, i]];
        }
        Ok(x)
    }
}
