//! Core Gaussian Process implementation
//!
//! This module provides the fundamental GP regression functionality.

use super::kernel::Kernel;
use super::prior::Prior;
use crate::error::StatsResult;
use scirs2_core::error::CoreError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};

/// Gaussian Process for regression
///
/// A Gaussian Process defines a distribution over functions. Given training data,
/// it can make predictions with uncertainty estimates.
#[derive(Clone)]
pub struct GaussianProcess<K: Kernel, P: Prior> {
    /// Kernel (covariance) function
    pub kernel: K,
    /// Prior mean function
    pub prior: P,
    /// Training inputs
    x_train: Option<Array2<f64>>,
    /// Training outputs (mean-subtracted)
    y_train_centered: Option<Array1<f64>>,
    /// Cholesky decomposition of K(X, X) + noise * I
    l_matrix: Option<Array2<f64>>,
    /// alpha = L^T \ (L \ y)
    alpha: Option<Array1<f64>>,
    /// Noise level (observation noise)
    pub noise: f64,
}

impl<K: Kernel, P: Prior> GaussianProcess<K, P> {
    /// Create a new Gaussian Process
    pub fn new(kernel: K, prior: P, noise: f64) -> Self {
        Self {
            kernel,
            prior,
            x_train: None,
            y_train_centered: None,
            l_matrix: None,
            alpha: None,
            noise: noise.max(1e-10), // Ensure numerical stability
        }
    }

    /// Fit the Gaussian Process to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> StatsResult<()> {
        if x.nrows() != y.len() {
            return Err(
                CoreError::InvalidInput(scirs2_core::error::ErrorContext::new(
                    "Number of samples in X and y must match",
                ))
                .into(),
            );
        }

        if x.nrows() == 0 {
            return Err(
                CoreError::InvalidInput(scirs2_core::error::ErrorContext::new(
                    "Cannot fit with zero samples",
                ))
                .into(),
            );
        }

        // Compute prior mean
        let prior_mean = self.prior.compute_vector(x);

        // Center the targets
        let y_centered = y - &prior_mean;

        // Compute covariance matrix K(X, X)
        let mut k = self.kernel.compute_matrix(x);

        // Add noise to diagonal for numerical stability
        for i in 0..k.nrows() {
            k[[i, i]] += self.noise;
        }

        // Cholesky decomposition: K = L L^T
        let l = match cholesky_decomposition(&k) {
            Ok(l) => l,
            Err(_) => {
                // If Cholesky fails, add more jitter
                let jitter = 1e-6;
                for i in 0..k.nrows() {
                    k[[i, i]] += jitter;
                }
                cholesky_decomposition(&k).map_err(|e| {
                    CoreError::ComputationError(scirs2_core::error::ErrorContext::new(format!(
                        "Cholesky decomposition failed: {}",
                        e
                    )))
                })?
            }
        };

        // Solve L alpha_1 = y  (forward substitution)
        let alpha_1 = solve_lower_triangular(&l, &y_centered)?;

        // Solve L^T alpha = alpha_1  (backward substitution)
        let alpha = solve_upper_triangular(&l.t().to_owned(), &alpha_1)?;

        // Store results
        self.x_train = Some(x.clone());
        self.y_train_centered = Some(y_centered);
        self.l_matrix = Some(l);
        self.alpha = Some(alpha);

        Ok(())
    }

    /// Predict mean values for new inputs
    pub fn predict(&self, x: &Array2<f64>) -> StatsResult<Array1<f64>> {
        let (mean, _std) = self.predict_with_std(x)?;
        Ok(mean)
    }

    /// Predict with mean and standard deviation
    pub fn predict_with_std(&self, x: &Array2<f64>) -> StatsResult<(Array1<f64>, Array1<f64>)> {
        if self.x_train.is_none() || self.alpha.is_none() {
            return Err(
                CoreError::InvalidInput(scirs2_core::error::ErrorContext::new(
                    "GP must be fitted before making predictions",
                ))
                .into(),
            );
        }

        let x_train = self.x_train.as_ref().expect("Operation failed");
        let alpha = self.alpha.as_ref().expect("Operation failed");
        let l = self.l_matrix.as_ref().expect("Operation failed");

        // Compute K(X_test, X_train)
        let k_trans = self.kernel.compute_cross_matrix(x, x_train);

        // Mean: K(X_test, X_train) @ alpha + prior
        let mean_centered = k_trans.dot(alpha);
        let prior_mean = self.prior.compute_vector(x);
        let mean = mean_centered + prior_mean;

        // Variance calculation
        // v = L \ K(X_test, X_train)^T
        let k_trans_t = k_trans.t().to_owned();
        let v = solve_lower_triangular_matrix(l, &k_trans_t)?;

        // Compute variance for each test point
        let mut variance = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            // K(x_test[i], x_test[i])
            let k_self = self.kernel.compute(&x.row(i), &x.row(i));

            // ||v[i]||^2
            let v_norm_sq: f64 = v.column(i).iter().map(|&x| x * x).sum();

            // var = k_self - ||v||^2 + noise
            variance[i] = (k_self - v_norm_sq + self.noise).max(0.0);
        }

        let std = variance.mapv(|x| x.sqrt());

        Ok((mean, std))
    }

    /// Predict mean for a single point
    pub fn predict_single(&self, x: &ArrayView1<f64>) -> StatsResult<f64> {
        let x_mat = x.to_owned().insert_axis(Axis(0));
        let pred = self.predict(&x_mat)?;
        Ok(pred[0])
    }

    /// Predict variance for a single point
    pub fn predict_variance_single(&self, x: &ArrayView1<f64>) -> StatsResult<f64> {
        let x_mat = x.to_owned().insert_axis(Axis(0));
        let (_mean, std) = self.predict_with_std(&x_mat)?;
        Ok(std[0] * std[0])
    }

    /// Compute log marginal likelihood
    ///
    /// This is useful for hyperparameter optimization.
    pub fn log_marginal_likelihood(&self) -> StatsResult<f64> {
        if self.y_train_centered.is_none() || self.l_matrix.is_none() {
            return Err(
                CoreError::InvalidInput(scirs2_core::error::ErrorContext::new(
                    "GP must be fitted before computing log marginal likelihood",
                ))
                .into(),
            );
        }

        let y = self.y_train_centered.as_ref().expect("Operation failed");
        let l = self.l_matrix.as_ref().expect("Operation failed");
        let alpha = self.alpha.as_ref().expect("Operation failed");

        let n = y.len() as f64;

        // Compute data fit: -0.5 * y^T @ alpha
        let data_fit = -0.5 * y.dot(alpha);

        // Compute complexity penalty: -sum(log(diag(L)))
        let log_det: f64 = l.diag().iter().map(|&x| x.ln()).sum();
        let complexity = -log_det;

        // Normalization constant: -n/2 * log(2π)
        let normalization = -0.5 * n * (2.0 * std::f64::consts::PI).ln();

        Ok(data_fit + complexity + normalization)
    }

    /// Get number of training samples
    pub fn n_train_samples(&self) -> usize {
        self.x_train.as_ref().map_or(0, |x| x.nrows())
    }
}

/// Cholesky decomposition: A = L L^T where L is lower triangular
fn cholesky_decomposition(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err("Matrix must be square".to_string());
    }

    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;

            if j == i {
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let val = a[[j, j]] - sum;
                if val <= 0.0 {
                    return Err(format!(
                        "Matrix is not positive definite (diagonal {} = {})",
                        j, val
                    ));
                }
                l[[j, j]] = val.sqrt();
            } else {
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Solve L x = b where L is lower triangular
fn solve_lower_triangular(l: &Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
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

/// Solve U x = b where U is upper triangular
fn solve_upper_triangular(u: &Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = u.nrows();
    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += u[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / u[[i, i]];
    }

    Ok(x)
}

/// Solve L X = B where L is lower triangular and B is a matrix
fn solve_lower_triangular_matrix(l: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = l.nrows();
    let m = b.ncols();
    let mut x = Array2::zeros((n, m));

    for col in 0..m {
        let b_col = b.column(col).to_owned();
        let x_col = solve_lower_triangular(l, &b_col)?;
        for row in 0..n {
            x[[row, col]] = x_col[row];
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian_process::kernel::SquaredExponential;
    use crate::gaussian_process::prior::ZeroPrior;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_gp_fit_predict() {
        let kernel = SquaredExponential::new(1.0, 1.0);
        let prior = ZeroPrior::new();
        let mut gp = GaussianProcess::new(kernel, prior, 0.01);

        // Simple training data
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Operation failed");
        let y_train = array![0.0, 1.0, 0.0];

        gp.fit(&x_train, &y_train).expect("Operation failed");

        // Predict at training points
        let predictions = gp.predict(&x_train).expect("Operation failed");

        // Should be close to training values
        for i in 0..3 {
            assert!((predictions[i] - y_train[i]).abs() < 0.1);
        }
    }

    #[test]
    fn test_gp_uncertainty() {
        let kernel = SquaredExponential::new(1.0, 1.0);
        let prior = ZeroPrior::new();
        let mut gp = GaussianProcess::new(kernel, prior, 0.01);

        let x_train = Array2::from_shape_vec((2, 1), vec![0.0, 2.0]).expect("Operation failed");
        let y_train = array![1.0, -1.0];

        gp.fit(&x_train, &y_train).expect("Operation failed");

        // Predict at interpolation point
        let x_test = Array2::from_shape_vec((1, 1), vec![1.0]).expect("Operation failed");
        let (_mean, std) = gp.predict_with_std(&x_test).expect("Operation failed");

        // Uncertainty should be positive and reasonable
        assert!(std[0] > 0.0);
        assert!(std[0] < 2.0); // Not too large
    }
}
