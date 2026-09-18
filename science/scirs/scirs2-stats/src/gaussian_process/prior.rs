//! Prior mean functions for Gaussian Processes
//!
//! Prior functions define the mean function of the GP before observing any data.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Trait for prior mean functions
pub trait Prior: Clone + Send + Sync {
    /// Compute prior mean at a point
    fn compute(&self, x: &ArrayView1<f64>) -> f64;

    /// Compute prior mean for multiple points
    fn compute_vector(&self, x: &Array2<f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut means = Array1::zeros(n);

        for i in 0..n {
            means[i] = self.compute(&x.row(i));
        }

        means
    }

    /// Get prior parameters
    fn get_params(&self) -> Vec<f64>;

    /// Set prior parameters
    fn set_params(&mut self, params: &[f64]);
}

/// Constant prior: m(x) = c
///
/// This is the most common choice, assuming the function has a constant mean.
#[derive(Debug, Clone)]
pub struct ConstantPrior {
    pub constant: f64,
}

impl ConstantPrior {
    /// Create a new constant prior
    pub fn new(constant: f64) -> Self {
        Self { constant }
    }
}

impl Default for ConstantPrior {
    fn default() -> Self {
        Self { constant: 0.0 }
    }
}

impl Prior for ConstantPrior {
    fn compute(&self, _x: &ArrayView1<f64>) -> f64 {
        self.constant
    }

    fn get_params(&self) -> Vec<f64> {
        vec![self.constant]
    }

    fn set_params(&mut self, params: &[f64]) {
        if !params.is_empty() {
            self.constant = params[0];
        }
    }
}

/// Zero prior: m(x) = 0
///
/// A special case of constant prior with c = 0.
#[derive(Debug, Clone, Copy)]
pub struct ZeroPrior;

impl ZeroPrior {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ZeroPrior {
    fn default() -> Self {
        Self::new()
    }
}

impl Prior for ZeroPrior {
    fn compute(&self, _x: &ArrayView1<f64>) -> f64 {
        0.0
    }

    fn get_params(&self) -> Vec<f64> {
        vec![]
    }

    fn set_params(&mut self, _params: &[f64]) {
        // No parameters to set
    }
}

/// Linear prior: m(x) = a^T x + b
///
/// Assumes a linear trend in the data.
#[derive(Debug, Clone)]
pub struct LinearPrior {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
}

impl LinearPrior {
    /// Create a new linear prior
    pub fn new(coefficients: Vec<f64>, intercept: f64) -> Self {
        Self {
            coefficients,
            intercept,
        }
    }

    /// Create a zero linear prior with given dimensionality
    pub fn zeros(n_dims: usize) -> Self {
        Self {
            coefficients: vec![0.0; n_dims],
            intercept: 0.0,
        }
    }
}

impl Prior for LinearPrior {
    fn compute(&self, x: &ArrayView1<f64>) -> f64 {
        let mut result = self.intercept;
        for (i, &coef) in self.coefficients.iter().enumerate() {
            if i < x.len() {
                result += coef * x[i];
            }
        }
        result
    }

    fn get_params(&self) -> Vec<f64> {
        let mut params = self.coefficients.clone();
        params.push(self.intercept);
        params
    }

    fn set_params(&mut self, params: &[f64]) {
        if params.is_empty() {
            return;
        }

        let n_coef = self.coefficients.len();
        if params.len() >= n_coef {
            self.coefficients.copy_from_slice(&params[..n_coef]);
            if params.len() > n_coef {
                self.intercept = params[n_coef];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_constant_prior() {
        let prior = ConstantPrior::new(5.0);
        let x = array![1.0, 2.0, 3.0];

        assert_eq!(prior.compute(&x.view()), 5.0);
    }

    #[test]
    fn test_zero_prior() {
        let prior = ZeroPrior::new();
        let x = array![1.0, 2.0, 3.0];

        assert_eq!(prior.compute(&x.view()), 0.0);
    }

    #[test]
    fn test_linear_prior() {
        let prior = LinearPrior::new(vec![2.0, 3.0], 1.0);
        let x = array![1.0, 1.0];

        // 2.0 * 1.0 + 3.0 * 1.0 + 1.0 = 6.0
        assert_eq!(prior.compute(&x.view()), 6.0);
    }
}
