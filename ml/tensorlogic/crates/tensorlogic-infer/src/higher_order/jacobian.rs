//! Jacobian matrix computation via finite differences.
//!
//! Computes `J[i,j] = ∂f_i/∂x_j` for vector-valued functions f: ℝ^n → ℝ^m.

use ndarray::{Array, ArrayD, IxDyn};

/// Finite difference method for numerical differentiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteDiffMethod {
    /// Forward differences: (f(x+ε) - f(x)) / ε — O(ε) error.
    Forward,
    /// Central differences: (f(x+ε) - f(x-ε)) / (2ε) — O(ε²) error, more accurate.
    Central,
}

/// Configuration for Jacobian computation.
#[derive(Debug, Clone)]
pub struct JacobianConfig {
    /// Step size for finite differences.
    pub epsilon: f64,
    /// Finite difference method.
    pub method: FiniteDiffMethod,
}

impl Default for JacobianConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            method: FiniteDiffMethod::Central,
        }
    }
}

impl JacobianConfig {
    /// Create a new configuration.
    pub fn new(epsilon: f64, method: FiniteDiffMethod) -> Self {
        Self { epsilon, method }
    }

    /// Builder: set epsilon.
    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Builder: set method.
    pub fn with_method(mut self, method: FiniteDiffMethod) -> Self {
        self.method = method;
        self
    }
}

/// Errors that can occur during Jacobian computation.
#[derive(Debug, thiserror::Error)]
pub enum JacobianError {
    /// Input tensor must be 1-D (flat vector).
    #[error("Input must be 1-D, got shape {0:?}")]
    NonFlatInput(Vec<usize>),

    /// Epsilon must be strictly positive.
    #[error("Epsilon must be positive, got {0}")]
    InvalidEpsilon(f64),

    /// The wrapped function returned an error.
    #[error("Function evaluation failed: {0}")]
    EvalFailed(String),

    /// Input has zero elements.
    #[error("Empty input")]
    EmptyInput,
}

/// Computes Jacobian matrices via finite differences.
///
/// For f: ℝ^n → ℝ^m the Jacobian J has shape `[m, n]` where `J[i,j] = ∂f_i/∂x_j`.
pub struct JacobianComputer {
    config: JacobianConfig,
}

impl JacobianComputer {
    /// Create a new `JacobianComputer` with the given configuration.
    pub fn new(config: JacobianConfig) -> Self {
        Self { config }
    }

    /// Compute the Jacobian matrix for a vector-valued function.
    ///
    /// `input` must be 1-D. Returns a 2-D array of shape `[m, n]`.
    pub fn compute<F>(&self, input: &ArrayD<f64>, f: F) -> Result<ArrayD<f64>, JacobianError>
    where
        F: Fn(&ArrayD<f64>) -> Result<ArrayD<f64>, String>,
    {
        self.validate(input)?;

        let n = input.len();
        let eps = self.config.epsilon;

        // Get output size m via one evaluation.
        let f0_for_shape = f(input).map_err(JacobianError::EvalFailed)?;
        let m = f0_for_shape.len();

        let mut jacobian_flat = vec![0.0f64; m * n];

        for j in 0..n {
            let f_plus = {
                let mut x_plus = input.clone();
                x_plus[j] += eps;
                f(&x_plus).map_err(JacobianError::EvalFailed)?
            };

            let col: Vec<f64> = match self.config.method {
                FiniteDiffMethod::Central => {
                    let mut x_minus = input.clone();
                    x_minus[j] -= eps;
                    let f_minus = f(&x_minus).map_err(JacobianError::EvalFailed)?;
                    f_plus
                        .iter()
                        .zip(f_minus.iter())
                        .map(|(p, m_val)| (p - m_val) / (2.0 * eps))
                        .collect()
                }
                FiniteDiffMethod::Forward => {
                    let f_base = f(input).map_err(JacobianError::EvalFailed)?;
                    f_plus
                        .iter()
                        .zip(f_base.iter())
                        .map(|(p, b)| (p - b) / eps)
                        .collect()
                }
            };

            // Fill column j of [m x n] stored row-major.
            for (i, val) in col.into_iter().enumerate() {
                jacobian_flat[i * n + j] = val;
            }
        }

        let jacobian = Array::from_shape_vec(IxDyn(&[m, n]), jacobian_flat)
            .map_err(|e| JacobianError::EvalFailed(e.to_string()))?;

        Ok(jacobian)
    }

    /// Compute gradient for a scalar-output function f: ℝ^n → ℝ.
    ///
    /// Returns a 1-D array of shape `[n]`.
    pub fn compute_gradient<F>(
        &self,
        input: &ArrayD<f64>,
        f: F,
    ) -> Result<ArrayD<f64>, JacobianError>
    where
        F: Fn(&ArrayD<f64>) -> Result<f64, String>,
    {
        let scalar_f = move |x: &ArrayD<f64>| -> Result<ArrayD<f64>, String> {
            let v = f(x)?;
            Array::from_shape_vec(IxDyn(&[1]), vec![v]).map_err(|e| e.to_string())
        };

        let jac = self.compute(input, scalar_f)?;
        // jac has shape [1, n]; squeeze to [n].
        let n = input.len();
        let grad_flat: Vec<f64> = jac.iter().cloned().collect();
        let grad = Array::from_shape_vec(IxDyn(&[n]), grad_flat)
            .map_err(|e| JacobianError::EvalFailed(e.to_string()))?;
        Ok(grad)
    }

    /// Check whether computed and analytical Jacobians agree within `tol`.
    ///
    /// Uses element-wise relative error: |computed - analytical| / (|analytical| + 1e-8).
    pub fn check_accuracy(computed: &ArrayD<f64>, analytical: &ArrayD<f64>, tol: f64) -> bool {
        if computed.shape() != analytical.shape() {
            return false;
        }
        computed.iter().zip(analytical.iter()).all(|(c, a)| {
            let rel = (c - a).abs() / (a.abs() + 1e-8);
            rel <= tol
        })
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn validate(&self, input: &ArrayD<f64>) -> Result<(), JacobianError> {
        if self.config.epsilon <= 0.0 {
            return Err(JacobianError::InvalidEpsilon(self.config.epsilon));
        }
        if input.ndim() != 1 {
            return Err(JacobianError::NonFlatInput(input.shape().to_vec()));
        }
        if input.is_empty() {
            return Err(JacobianError::EmptyInput);
        }
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn vec1d(data: &[f64]) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap()
    }

    #[test]
    fn test_jacobian_identity_function() {
        // f(x) = x, J = I
        let config = JacobianConfig::default();
        let comp = JacobianComputer::new(config);
        let x = vec1d(&[1.0, 2.0, 3.0]);
        let jac = comp
            .compute(&x, |v| Ok(v.clone()))
            .expect("jacobian identity");
        assert_eq!(jac.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (jac[[i, j]] - expected).abs();
                assert!(
                    err < 1e-8,
                    "J[{},{}]={} expected {}",
                    i,
                    j,
                    jac[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_jacobian_linear_f2x() {
        // f(x) = 2x, J = 2*I
        let comp = JacobianComputer::new(JacobianConfig::default());
        let x = vec1d(&[1.0, -1.0]);
        let jac = comp
            .compute(&x, |v| {
                let out: Vec<f64> = v.iter().map(|&a| 2.0 * a).collect();
                Ok(Array::from_shape_vec(IxDyn(&[out.len()]), out).unwrap())
            })
            .expect("jacobian 2x");
        assert!((jac[[0, 0]] - 2.0).abs() < 1e-8);
        assert!((jac[[1, 1]] - 2.0).abs() < 1e-8);
        assert!(jac[[0, 1]].abs() < 1e-8);
        assert!(jac[[1, 0]].abs() < 1e-8);
    }

    #[test]
    fn test_jacobian_quadratic_gradient() {
        // f(x) = x0^2 + x1^2, ∂f/∂x0 = 2*x0
        let comp = JacobianComputer::new(JacobianConfig::default());
        let x = vec1d(&[3.0, 4.0]);
        let grad = comp
            .compute_gradient(&x, |v| Ok(v[0] * v[0] + v[1] * v[1]))
            .expect("gradient quadratic");
        assert!((grad[0] - 6.0).abs() < 1e-8, "grad[0]={}", grad[0]);
        assert!((grad[1] - 8.0).abs() < 1e-8, "grad[1]={}", grad[1]);
    }

    #[test]
    fn test_jacobian_output_shape_2x3() {
        // f: ℝ^3 → ℝ^2, J should be [2,3]
        let comp = JacobianComputer::new(JacobianConfig::default());
        let x = vec1d(&[1.0, 2.0, 3.0]);
        let jac = comp
            .compute(&x, |v| {
                let out = vec![v[0] + v[1], v[1] + v[2]];
                Ok(Array::from_shape_vec(IxDyn(&[2]), out).unwrap())
            })
            .expect("jacobian 2x3");
        assert_eq!(jac.shape(), &[2, 3]);
    }

    #[test]
    fn test_jacobian_central_more_accurate_than_forward() {
        // For a nonlinear function, central diff should yield smaller error vs analytical.
        let x = vec1d(&[1.0]);
        // f(x) = x^3, f'(x) = 3x^2 = 3 at x=1
        let analytical_grad = vec1d(&[3.0]);

        let comp_central =
            JacobianComputer::new(JacobianConfig::new(1e-4, FiniteDiffMethod::Central));
        let comp_forward =
            JacobianComputer::new(JacobianConfig::new(1e-4, FiniteDiffMethod::Forward));

        let central_grad = comp_central
            .compute_gradient(&x, |v| Ok(v[0].powi(3)))
            .unwrap();
        let forward_grad = comp_forward
            .compute_gradient(&x, |v| Ok(v[0].powi(3)))
            .unwrap();

        let central_err = (central_grad[0] - analytical_grad[0]).abs();
        let forward_err = (forward_grad[0] - analytical_grad[0]).abs();
        assert!(
            central_err <= forward_err,
            "central_err={} should be <= forward_err={}",
            central_err,
            forward_err
        );
    }

    #[test]
    fn test_jacobian_no_nan() {
        // Smooth function should produce no NaN entries.
        let comp = JacobianComputer::new(JacobianConfig::default());
        let x = vec1d(&[0.5, 1.5, 2.5]);
        let jac = comp
            .compute(&x, |v| {
                let out: Vec<f64> = v.iter().map(|&a| a.sin()).collect();
                Ok(Array::from_shape_vec(IxDyn(&[out.len()]), out).unwrap())
            })
            .expect("jacobian sin");
        assert!(jac.iter().all(|v| !v.is_nan()), "Jacobian contains NaN");
    }

    #[test]
    fn test_jacobian_check_accuracy() {
        // f(x) = 2*x, analytical J = 2*I_3
        let comp = JacobianComputer::new(JacobianConfig::default());
        let x = vec1d(&[1.0, 2.0, 3.0]);
        let computed = comp
            .compute(&x, |v| {
                let out: Vec<f64> = v.iter().map(|&a| 2.0 * a).collect();
                Ok(Array::from_shape_vec(IxDyn(&[out.len()]), out).unwrap())
            })
            .unwrap();

        // Build analytical 2*I_3
        let mut analytical_flat = vec![0.0f64; 9];
        for i in 0..3 {
            analytical_flat[i * 3 + i] = 2.0;
        }
        let analytical = Array::from_shape_vec(IxDyn(&[3, 3]), analytical_flat).unwrap();

        assert!(JacobianComputer::check_accuracy(
            &computed,
            &analytical,
            1e-6
        ));
    }

    #[test]
    fn test_jacobian_flat_input_required_error() {
        let comp = JacobianComputer::new(JacobianConfig::default());
        let x_2d = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0; 4]).unwrap();
        let result = comp.compute(&x_2d, |v| Ok(v.clone()));
        assert!(matches!(result, Err(JacobianError::NonFlatInput(_))));
    }

    #[test]
    fn test_jacobian_vector_valued() {
        // f: ℝ^3 → ℝ^2 with f = [x0+x1, x1+x2]
        // J = [[1,1,0],[0,1,1]]
        let comp = JacobianComputer::new(JacobianConfig::default());
        let x = vec1d(&[1.0, 2.0, 3.0]);
        let jac = comp
            .compute(&x, |v| {
                Ok(Array::from_shape_vec(IxDyn(&[2]), vec![v[0] + v[1], v[1] + v[2]]).unwrap())
            })
            .unwrap();

        assert!((jac[[0, 0]] - 1.0).abs() < 1e-7);
        assert!((jac[[0, 1]] - 1.0).abs() < 1e-7);
        assert!(jac[[0, 2]].abs() < 1e-7);
        assert!(jac[[1, 0]].abs() < 1e-7);
        assert!((jac[[1, 1]] - 1.0).abs() < 1e-7);
        assert!((jac[[1, 2]] - 1.0).abs() < 1e-7);
    }
}
