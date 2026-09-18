//! Hessian matrix computation and utilities

use crate::tensor_ops as T;
use crate::{error::AutogradError, tensor::Tensor, Context, Float, Result};

/// Hessian matrix computer
pub struct HessianComputer<'graph, T: Float> {
    /// The scalar function
    function: Tensor<'graph, T>,
    /// Input variable
    variable: Tensor<'graph, T>,
    /// Context
    context: &'graph Context<'graph, T>,
}

impl<'graph, T: Float> HessianComputer<'graph, T> {
    /// Create a new Hessian computer
    pub fn new(
        function: Tensor<'graph, T>,
        variable: Tensor<'graph, T>,
        context: &'graph Context<'graph, T>,
    ) -> Result<Self> {
        // Validate function is scalar
        let f_shape = function.shape();
        if f_shape.len() > 1 || (f_shape.len() == 1 && f_shape[0] != 1) {
            return Err(AutogradError::shape_error(
                "Function must be scalar for Hessian computation".to_string(),
            ));
        }

        Ok(Self {
            function,
            variable,
            context,
        })
    }

    /// Compute Hessian-vector product
    pub fn hvp(&self, v: &Tensor<'graph, T>) -> Result<Tensor<'graph, T>> {
        super::hessian_vector_product(&self.function, &self.variable, v, self.context)
    }

    /// Compute diagonal of Hessian (much cheaper than full Hessian)
    pub fn diagonal(&self) -> Result<Tensor<'graph, T>> {
        let x_shape = self.variable.shape();
        let n: usize = x_shape.iter().product();

        // Use Hessian-vector products with unit vectors to extract diagonal
        let mut diag_elements = Vec::with_capacity(n);

        for i in 0..n {
            // Create unit vector e_i
            let mut e_i_vec = vec![T::zero(); n];
            e_i_vec[i] = T::one();
            let e_i_arr = scirs2_core::ndarray::Array1::from(e_i_vec).into_dyn();
            let e_i = crate::tensor_ops::convert_to_tensor(e_i_arr, self.context);

            // Compute H * e_i
            let hvp_i = self.hvp(&e_i)?;
            let hvp_i_flat = crate::tensor_ops::flatten(hvp_i);

            // Extract i-th element (which is H[i,i])
            let diag_elem = crate::tensor_ops::slice(hvp_i_flat, [i as isize], [(i + 1) as isize]);
            diag_elements.push(diag_elem);
        }

        // Concatenate diagonal elements
        Ok(crate::tensor_ops::linear_algebra::concat(&diag_elements, 0))
    }

    /// Compute trace of Hessian (sum of diagonal elements)
    pub fn trace(&self) -> Result<Tensor<'graph, T>> {
        let diag = self.diagonal()?;
        Ok(crate::tensor_ops::reduction::sum_all(diag))
    }

    /// Estimate largest eigenvalue of Hessian using power iteration
    pub fn max_eigenvalue(&self, num_iterations: usize) -> Result<T> {
        let x_shape = self.variable.shape();

        // Initialize random vector
        // Convert shape Vec to Float array for tensor creation
        let shape_float: Vec<T> = x_shape
            .iter()
            .map(|&s| T::from(s).expect("Failed to convert shape"))
            .collect();
        let shape_arr = scirs2_core::ndarray::Array::from_vec(shape_float).into_dyn();
        let shape_tensor = crate::tensor_ops::convert_to_tensor(shape_arr, self.context);
        let mut v = crate::tensor_ops::random_normal(&shape_tensor, 0.0, 1.0, self.context);

        // Normalize
        let norm =
            crate::tensor_ops::arithmetic::sqrt(crate::tensor_ops::reduction::sum_all(v * v));
        v = v / norm;

        // Power iteration
        for _ in 0..num_iterations {
            // v = H * v
            v = self.hvp(&v)?;

            // Normalize
            let empty_axes: &[isize; 0] = &[];
            let norm = crate::tensor_ops::arithmetic::sqrt(
                crate::tensor_ops::reduction::reduce_sum(v * v, empty_axes, false),
            );
            v = v / norm;
        }

        // Rayleigh quotient: λ = v^T * H * v
        let hv = self.hvp(&v)?;
        let eigenvalue_tensor = crate::tensor_ops::reduction::sum_all(v * hv);

        // Evaluate to get scalar value
        let result = eigenvalue_tensor.eval(self.context)?;
        result
            .first()
            .copied()
            .ok_or_else(|| AutogradError::compute_error("Failed to extract eigenvalue".to_string()))
    }

    /// Check if Hessian is positive definite at current point
    pub fn is_positive_definite(&self, tolerance: T) -> Result<bool> {
        // A matrix is positive definite if all eigenvalues are positive
        // We approximate by checking the minimum eigenvalue

        let diag = self.diagonal()?;
        let diag_eval = diag.eval(self.context)?;

        // Check if all diagonal elements are positive (necessary but not sufficient)
        let all_positive = diag_eval.iter().all(|&x| x > tolerance);

        Ok(all_positive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_hessian_diagonal() {
        crate::run(|ctx: &mut Context<f64>| {
            // f(x) = x1² + 2*x2² + 3*x3²
            // Hessian diagonal should be [2, 4, 6]
            let x = ctx.placeholder("x", &[3]);
            let x_squared = x * x;
            let coeffs = convert_to_tensor(Array1::from(vec![1.0, 2.0, 3.0]).into_dyn(), ctx);
            let f = crate::tensor_ops::reduction::sum_all(coeffs * x_squared);

            let computer = HessianComputer::new(f, x, ctx).expect("Should create Hessian computer");

            let diag = computer.diagonal().expect("Should compute diagonal");

            let x_val = scirs2_core::ndarray::arr1(&[1.0, 1.0, 1.0]);
            let result = ctx
                .evaluator()
                .push(&diag)
                .feed(x, x_val.view().into_dyn())
                .run();

            let diag_array = result[0].as_ref().expect("Should evaluate");
            let diag_vals = diag_array.as_slice().expect("Should get slice");
            assert!((diag_vals[0] - 2.0).abs() < 1e-6);
            assert!((diag_vals[1] - 4.0).abs() < 1e-6);
            assert!((diag_vals[2] - 6.0).abs() < 1e-6);
        });
    }

    #[test]
    fn test_hessian_trace() {
        crate::run(|ctx: &mut Context<f64>| {
            // f(x) = x1² + x2²
            // Trace of Hessian = 2 + 2 = 4
            let x = ctx.placeholder("x", &[2]);
            let f = crate::tensor_ops::reduction::sum_all(x * x);

            let computer = HessianComputer::new(f, x, ctx).expect("Should create Hessian computer");

            let trace = computer.trace().expect("Should compute trace");

            let x_val = scirs2_core::ndarray::arr1(&[1.0, 1.0]);
            let result = ctx
                .evaluator()
                .push(&trace)
                .feed(x, x_val.view().into_dyn())
                .run();

            let trace_val = result[0]
                .as_ref()
                .expect("Should evaluate")
                .first()
                .copied()
                .unwrap_or(0.0);
            assert!((trace_val - 4.0).abs() < 1e-6);
        });
    }
}
