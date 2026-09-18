//! Jacobian matrix computation

use crate::tensor_ops as T;
use crate::{error::AutogradError, tensor::Tensor, Context, Float, Result};

/// Compute full Jacobian matrix
///
/// For function f: R^n -> R^m, computes the m x n Jacobian matrix.
/// Requires explicit output dimension `m` and input dimension `n` because
/// tensor shapes may not be available at graph-building time.
pub fn jacobian<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    // Try to infer dimensions from known shapes
    let f_shape = f.shape();
    let x_shape = x.shape();

    let m: usize = if f_shape.is_empty() {
        1
    } else {
        f_shape.iter().product::<usize>().max(1)
    };
    let n: usize = if x_shape.is_empty() {
        1
    } else {
        x_shape.iter().product::<usize>().max(1)
    };

    jacobian_with_dims(f, x, m, n, ctx)
}

/// Compute full Jacobian matrix with explicit dimensions.
///
/// For function f: R^n -> R^m, computes the m x n Jacobian matrix.
/// Use this when tensor shapes are not tracked at graph-building time.
pub fn jacobian_with_dims<'graph, T: Float>(
    f: &Tensor<'graph, T>,
    x: &Tensor<'graph, T>,
    m: usize,
    n: usize,
    _ctx: &'graph Context<'graph, T>,
) -> Result<Tensor<'graph, T>> {
    if m > 10000 || n > 10000 {
        eprintln!(
            "Warning: Computing large Jacobian {}x{}. Consider using JVP/VJP instead.",
            m, n
        );
    }

    let f_flat = crate::tensor_ops::flatten(*f);
    let mut jacobian_rows = Vec::with_capacity(m);

    // Compute gradient of each output with respect to input
    for i in 0..m {
        let f_i = crate::tensor_ops::slice(f_flat, [i as isize], [(i + 1) as isize]);
        let grad_i = crate::tensor_ops::grad(&[f_i], &[*x])[0];
        let grad_i_flat = crate::tensor_ops::flatten(grad_i);
        jacobian_rows.push(grad_i_flat);
    }

    Ok(crate::tensor_ops::linear_algebra::concat(&jacobian_rows, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;

    #[test]
    fn test_jacobian_2d() {
        crate::run(|ctx: &mut Context<f64>| {
            // f([x, y]) = [x², xy]
            // Jacobian = [[2x, 0], [y, x]]
            let x = ctx.placeholder("x", &[2]);
            let x0 = slice(x, [0_isize], [1_isize]);
            let x1 = slice(x, [1_isize], [2_isize]);

            let f0 = x0 * x0;
            let f1 = x0 * x1;
            let f = concat(&[f0, f1], 0);

            // f has 2 outputs, x has 2 inputs => 2x2 Jacobian
            let jac = jacobian_with_dims(&f, &x, 2, 2, ctx).expect("Should compute Jacobian");

            // Evaluate at x = [2, 3]
            let x_val = scirs2_core::ndarray::arr1(&[2.0, 3.0]);
            let result = ctx
                .evaluator()
                .push(&jac)
                .feed(x, x_val.view().into_dyn())
                .run();

            // Expected: [[4, 0], [3, 2]]
            let jac_array = result[0].as_ref().expect("Should evaluate");
            let jac_vals = jac_array.as_slice().expect("Should get slice");
            assert!((jac_vals[0] - 4.0).abs() < 1e-6); // ∂(x²)/∂x = 2x = 4
            assert!((jac_vals[1] - 0.0).abs() < 1e-6); // ∂(x²)/∂y = 0
            assert!((jac_vals[2] - 3.0).abs() < 1e-6); // ∂(xy)/∂x = y = 3
            assert!((jac_vals[3] - 2.0).abs() < 1e-6); // ∂(xy)/∂y = x = 2
        });
    }
}
