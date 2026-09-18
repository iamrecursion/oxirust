//! Vector-Jacobian product utilities

use crate::tensor_ops as T;
use crate::{tensor::Tensor, Context, Float, Result};

/// Compute VJP using custom cotangents
pub fn vjp_with_cotangent<'graph, T: Float>(
    outputs: &[Tensor<'graph, T>],
    inputs: &[Tensor<'graph, T>],
    cotangents: &[Tensor<'graph, T>],
    _ctx: &'graph Context<'graph, T>,
) -> Result<Vec<Tensor<'graph, T>>> {
    // Weight the outputs by cotangents
    let weighted: Vec<Tensor<'graph, T>> = outputs
        .iter()
        .zip(cotangents.iter())
        .map(|(out, cot)| *out * *cot)
        .collect();

    // Sum weighted outputs
    let total = weighted.iter().skip(1).fold(weighted[0], |acc, &x| acc + x);

    // Compute gradients (this is the VJP)
    let vjps = crate::tensor_ops::grad(&[total], inputs);

    Ok(vjps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_vjp_basic() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);

            // f(x) = [x1², x2²]
            let x0 = slice(x, [0_isize], [1_isize]);
            let x1 = slice(x, [1_isize], [2_isize]);
            let f = concat(&[x0 * x0, x1 * x1], 0);

            // Cotangent v = [1, 2]
            let v = convert_to_tensor(Array1::from(vec![1.0, 2.0]).into_dyn(), ctx);

            let vjp = vjp_with_cotangent(&[f], &[x], &[v], ctx).expect("Should compute VJP");

            // Evaluate at x = [3, 4]
            let x_val = scirs2_core::ndarray::arr1(&[3.0, 4.0]);
            let result = ctx
                .evaluator()
                .push(&vjp[0])
                .feed(x, x_val.view().into_dyn())
                .run();

            // v^T * J = [1, 2] * [[6, 0], [0, 8]] = [6, 16]
            let vjp_array = result[0].as_ref().expect("Should evaluate");
            let vjp_vals = vjp_array.as_slice().expect("Should get slice");
            assert!((vjp_vals[0] - 6.0).abs() < 1e-6);
            assert!((vjp_vals[1] - 16.0).abs() < 1e-6);
        });
    }
}
