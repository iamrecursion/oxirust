//! Mish activation function implementation

use crate::activations::Activation;
use crate::error::Result;
use scirs2_core::ndarray::{Array, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
/// Mish activation function.
///
/// Mish is defined as:
/// f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
/// It was proposed in "Mish: A Self Regularized Non-Monotonic Activation Function"
/// by Diganta Misra, and has been shown to work well in deep networks.
/// # Examples
/// ```
/// use scirs2_neural::activations::Mish;
/// use scirs2_neural::activations::Activation;
/// use scirs2_core::ndarray::Array;
/// let mish = Mish::new();
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = mish.forward(&input).expect("Operation failed");
#[derive(Debug, Clone, Copy)]
pub struct Mish;
impl Mish {
    /// Create a new Mish activation function.
    pub fn new() -> Self {
        Self
    }
}
impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug + NumAssign> Activation<F> for Mish {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();
        // Compute x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        Zip::from(&mut output).for_each(|x| {
            // Compute softplus(x) = ln(1 + e^x)
            // Use a numerically stable version for large values
            let sp = if *x > F::from(20.0).expect("Failed to convert constant to float") {
                // For large x, softplus(x) ≈ x
                *x
            } else {
                (F::one() + x.exp()).ln()
            };
            // Apply tanh(softplus(x))
            *x *= sp.tanh();
        });
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // We need to compute the derivative of Mish: d(mish)/dx
        // Note: Like GELU and Swish, we treat the second parameter as input
        // for gradient computation accuracy
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        // Mish(x) = x * tanh(softplus(x))
        // d(mish)/dx = tanh(sp) + x * sech²(sp) * sigmoid(x)
        // where sp = softplus(x) = ln(1 + e^x) and sigmoid(x) = e^x / (1 + e^x)
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|grad_in, &grad_out, &x| {
                // Compute softplus(x) = ln(1 + e^x) with numerical stability
                let sp = if x > F::from(20.0).expect("Failed to convert constant to float") {
                    x // For large x, softplus(x) ≈ x
                } else {
                    (F::one() + x.exp()).ln()
                };
                let tanh_sp = sp.tanh();
                let sech_sp_sq = F::one() - tanh_sp * tanh_sp; // sech²(sp)

                // sigmoid(x) = e^x / (1 + e^x) = 1 / (1 + e^(-x))
                let sigmoid_x = F::one() / (F::one() + (-x).exp());

                // d(mish)/dx = tanh(sp) + x * sech²(sp) * sigmoid(x)
                let dmish_dx = tanh_sp + x * sech_sp_sq * sigmoid_x;

                // Multiply by the gradient from the next layer
                *grad_in = grad_out * dmish_dx;
            });
        Ok(grad_input)
    }
}
