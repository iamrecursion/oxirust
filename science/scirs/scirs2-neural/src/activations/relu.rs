//! ReLU activation function implementations including ReLU, LeakyReLU, and ELU
//!
//! This module provides SIMD-accelerated implementations for f32/f64 1D arrays,
//! with automatic fallback to generic implementations for other types/dimensions.

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, ArrayView1, Ix1, IxDyn, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
/// Rectified Linear Unit (ReLU) activation function.
///
/// The ReLU function returns the input if it's positive, and 0 otherwise:
/// f(x) = max(0, x)
/// # Examples
/// ```
/// use scirs2_neural::activations::ReLU;
/// use scirs2_neural::activations::Activation;
/// use scirs2_core::ndarray::Array;
/// let relu = ReLU::new();
/// let input = Array::from_vec(vec![1.0, -1.0, 2.0, -2.0]).into_dyn();
/// let output = relu.forward(&input).expect("Operation failed");
/// assert_eq!(output, Array::from_vec(vec![1.0, 0.0, 2.0, 0.0]).into_dyn());
#[derive(Debug, Clone, Copy)]
pub struct ReLU {
    /// Alpha parameter for leaky ReLU.
    /// If alpha = 0, it's a standard ReLU.
    /// If alpha > 0, it's a leaky ReLU.
    alpha: f64,
}
impl ReLU {
    /// Create a new ReLU activation function.
    pub fn new() -> Self {
        Self { alpha: 0.0 }
    }
    /// Create a new Leaky ReLU activation function with given alpha.
    pub fn leaky(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Try SIMD path for f64 arrays
    #[inline]
    fn try_simd_f64<F: Float + NumAssign>(
        &self,
        input: &Array<F, IxDyn>,
        alpha: F,
    ) -> Option<Array<F, IxDyn>> {
        // Check if F is f64 using size_of
        if std::mem::size_of::<F>() != std::mem::size_of::<f64>() {
            return None;
        }

        // SAFETY: We've verified the type size matches f64
        // Convert via raw pointer cast
        unsafe {
            let ptr = input.as_ptr() as *const f64;
            let len = input.len();
            let slice = std::slice::from_raw_parts(ptr, len);
            let arr1d = ArrayView1::from(slice);

            let alpha_f64 = *(&alpha as *const F as *const f64);
            let result = if alpha_f64 == 0.0 {
                f64::simd_relu(&arr1d)
            } else {
                f64::simd_leaky_relu(&arr1d, alpha_f64)
            };

            // Convert back to F
            let result_ptr = result.as_ptr() as *const F;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.len());
            let result_dyn =
                Array::from_shape_vec(IxDyn(&[result.len()]), result_slice.to_vec()).ok()?;

            Some(result_dyn)
        }
    }

    /// Try SIMD path for f32 arrays
    #[inline]
    fn try_simd_f32<F: Float + NumAssign>(
        &self,
        input: &Array<F, IxDyn>,
        alpha: F,
    ) -> Option<Array<F, IxDyn>> {
        // Check if F is f32 using size_of
        if std::mem::size_of::<F>() != std::mem::size_of::<f32>() {
            return None;
        }

        // SAFETY: We've verified the type size matches f32
        unsafe {
            let ptr = input.as_ptr() as *const f32;
            let len = input.len();
            let slice = std::slice::from_raw_parts(ptr, len);
            let arr1d = ArrayView1::from(slice);

            let alpha_f32 = *(&alpha as *const F as *const f32);
            let result = if alpha_f32 == 0.0 {
                f32::simd_relu(&arr1d)
            } else {
                f32::simd_leaky_relu(&arr1d, alpha_f32)
            };

            // Convert back to F
            let result_ptr = result.as_ptr() as *const F;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.len());
            let result_dyn =
                Array::from_shape_vec(IxDyn(&[result.len()]), result_slice.to_vec()).ok()?;

            Some(result_dyn)
        }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug + NumAssign> Activation<F> for ReLU {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;

        // SIMD fast path for 1D f32/f64 arrays
        if input.ndim() == 1 {
            // Try f64 SIMD path
            if let Some(result) = self.try_simd_f64(input, alpha) {
                return Ok(result);
            }
            // Try f32 SIMD path
            if let Some(result) = self.try_simd_f32(input, alpha) {
                return Ok(result);
            }
        }

        // Generic fallback for all other cases
        let mut output = input.clone();
        let zero = F::zero();
        Zip::from(&mut output).for_each(|x| {
            if *x < zero {
                *x = alpha * *x;
            }
        });
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        output: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let one = F::one();
        let zero = F::zero();
        let alpha = F::from(self.alpha).ok_or_else(|| {
            NeuralError::InferenceError(
                "Could not convert alpha to the required float type".to_string(),
            )
        })?;
        // Create derivative mask based on the output (1 where x > 0, alpha where x <= 0)
        let mut mask = Array::from_elem(output.dim(), one);
        Zip::from(&mut mask).and(output).for_each(|mask_val, &out| {
            if out <= zero {
                *mask_val = alpha;
            }
        });
        // Multiply element-wise with the gradient from the next layer
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(&mask)
            .for_each(|grad_in, &grad_out, &mask_val| {
                *grad_in = grad_out * mask_val;
            });
        Ok(grad_input)
    }
}

/// Leaky Rectified Linear Unit (LeakyReLU) activation function.
/// The LeakyReLU function is similar to ReLU but allows a small, non-zero
/// gradient when the unit is not active:
/// f(x) = max(alpha*x, x) or f(x) = x if x > 0, alpha*x otherwise
/// use scirs2_neural::activations::LeakyReLU;
/// let leaky_relu = LeakyReLU::new(0.01);
/// let output = leaky_relu.forward(&input).expect("Operation failed");
/// assert_eq!(output, Array::from_vec(vec![1.0, -0.01, 2.0, -0.02]).into_dyn());
pub struct LeakyReLU {
    /// Alpha parameter for leaky ReLU (small positive value)
    alpha: f64,
}

impl LeakyReLU {
    /// Create a new LeakyReLU activation function with given alpha.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01) // Common default for LeakyReLU
    }
}

impl<F: Float + Debug + NumAssign> Activation<F> for LeakyReLU {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Use ReLU implementation with the alpha parameter
        let relu = ReLU::leaky(self.alpha);
        relu.forward(input)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        output: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Use ReLU implementation with the alpha parameter
        let relu = ReLU::leaky(self.alpha);
        relu.backward(grad_output, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_relu_simd_f64_1d() {
        let relu = ReLU::new();
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = relu.forward(&input).expect("Operation failed");
        let expected = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]).into_dyn();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_simd_f32_1d() {
        let relu = ReLU::new();
        let input = Array::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = relu.forward(&input).expect("Operation failed");
        let expected = Array::from_vec(vec![0.0f32, 0.0, 0.0, 1.0, 2.0]).into_dyn();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_leaky_relu_simd_f64_1d() {
        let relu = ReLU::leaky(0.01);
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = relu.forward(&input).expect("Operation failed");
        let expected = Array::from_vec(vec![-0.02, -0.01, 0.0, 1.0, 2.0]).into_dyn();
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_leaky_relu_simd_f32_1d() {
        let relu = ReLU::leaky(0.01);
        let input = Array::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = relu.forward(&input).expect("Operation failed");
        let expected = Array::from_vec(vec![-0.02f32, -0.01, 0.0, 1.0, 2.0]).into_dyn();
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_relu_large_array_f64() {
        // Test with large array to verify SIMD path is used
        let relu = ReLU::new();
        let input: Vec<f64> = (0..10000).map(|i| i as f64 - 5000.0).collect();
        let input_arr = Array::from_vec(input.clone()).into_dyn();
        let output = relu.forward(&input_arr).expect("Operation failed");

        // Verify correctness
        for (i, &val) in output.iter().enumerate() {
            let expected = if input[i] > 0.0 { input[i] } else { 0.0 };
            assert!((val - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_leaky_relu_large_array_f32() {
        // Test with large array to verify SIMD path is used
        let relu = ReLU::leaky(0.01);
        let input: Vec<f32> = (0..10000).map(|i| i as f32 - 5000.0).collect();
        let input_arr = Array::from_vec(input.clone()).into_dyn();
        let output = relu.forward(&input_arr).expect("Operation failed");

        // Verify correctness
        for (i, &val) in output.iter().enumerate() {
            let expected = if input[i] > 0.0 {
                input[i]
            } else {
                0.01 * input[i]
            };
            assert!((val - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_relu_2d_fallback() {
        // Test that 2D arrays still work (using generic fallback)
        let relu = ReLU::new();
        let input = Array::from_shape_vec((2, 3), vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            .expect("Operation failed")
            .into_dyn();
        let output = relu.forward(&input).expect("Operation failed");
        let expected = Array::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
            .expect("Operation failed")
            .into_dyn();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_leaky_relu_wrapper() {
        // Test LeakyReLU wrapper struct
        let leaky_relu = LeakyReLU::new(0.01);
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = leaky_relu.forward(&input).expect("Operation failed");
        let expected = Array::from_vec(vec![-0.02, -0.01, 0.0, 1.0, 2.0]).into_dyn();
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_relu_backward() {
        let relu = ReLU::new();
        let output = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]).into_dyn();
        let grad_output = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]).into_dyn();
        let grad_input = relu
            .backward(&grad_output, &output)
            .expect("Operation failed");
        let expected = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0]).into_dyn();
        assert_eq!(grad_input, expected);
    }

    #[test]
    fn test_leaky_relu_backward() {
        let relu = ReLU::leaky(0.01);
        let output = Array::from_vec(vec![-0.02, -0.01, 0.0, 1.0, 2.0]).into_dyn();
        let grad_output = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]).into_dyn();
        let grad_input = relu
            .backward(&grad_output, &output)
            .expect("Operation failed");
        let expected = Array::from_vec(vec![0.01, 0.01, 0.01, 1.0, 1.0]).into_dyn();
        for (a, b) in grad_input.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
