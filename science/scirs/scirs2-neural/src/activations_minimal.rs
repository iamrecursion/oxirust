//! Minimal activation functions without Layer trait dependencies
//!
//! This module provides SIMD-accelerated implementations for f32/f64 1D arrays,
//! with automatic fallback to generic implementations for other types/dimensions.

use crate::error::Result;
use scirs2_core::ndarray::{Array, ArrayView1, IxDyn, Zip};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;

/// Trait for activation functions
pub trait Activation<F> {
    /// Forward pass of the activation function
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>>;

    /// Backward pass of the activation function
    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>>;
}

/// GELU activation function
#[derive(Debug, Clone, Copy)]
pub struct GELU {
    fast: bool,
}

impl GELU {
    pub fn new() -> Self {
        Self { fast: false }
    }

    pub fn fast() -> Self {
        Self { fast: true }
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for GELU {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();

        if self.fast {
            let sqrt_2_over_pi =
                F::from(0.7978845608028654).expect("Failed to convert constant to float");
            let coeff = F::from(0.044715).expect("Failed to convert constant to float");
            let half = F::from(0.5).expect("Failed to convert constant to float");
            let one = F::one();

            Zip::from(&mut output).for_each(|x| {
                let x3 = *x * *x * *x;
                let inner = sqrt_2_over_pi * (*x + coeff * x3);
                *x = half * *x * (one + inner.tanh());
            });
        } else {
            let sqrt_pi_over_2 =
                F::from(1.2533141373155).expect("Failed to convert constant to float");
            let coeff = F::from(0.044715).expect("Failed to convert constant to float");
            let half = F::from(0.5).expect("Failed to convert constant to float");
            let one = F::one();

            Zip::from(&mut output).for_each(|x| {
                let x2 = *x * *x;
                let inner = sqrt_pi_over_2 * *x * (one + coeff * x2);
                *x = half * *x * (one + inner.tanh());
            });
        }

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        if self.fast {
            let sqrt_2_over_pi =
                F::from(0.7978845608028654).expect("Failed to convert constant to float");
            let coeff = F::from(0.044715).expect("Failed to convert constant to float");
            let half = F::from(0.5).expect("Failed to convert constant to float");
            let one = F::one();
            let three = F::from(3.0).expect("Failed to convert constant to float");

            Zip::from(&mut grad_input)
                .and(grad_output)
                .and(input)
                .for_each(|grad_in, &grad_out, &x| {
                    let x2 = x * x;
                    let x3 = x2 * x;
                    let inner = sqrt_2_over_pi * (x + coeff * x3);
                    let tanh_inner = inner.tanh();
                    let sech_sq = one - tanh_inner * tanh_inner;
                    let d_inner_dx = sqrt_2_over_pi * (one + three * coeff * x2);
                    let dgelu_dx = half * (one + tanh_inner) + half * x * sech_sq * d_inner_dx;
                    *grad_in = grad_out * dgelu_dx;
                });
        } else {
            let sqrt_pi_over_2 =
                F::from(1.2533141373155).expect("Failed to convert constant to float");
            let coeff = F::from(0.044715).expect("Failed to convert constant to float");
            let half = F::from(0.5).expect("Failed to convert constant to float");
            let one = F::one();
            let three = F::from(3.0).expect("Failed to convert constant to float");

            Zip::from(&mut grad_input)
                .and(grad_output)
                .and(input)
                .for_each(|grad_in, &grad_out, &x| {
                    let x2 = x * x;
                    let inner = sqrt_pi_over_2 * x * (one + coeff * x2);
                    let tanh_inner = inner.tanh();
                    let sech_sq = one - tanh_inner * tanh_inner;
                    let d_inner_dx = sqrt_pi_over_2 * (one + three * coeff * x2);
                    let dgelu_dx = half * (one + tanh_inner) + half * x * sech_sq * d_inner_dx;
                    *grad_in = grad_out * dgelu_dx;
                });
        }

        Ok(grad_input)
    }
}

/// Tanh activation function
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for Tanh {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();
        Zip::from(&mut output).for_each(|x| {
            *x = x.tanh();
        });
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|grad_in, &grad_out, &x| {
                let tanh_x = x.tanh();
                let derivative = F::one() - tanh_x * tanh_x;
                *grad_in = grad_out * derivative;
            });

        Ok(grad_input)
    }
}

/// Sigmoid activation function
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Activation<F> for Sigmoid {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut output = input.clone();
        let one = F::one();
        Zip::from(&mut output).for_each(|x| {
            *x = one / (one + (-*x).exp());
        });
        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        let one = F::one();

        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|grad_in, &grad_out, &x| {
                let sigmoid_x = one / (one + (-x).exp());
                let derivative = sigmoid_x * (one - sigmoid_x);
                *grad_in = grad_out * derivative;
            });

        Ok(grad_input)
    }
}

/// ReLU activation function
#[derive(Debug, Clone, Copy)]
pub struct ReLU {
    alpha: f64,
}

impl ReLU {
    pub fn new() -> Self {
        Self { alpha: 0.0 }
    }

    pub fn leaky(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Try SIMD path for f64 arrays
    #[inline]
    fn try_simd_f64<F: Float>(&self, input: &Array<F, IxDyn>, alpha: F) -> Option<Array<F, IxDyn>> {
        // Check if F is f64 using size_of
        if std::mem::size_of::<F>() != std::mem::size_of::<f64>() {
            return None;
        }

        // SAFETY: We've verified the type size matches f64
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
    fn try_simd_f32<F: Float>(&self, input: &Array<F, IxDyn>, alpha: F) -> Option<Array<F, IxDyn>> {
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

impl<F: Float + Debug> Activation<F> for ReLU {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let zero = F::zero();
        let alpha = F::from(self.alpha).unwrap_or(zero);

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
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        let mut grad_input = Array::zeros(grad_output.raw_dim());
        let zero = F::zero();
        let one = F::one();
        let alpha = F::from(self.alpha).unwrap_or(zero);

        Zip::from(&mut grad_input)
            .and(grad_output)
            .and(input)
            .for_each(|grad_in, &grad_out, &x| {
                let derivative = if x > zero { one } else { alpha };
                *grad_in = grad_out * derivative;
            });

        Ok(grad_input)
    }
}

/// Softmax activation function
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    axis: isize,
}

impl Softmax {
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }

    /// Try SIMD path for f64 arrays
    #[inline]
    fn try_simd_f64<F: Float>(&self, input: &Array<F, IxDyn>) -> Option<Array<F, IxDyn>> {
        // Check if F is f64 using size_of
        if std::mem::size_of::<F>() != std::mem::size_of::<f64>() {
            return None;
        }

        // SAFETY: We've verified the type size matches f64
        unsafe {
            let ptr = input.as_ptr() as *const f64;
            let len = input.len();
            let slice = std::slice::from_raw_parts(ptr, len);
            let arr1d = ArrayView1::from(slice);

            let result = f64::simd_softmax(&arr1d);

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
    fn try_simd_f32<F: Float>(&self, input: &Array<F, IxDyn>) -> Option<Array<F, IxDyn>> {
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

            let result = f32::simd_softmax(&arr1d);

            // Convert back to F
            let result_ptr = result.as_ptr() as *const F;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.len());
            let result_dyn =
                Array::from_shape_vec(IxDyn(&[result.len()]), result_slice.to_vec()).ok()?;

            Some(result_dyn)
        }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl<F: Float + Debug> Activation<F> for Softmax {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // SIMD fast path for 1D f32/f64 arrays
        if input.ndim() == 1 {
            // Try f64 SIMD path
            if let Some(result) = self.try_simd_f64(input) {
                return Ok(result);
            }
            // Try f32 SIMD path
            if let Some(result) = self.try_simd_f32(input) {
                return Ok(result);
            }
        }

        // Generic fallback for multi-dimensional or other types
        let mut output = input.clone();

        // Simple softmax implementation for the last axis
        if self.axis == -1 || self.axis as usize == input.ndim() - 1 {
            // For 1D case or applying to last axis
            let max_val = input.fold(F::neg_infinity(), |acc, &x| if x > acc { x } else { acc });

            // Subtract max for numerical stability
            Zip::from(&mut output).for_each(|x| {
                *x = (*x - max_val).exp();
            });

            // Sum all exponentials
            let sum = output.sum();

            // Normalize
            Zip::from(&mut output).for_each(|x| {
                *x = *x / sum;
            });
        }

        Ok(output)
    }

    fn backward(
        &self,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Forward pass to get softmax _output
        let softmax_output = self.forward(input)?;
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        // For softmax: grad = softmax * (grad_out - (softmax * grad_out).sum())
        let sum_grad = Zip::from(&softmax_output)
            .and(grad_output)
            .fold(F::zero(), |acc, &s, &g| acc + s * g);

        Zip::from(&mut grad_input)
            .and(&softmax_output)
            .and(grad_output)
            .for_each(|grad_in, &s, &grad_out| {
                *grad_in = s * (grad_out - sum_grad);
            });

        Ok(grad_input)
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
    fn test_relu_backward() {
        let relu = ReLU::new();
        let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
        let grad_output = Array::from_vec(vec![1.0, 1.0, 1.0]).into_dyn();
        let grad_input = relu
            .backward(&grad_output, &input)
            .expect("Operation failed");
        let expected = Array::from_vec(vec![0.0, 0.0, 1.0]).into_dyn();
        assert_eq!(grad_input, expected);
    }

    #[test]
    fn test_leaky_relu_backward() {
        let relu = ReLU::leaky(0.01);
        let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
        let grad_output = Array::from_vec(vec![1.0, 1.0, 1.0]).into_dyn();
        let grad_input = relu
            .backward(&grad_output, &input)
            .expect("Operation failed");
        let expected = Array::from_vec(vec![0.01, 0.01, 1.0]).into_dyn();
        for (a, b) in grad_input.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
