//! GPU-accelerated operations for automatic differentiation

use super::{GpuGradient, GpuTensor};
use crate::{error::AutogradError, Float, Result};
use scirs2_core::gpu::GpuKernel;
use std::sync::Arc;

/// GPU operation trait for automatic differentiation
pub trait GpuOp<T: Float + scirs2_core::gpu::GpuDataType> {
    /// Execute forward pass on GPU
    fn forward(&self, inputs: &[&GpuTensor<T>]) -> Result<GpuTensor<T>>;

    /// Execute backward pass on GPU
    fn backward(
        &self,
        grad_output: &GpuTensor<T>,
        inputs: &[&GpuTensor<T>],
    ) -> Result<Vec<GpuTensor<T>>>;
}

/// GPU matrix multiplication with gradient support
pub struct GpuMatMul<T: Float + scirs2_core::gpu::GpuDataType> {
    gradient_computer: Arc<GpuGradient<T>>,
}

impl<T: Float + scirs2_core::gpu::GpuDataType> GpuMatMul<T> {
    /// Create a new GPU matrix multiplication operation
    pub fn new(gradient_computer: Arc<GpuGradient<T>>) -> Self {
        Self { gradient_computer }
    }
}

impl<T: Float + scirs2_core::gpu::GpuDataType> GpuOp<T> for GpuMatMul<T> {
    fn forward(&self, inputs: &[&GpuTensor<T>]) -> Result<GpuTensor<T>> {
        if inputs.len() != 2 {
            return Err(AutogradError::invalid_argument(format!(
                "MatMul requires 2 inputs, got {}",
                inputs.len()
            )));
        }

        let a = inputs[0];
        let b = inputs[1];

        // Validate shapes for matrix multiplication
        if a.shape().len() != 2 || b.shape().len() != 2 {
            return Err(AutogradError::shape_error(
                "MatMul inputs must be 2D matrices".to_string(),
            ));
        }

        if a.shape()[1] != b.shape()[0] {
            return Err(AutogradError::shape_error(format!(
                "Invalid matrix multiplication dimensions: ({}, {}) x ({}, {})",
                a.shape()[0],
                a.shape()[1],
                b.shape()[0],
                b.shape()[1]
            )));
        }

        // Create output tensor
        let output_shape = vec![a.shape()[0], b.shape()[1]];
        let context = a.context().clone();

        // Use GPU GEMM kernel for actual computation
        let output_buffer = context
            .gemm(
                a.buffer(),
                b.buffer(),
                a.shape()[0],
                a.shape()[1],
                b.shape()[1],
            )
            .map_err(|e| AutogradError::gpu_error(format!("GEMM failed: {}", e)))?;

        Ok(GpuTensor {
            buffer: Arc::new(output_buffer),
            shape: output_shape,
            context,
        })
    }

    fn backward(
        &self,
        grad_output: &GpuTensor<T>,
        inputs: &[&GpuTensor<T>],
    ) -> Result<Vec<GpuTensor<T>>> {
        if inputs.len() != 2 {
            return Err(AutogradError::invalid_argument(
                "MatMul requires 2 inputs".to_string(),
            ));
        }

        let a = inputs[0];
        let b = inputs[1];
        let context = a.context().clone();

        // grad_a = grad_output @ b^T
        let grad_a_buffer = context
            .gemm_transpose_b(
                grad_output.buffer(),
                b.buffer(),
                grad_output.shape()[0],
                grad_output.shape()[1],
                b.shape()[0],
            )
            .map_err(|e| AutogradError::gpu_error(format!("Gradient computation failed: {}", e)))?;

        let grad_a = GpuTensor {
            buffer: Arc::new(grad_a_buffer),
            shape: a.shape().to_vec(),
            context: context.clone(),
        };

        // grad_b = a^T @ grad_output
        let grad_b_buffer = context
            .gemm_transpose_a(
                a.buffer(),
                grad_output.buffer(),
                a.shape()[1],
                a.shape()[0],
                grad_output.shape()[1],
            )
            .map_err(|e| AutogradError::gpu_error(format!("Gradient computation failed: {}", e)))?;

        let grad_b = GpuTensor {
            buffer: Arc::new(grad_b_buffer),
            shape: b.shape().to_vec(),
            context,
        };

        Ok(vec![grad_a, grad_b])
    }
}

/// GPU element-wise activation functions
pub struct GpuActivation<T: Float + scirs2_core::gpu::GpuDataType> {
    activation_type: ActivationType,
    gradient_computer: Arc<GpuGradient<T>>,
}

/// Supported activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Gaussian Error Linear Unit
    GELU,
}

impl<T: Float + scirs2_core::gpu::GpuDataType> GpuActivation<T> {
    /// Create a new GPU activation function
    pub fn new(activation_type: ActivationType, gradient_computer: Arc<GpuGradient<T>>) -> Self {
        Self {
            activation_type,
            gradient_computer,
        }
    }
}

impl<T: Float + scirs2_core::gpu::GpuDataType> GpuOp<T> for GpuActivation<T> {
    fn forward(&self, inputs: &[&GpuTensor<T>]) -> Result<GpuTensor<T>> {
        if inputs.len() != 1 {
            return Err(AutogradError::invalid_argument(format!(
                "Activation requires 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let context = input.context();

        // Execute appropriate activation kernel
        let output_buffer = match self.activation_type {
            ActivationType::ReLU => context.relu(input.buffer()),
            ActivationType::Sigmoid => context.sigmoid(input.buffer()),
            ActivationType::Tanh => context.tanh(input.buffer()),
            ActivationType::GELU => context.gelu(input.buffer()),
        }
        .map_err(|e| AutogradError::gpu_error(format!("Activation failed: {}", e)))?;

        Ok(GpuTensor {
            buffer: Arc::new(output_buffer),
            shape: input.shape().to_vec(),
            context: Arc::clone(context),
        })
    }

    fn backward(
        &self,
        grad_output: &GpuTensor<T>,
        inputs: &[&GpuTensor<T>],
    ) -> Result<Vec<GpuTensor<T>>> {
        if inputs.len() != 1 {
            return Err(AutogradError::invalid_argument(
                "Activation requires 1 input".to_string(),
            ));
        }

        let input = inputs[0];
        let context = input.context();

        // Compute gradient based on activation type
        let grad_input_buffer = match self.activation_type {
            ActivationType::ReLU => context.relu_backward(grad_output.buffer(), input.buffer()),
            ActivationType::Sigmoid => {
                context.sigmoid_backward(grad_output.buffer(), input.buffer())
            }
            ActivationType::Tanh => context.tanh_backward(grad_output.buffer(), input.buffer()),
            ActivationType::GELU => context.gelu_backward(grad_output.buffer(), input.buffer()),
        }
        .map_err(|e| AutogradError::gpu_error(format!("Activation gradient failed: {}", e)))?;

        let grad_input = GpuTensor {
            buffer: Arc::new(grad_input_buffer),
            shape: input.shape().to_vec(),
            context: Arc::clone(context),
        };

        Ok(vec![grad_input])
    }
}

/// GPU reduction operations
pub struct GpuReduction<T: Float + scirs2_core::gpu::GpuDataType> {
    reduction_type: ReductionType,
    axis: Option<usize>,
    gradient_computer: Arc<GpuGradient<T>>,
}

/// Supported reduction types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionType {
    /// Sum reduction
    Sum,
    /// Mean reduction
    Mean,
    /// Maximum reduction
    Max,
    /// Minimum reduction
    Min,
}

impl<T: Float + scirs2_core::gpu::GpuDataType> GpuReduction<T> {
    /// Create a new GPU reduction operation
    pub fn new(
        reduction_type: ReductionType,
        axis: Option<usize>,
        gradient_computer: Arc<GpuGradient<T>>,
    ) -> Self {
        Self {
            reduction_type,
            axis,
            gradient_computer,
        }
    }
}

impl<T: Float + scirs2_core::gpu::GpuDataType> GpuOp<T> for GpuReduction<T> {
    fn forward(&self, inputs: &[&GpuTensor<T>]) -> Result<GpuTensor<T>> {
        if inputs.len() != 1 {
            return Err(AutogradError::invalid_argument(format!(
                "Reduction requires 1 input, got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let context = input.context();

        // Execute reduction kernel
        let (output_buffer, output_shape) = match self.axis {
            Some(axis) => {
                // Reduce along specific axis
                let mut new_shape = input.shape().to_vec();
                if axis >= new_shape.len() {
                    return Err(AutogradError::invalid_argument(format!(
                        "Axis {} out of bounds for shape {:?}",
                        axis,
                        input.shape()
                    )));
                }
                new_shape[axis] = 1;

                let buffer = match self.reduction_type {
                    ReductionType::Sum => context.sum_axis(input.buffer(), input.shape(), axis),
                    ReductionType::Mean => context.mean_axis(input.buffer(), input.shape(), axis),
                    ReductionType::Max => context.max_axis(input.buffer(), input.shape(), axis),
                    ReductionType::Min => context.min_axis(input.buffer(), input.shape(), axis),
                }
                .map_err(|e| AutogradError::gpu_error(format!("Reduction failed: {}", e)))?;

                (buffer, new_shape)
            }
            None => {
                // Global reduction to scalar
                let buffer = match self.reduction_type {
                    ReductionType::Sum => context.sum_all(input.buffer()),
                    ReductionType::Mean => context.mean_all(input.buffer()),
                    ReductionType::Max => context.max_all(input.buffer()),
                    ReductionType::Min => context.min_all(input.buffer()),
                }
                .map_err(|e| AutogradError::gpu_error(format!("Reduction failed: {}", e)))?;

                (buffer, vec![1])
            }
        };

        Ok(GpuTensor {
            buffer: Arc::new(output_buffer),
            shape: output_shape,
            context: Arc::clone(context),
        })
    }

    fn backward(
        &self,
        grad_output: &GpuTensor<T>,
        inputs: &[&GpuTensor<T>],
    ) -> Result<Vec<GpuTensor<T>>> {
        if inputs.len() != 1 {
            return Err(AutogradError::invalid_argument(
                "Reduction requires 1 input".to_string(),
            ));
        }

        let input = inputs[0];
        let context = input.context();

        // Broadcast gradient back to input shape
        let grad_input_buffer = context
            .broadcast(grad_output.buffer(), grad_output.shape(), input.shape())
            .map_err(|e| AutogradError::gpu_error(format!("Gradient broadcast failed: {}", e)))?;

        // For mean, scale by 1/N
        let grad_input_buffer = if matches!(self.reduction_type, ReductionType::Mean) {
            let n = if let Some(axis) = self.axis {
                input.shape()[axis]
            } else {
                input.shape().iter().product()
            };
            let scale = T::from(n).ok_or_else(|| {
                AutogradError::compute_error("Failed to convert reduction size".to_string())
            })?;

            context
                .scale(&grad_input_buffer, T::one() / scale)
                .map_err(|e| AutogradError::gpu_error(format!("Gradient scaling failed: {}", e)))?
        } else {
            grad_input_buffer
        };

        let grad_input = GpuTensor {
            buffer: Arc::new(grad_input_buffer),
            shape: input.shape().to_vec(),
            context: Arc::clone(context),
        };

        Ok(vec![grad_input])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::gpu::GpuBackend;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_activation_types() {
        assert_eq!(format!("{:?}", ActivationType::ReLU), "ReLU");
        assert_eq!(format!("{:?}", ActivationType::Sigmoid), "Sigmoid");
    }

    #[test]
    fn test_reduction_types() {
        assert_eq!(format!("{:?}", ReductionType::Sum), "Sum");
        assert_eq!(format!("{:?}", ReductionType::Mean), "Mean");
    }
}
