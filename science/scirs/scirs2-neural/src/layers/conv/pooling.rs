//! Pooling layer implementations
//!
//! Provides MaxPool2D and AvgPool2D layers with full forward/backward support.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, ArrayView1, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Threshold for using SIMD-accelerated GlobalAvgPool2D
const GLOBAL_AVGPOOL_SIMD_THRESHOLD: usize = 64;

/// 2D Max Pooling layer
///
/// Applies max pooling over an input signal composed of several input planes.
///
/// # Input Shape
/// - 4D tensor: (batch_size, channels, height, width)
///
/// # Output Shape
/// - 4D tensor: (batch_size, channels, out_height, out_width)
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{MaxPool2D, Layer};
/// use scirs2_core::ndarray::Array4;
///
/// // Create a 2x2 max pooling layer with stride 2
/// let pool = MaxPool2D::<f64>::new((2, 2), (2, 2), Some("maxpool")).expect("Operation failed");
///
/// // Input: (batch=1, channels=3, height=8, width=8)
/// let input = Array4::<f64>::from_elem((1, 3, 8, 8), 1.0).into_dyn();
/// let output = pool.forward(&input).expect("Operation failed");
///
/// // Output: (batch=1, channels=3, height=4, width=4)
/// assert_eq!(output.shape(), &[1, 3, 4, 4]);
/// ```
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct MaxPool2D<F: Float + Debug + Send + Sync + NumAssign> {
    /// Pool size (height, width)
    pool_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Layer name
    name: Option<String>,
    /// Cache for max indices (for backward pass)
    max_indices: Arc<RwLock<Option<Array<(usize, usize), IxDyn>>>>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> MaxPool2D<F> {
    /// Create a new MaxPool2D layer
    ///
    /// # Arguments
    /// * `pool_size` - Size of the pooling window (height, width)
    /// * `stride` - Stride of the pooling operation (height, width)
    /// * `name` - Optional name for the layer
    pub fn new(
        pool_size: (usize, usize),
        stride: (usize, usize),
        name: Option<&str>,
    ) -> Result<Self> {
        if pool_size.0 == 0 || pool_size.1 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Pool size must be greater than 0".to_string(),
            ));
        }

        if stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Stride must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            pool_size,
            stride,
            name: name.map(String::from),
            max_indices: Arc::new(RwLock::new(None)),
            input_cache: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        })
    }

    /// Calculate output dimensions
    fn calculate_output_dims(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h - self.pool_size.0) / self.stride.0 + 1;
        let out_w = (input_w - self.pool_size.1) / self.stride.1 + 1;
        (out_h, out_w)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for MaxPool2D<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "MaxPool2D expects 4D input (batch, channels, height, width), got {}D",
                shape.len()
            )));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let input_h = shape[2];
        let input_w = shape[3];

        if input_h < self.pool_size.0 || input_w < self.pool_size.1 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Input size ({}x{}) is smaller than pool size ({}x{})",
                input_h, input_w, self.pool_size.0, self.pool_size.1
            )));
        }

        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }

        let (out_h, out_w) = self.calculate_output_dims(input_h, input_w);
        let output_shape = vec![batch_size, channels, out_h, out_w];
        let mut output = Array::zeros(IxDyn(&output_shape));

        // Store max indices for backward pass
        let indices_shape = vec![batch_size, channels, out_h, out_w];
        let mut max_indices = Array::from_elem(IxDyn(&indices_shape), (0usize, 0usize));

        // Perform max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;

                        let mut max_val = F::neg_infinity();
                        let mut max_h = 0;
                        let mut max_w = 0;

                        for ph in 0..self.pool_size.0 {
                            for pw in 0..self.pool_size.1 {
                                let ih = h_start + ph;
                                let iw = w_start + pw;

                                if ih < input_h && iw < input_w {
                                    let val = input[[b, c, ih, iw]];
                                    if val > max_val {
                                        max_val = val;
                                        max_h = ih;
                                        max_w = iw;
                                    }
                                }
                            }
                        }

                        output[[b, c, oh, ow]] = max_val;
                        max_indices[[b, c, oh, ow]] = (max_h, max_w);
                    }
                }
            }
        }

        // Store max indices
        if let Ok(mut indices_cache) = self.max_indices.write() {
            *indices_cache = Some(max_indices);
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Get cached input
        let input_guard = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on input cache".to_string())
        })?;

        let input = input_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            )
        })?;

        // Get max indices
        let indices_guard = self.max_indices.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on max indices".to_string())
        })?;

        let max_indices = indices_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError("No cached max indices for backward pass".to_string())
        })?;

        // Create gradient input with same shape as input
        let mut grad_input = Array::zeros(input.raw_dim());

        let grad_shape = grad_output.shape();
        let batch_size = grad_shape[0];
        let channels = grad_shape[1];
        let out_h = grad_shape[2];
        let out_w = grad_shape[3];

        // Distribute gradients back to max positions
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let (max_h, max_w) = max_indices[[b, c, oh, ow]];
                        grad_input[[b, c, max_h, max_w]] += grad_output[[b, c, oh, ow]];
                    }
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // MaxPool2D has no learnable parameters
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "MaxPool2D"
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        None
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn parameter_count(&self) -> usize {
        0 // No learnable parameters
    }

    fn layer_description(&self) -> String {
        format!(
            "type:MaxPool2D, pool_size:{}x{}, stride:{}x{}",
            self.pool_size.0, self.pool_size.1, self.stride.0, self.stride.1
        )
    }
}

/// 2D Average Pooling layer
///
/// Applies average pooling over an input signal composed of several input planes.
///
/// # Input Shape
/// - 4D tensor: (batch_size, channels, height, width)
///
/// # Output Shape
/// - 4D tensor: (batch_size, channels, out_height, out_width)
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{AvgPool2D, Layer};
/// use scirs2_core::ndarray::Array4;
///
/// // Create a 2x2 average pooling layer with stride 2
/// let pool = AvgPool2D::<f64>::new((2, 2), (2, 2), Some("avgpool")).expect("Operation failed");
///
/// // Input: (batch=1, channels=3, height=8, width=8)
/// let input = Array4::<f64>::from_elem((1, 3, 8, 8), 1.0).into_dyn();
/// let output = pool.forward(&input).expect("Operation failed");
///
/// // Output: (batch=1, channels=3, height=4, width=4)
/// assert_eq!(output.shape(), &[1, 3, 4, 4]);
/// ```
#[derive(Debug)]
pub struct AvgPool2D<F: Float + Debug + Send + Sync + NumAssign> {
    /// Pool size (height, width)
    pool_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Layer name
    name: Option<String>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> AvgPool2D<F> {
    /// Create a new AvgPool2D layer
    ///
    /// # Arguments
    /// * `pool_size` - Size of the pooling window (height, width)
    /// * `stride` - Stride of the pooling operation (height, width)
    /// * `name` - Optional name for the layer
    pub fn new(
        pool_size: (usize, usize),
        stride: (usize, usize),
        name: Option<&str>,
    ) -> Result<Self> {
        if pool_size.0 == 0 || pool_size.1 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Pool size must be greater than 0".to_string(),
            ));
        }

        if stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Stride must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            pool_size,
            stride,
            name: name.map(String::from),
            input_cache: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        })
    }

    /// Calculate output dimensions
    fn calculate_output_dims(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h - self.pool_size.0) / self.stride.0 + 1;
        let out_w = (input_w - self.pool_size.1) / self.stride.1 + 1;
        (out_h, out_w)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for AvgPool2D<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "AvgPool2D expects 4D input (batch, channels, height, width), got {}D",
                shape.len()
            )));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let input_h = shape[2];
        let input_w = shape[3];

        if input_h < self.pool_size.0 || input_w < self.pool_size.1 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Input size ({}x{}) is smaller than pool size ({}x{})",
                input_h, input_w, self.pool_size.0, self.pool_size.1
            )));
        }

        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }

        let (out_h, out_w) = self.calculate_output_dims(input_h, input_w);
        let output_shape = vec![batch_size, channels, out_h, out_w];
        let mut output = Array::zeros(IxDyn(&output_shape));

        let pool_area =
            F::from(self.pool_size.0 * self.pool_size.1).expect("Failed to convert to float");

        // Perform average pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;

                        let mut sum = F::zero();
                        let mut count = 0;

                        for ph in 0..self.pool_size.0 {
                            for pw in 0..self.pool_size.1 {
                                let ih = h_start + ph;
                                let iw = w_start + pw;

                                if ih < input_h && iw < input_w {
                                    sum += input[[b, c, ih, iw]];
                                    count += 1;
                                }
                            }
                        }

                        output[[b, c, oh, ow]] =
                            sum / F::from(count).expect("Failed to convert to float");
                    }
                }
            }
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Get cached input
        let input_guard = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on input cache".to_string())
        })?;

        let input = input_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            )
        })?;

        // Create gradient input with same shape as input
        let mut grad_input = Array::zeros(input.raw_dim());

        let grad_shape = grad_output.shape();
        let batch_size = grad_shape[0];
        let channels = grad_shape[1];
        let out_h = grad_shape[2];
        let out_w = grad_shape[3];

        let input_h = input.shape()[2];
        let input_w = input.shape()[3];

        // Distribute gradients equally to all pooled positions
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;

                        // Count actual elements in pool window
                        let mut count = 0;
                        for ph in 0..self.pool_size.0 {
                            for pw in 0..self.pool_size.1 {
                                let ih = h_start + ph;
                                let iw = w_start + pw;
                                if ih < input_h && iw < input_w {
                                    count += 1;
                                }
                            }
                        }

                        let grad_per_elem = grad_output[[b, c, oh, ow]]
                            / F::from(count).expect("Failed to convert to float");

                        // Distribute gradient
                        for ph in 0..self.pool_size.0 {
                            for pw in 0..self.pool_size.1 {
                                let ih = h_start + ph;
                                let iw = w_start + pw;
                                if ih < input_h && iw < input_w {
                                    grad_input[[b, c, ih, iw]] += grad_per_elem;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // AvgPool2D has no learnable parameters
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "AvgPool2D"
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        None
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn parameter_count(&self) -> usize {
        0 // No learnable parameters
    }

    fn layer_description(&self) -> String {
        format!(
            "type:AvgPool2D, pool_size:{}x{}, stride:{}x{}",
            self.pool_size.0, self.pool_size.1, self.stride.0, self.stride.1
        )
    }
}

/// Global Average Pooling 2D layer
///
/// Applies global average pooling, reducing spatial dimensions to 1x1.
///
/// # Input Shape
/// - 4D tensor: (batch_size, channels, height, width)
///
/// # Output Shape
/// - 4D tensor: (batch_size, channels, 1, 1)
#[derive(Debug)]
pub struct GlobalAvgPool2D<F: Float + Debug + Send + Sync + NumAssign> {
    /// Layer name
    name: Option<String>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> GlobalAvgPool2D<F> {
    /// Create a new GlobalAvgPool2D layer
    pub fn new(name: Option<&str>) -> Self {
        Self {
            name: name.map(String::from),
            input_cache: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        }
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + SimdUnifiedOps + NumAssign + 'static> Layer<F>
    for GlobalAvgPool2D<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "GlobalAvgPool2D expects 4D input (batch, channels, height, width), got {}D",
                shape.len()
            )));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }

        let output_shape = vec![batch_size, channels, 1, 1];
        let mut output = Array::zeros(IxDyn(&output_shape));

        let spatial_size = height * width;
        let spatial_size_f = F::from(spatial_size).expect("Failed to convert to float");

        // Use SIMD for larger spatial dimensions
        if spatial_size >= GLOBAL_AVGPOOL_SIMD_THRESHOLD {
            // SIMD path: flatten each channel and use simd_sum
            for b in 0..batch_size {
                for c in 0..channels {
                    // Extract the spatial slice [height, width] and flatten it
                    let channel_slice = input.slice(scirs2_core::ndarray::s![b, c, .., ..]);
                    // Flatten to 1D for SIMD sum
                    let flat_view = channel_slice
                        .to_owned()
                        .into_shape_with_order(spatial_size)
                        .expect("Operation failed");
                    let view_1d: ArrayView1<F> = flat_view.view();
                    let sum = F::simd_sum(&view_1d);
                    output[[b, c, 0, 0]] = sum / spatial_size_f;
                }
            }
        } else {
            // Scalar fallback for small spatial dimensions
            for b in 0..batch_size {
                for c in 0..channels {
                    let mut sum = F::zero();
                    for h in 0..height {
                        for w in 0..width {
                            sum += input[[b, c, h, w]];
                        }
                    }
                    output[[b, c, 0, 0]] = sum / spatial_size_f;
                }
            }
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let input_guard = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on input cache".to_string())
        })?;

        let input = input_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            )
        })?;

        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let mut grad_input = Array::zeros(input.raw_dim());
        let spatial_size = F::from(height * width).expect("Failed to convert to float");

        for b in 0..batch_size {
            for c in 0..channels {
                let grad_per_elem = grad_output[[b, c, 0, 0]] / spatial_size;
                for h in 0..height {
                    for w in 0..width {
                        grad_input[[b, c, h, w]] = grad_per_elem;
                    }
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "GlobalAvgPool2D"
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        None
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn parameter_count(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array4;

    #[test]
    fn test_maxpool2d_creation() {
        let pool =
            MaxPool2D::<f64>::new((2, 2), (2, 2), Some("maxpool")).expect("Operation failed");
        assert_eq!(pool.pool_size, (2, 2));
        assert_eq!(pool.stride, (2, 2));
    }

    #[test]
    fn test_maxpool2d_forward_shape() {
        let pool =
            MaxPool2D::<f64>::new((2, 2), (2, 2), Some("maxpool")).expect("Operation failed");

        let input = Array4::<f64>::from_elem((2, 3, 8, 8), 1.0).into_dyn();
        let output = pool.forward(&input).expect("Operation failed");

        // Output: (2, 3, 4, 4)
        assert_eq!(output.shape(), &[2, 3, 4, 4]);
    }

    #[test]
    fn test_maxpool2d_forward_values() {
        let pool = MaxPool2D::<f64>::new((2, 2), (2, 2), None).expect("Operation failed");

        // Create input with known values
        let mut input = Array4::<f64>::zeros((1, 1, 4, 4));
        input[[0, 0, 0, 0]] = 1.0;
        input[[0, 0, 0, 1]] = 2.0;
        input[[0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 1]] = 4.0; // This should be the max for first pool
        input[[0, 0, 2, 2]] = 10.0; // This should be the max for last pool

        let output = pool.forward(&input.into_dyn()).expect("Operation failed");

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output[[0, 0, 0, 0]], 4.0); // Max of top-left 2x2
        assert_eq!(output[[0, 0, 1, 1]], 10.0); // Max of bottom-right 2x2
    }

    #[test]
    fn test_maxpool2d_backward() {
        let pool = MaxPool2D::<f64>::new((2, 2), (2, 2), None).expect("Operation failed");

        let mut input = Array4::<f64>::zeros((1, 1, 4, 4));
        input[[0, 0, 1, 1]] = 4.0;
        input[[0, 0, 3, 3]] = 8.0;

        let _output = pool
            .forward(&input.clone().into_dyn())
            .expect("Operation failed");

        let grad_output = Array4::<f64>::from_elem((1, 1, 2, 2), 1.0).into_dyn();
        let grad_input = pool
            .backward(&input.into_dyn(), &grad_output)
            .expect("Operation failed");

        assert_eq!(grad_input.shape(), &[1, 1, 4, 4]);
        // Gradient should be at max positions
        assert_eq!(grad_input[[0, 0, 1, 1]], 1.0);
        assert_eq!(grad_input[[0, 0, 3, 3]], 1.0);
    }

    #[test]
    fn test_avgpool2d_forward_shape() {
        let pool =
            AvgPool2D::<f64>::new((2, 2), (2, 2), Some("avgpool")).expect("Operation failed");

        let input = Array4::<f64>::from_elem((2, 3, 8, 8), 1.0).into_dyn();
        let output = pool.forward(&input).expect("Operation failed");

        assert_eq!(output.shape(), &[2, 3, 4, 4]);
    }

    #[test]
    fn test_avgpool2d_forward_values() {
        let pool = AvgPool2D::<f64>::new((2, 2), (2, 2), None).expect("Operation failed");

        // Create input with known values
        let mut input = Array4::<f64>::zeros((1, 1, 4, 4));
        input[[0, 0, 0, 0]] = 1.0;
        input[[0, 0, 0, 1]] = 2.0;
        input[[0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 1]] = 4.0;

        let output = pool.forward(&input.into_dyn()).expect("Operation failed");

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        // Average of (1, 2, 3, 4) = 2.5
        assert!((output[[0, 0, 0, 0]] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_avgpool2d_backward() {
        let pool = AvgPool2D::<f64>::new((2, 2), (2, 2), None).expect("Operation failed");

        let input = Array4::<f64>::from_elem((1, 1, 4, 4), 1.0);
        let _output = pool
            .forward(&input.clone().into_dyn())
            .expect("Operation failed");

        let grad_output = Array4::<f64>::from_elem((1, 1, 2, 2), 4.0).into_dyn();
        let grad_input = pool
            .backward(&input.into_dyn(), &grad_output)
            .expect("Operation failed");

        assert_eq!(grad_input.shape(), &[1, 1, 4, 4]);
        // Gradient should be distributed equally: 4.0 / 4 = 1.0
        assert!((grad_input[[0, 0, 0, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_avgpool2d_forward() {
        let pool = GlobalAvgPool2D::<f64>::new(Some("gap"));

        let input = Array4::<f64>::from_elem((2, 3, 8, 8), 2.0).into_dyn();
        let output = pool.forward(&input).expect("Operation failed");

        assert_eq!(output.shape(), &[2, 3, 1, 1]);
        assert!((output[[0, 0, 0, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_avgpool2d_backward() {
        let pool = GlobalAvgPool2D::<f64>::new(None);

        let input = Array4::<f64>::from_elem((1, 1, 4, 4), 1.0);
        let _output = pool
            .forward(&input.clone().into_dyn())
            .expect("Operation failed");

        let grad_output = Array4::<f64>::from_elem((1, 1, 1, 1), 16.0).into_dyn();
        let grad_input = pool
            .backward(&input.into_dyn(), &grad_output)
            .expect("Operation failed");

        assert_eq!(grad_input.shape(), &[1, 1, 4, 4]);
        // Gradient per element: 16.0 / 16 = 1.0
        assert!((grad_input[[0, 0, 0, 0]] - 1.0).abs() < 1e-10);
    }
}
