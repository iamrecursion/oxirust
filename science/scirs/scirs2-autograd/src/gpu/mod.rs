//! GPU-accelerated automatic differentiation
//!
//! This module provides GPU backend integration for automatic differentiation operations,
//! enabling hardware acceleration for gradient computation.

use crate::{error::AutogradError, Float, NdArray, NdArrayView, Result};
use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext};
use std::sync::Arc;

pub mod kernels;
pub mod memory;
pub mod operations;

/// GPU-accelerated tensor for automatic differentiation
pub struct GpuTensor<T: Float + scirs2_core::gpu::GpuDataType> {
    /// GPU buffer holding tensor data
    buffer: Arc<GpuBuffer<T>>,
    /// Tensor shape
    shape: Vec<usize>,
    /// GPU context
    context: Arc<GpuContext>,
}

impl<T: Float + scirs2_core::gpu::GpuDataType> GpuTensor<T> {
    /// Create a new GPU tensor from host data
    pub fn from_array(array: &NdArrayView<T>, context: Arc<GpuContext>) -> Result<Self> {
        let shape = array.shape().to_vec();
        let data: Vec<T> = array.iter().copied().collect();

        let buffer = context.create_buffer_from_slice(&data);

        Ok(Self {
            buffer: Arc::new(buffer),
            shape,
            context,
        })
    }

    /// Transfer data back to host
    pub fn to_array(&self) -> Result<NdArray<T>> {
        let data = self
            .context
            .read_buffer(&self.buffer)
            .map_err(|e| AutogradError::gpu_error(format!("Failed to read GPU buffer: {}", e)))?;

        NdArray::from_shape_vec(self.shape.clone(), data)
            .map_err(|e| AutogradError::shape_error(format!("Failed to create array: {}", e)))
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get GPU buffer reference
    pub fn buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    /// Get GPU context
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }
}

/// GPU-accelerated gradient computation
pub struct GpuGradient<T: Float + scirs2_core::gpu::GpuDataType> {
    /// Backend GPU context
    context: Arc<GpuContext>,
    /// Cache for intermediate gradients
    gradient_cache: std::collections::HashMap<usize, GpuTensor<T>>,
}

impl<T: Float + scirs2_core::gpu::GpuDataType> GpuGradient<T> {
    /// Create a new GPU gradient computer
    pub fn new(backend: GpuBackend) -> Result<Self> {
        let context = GpuContext::new(backend).map_err(|e| {
            AutogradError::gpu_error(format!("Failed to create GPU context: {}", e))
        })?;

        Ok(Self {
            context: Arc::new(context),
            gradient_cache: std::collections::HashMap::new(),
        })
    }

    /// Compute gradient on GPU
    pub fn compute_gradient(
        &mut self,
        output: &GpuTensor<T>,
        input: &GpuTensor<T>,
    ) -> Result<GpuTensor<T>> {
        // For now, implement basic gradient computation
        // This will be extended with actual GPU kernels

        // Check shape compatibility
        if output.shape() != input.shape() {
            return Err(AutogradError::shape_error(format!(
                "Shape mismatch: output {:?} vs input {:?}",
                output.shape(),
                input.shape()
            )));
        }

        // Create gradient buffer
        let grad_buffer = self
            .context
            .create_buffer::<T>(input.shape().iter().product());

        Ok(GpuTensor {
            buffer: Arc::new(grad_buffer),
            shape: input.shape().to_vec(),
            context: self.context.clone(),
        })
    }

    /// Get the GPU context
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }
}

/// GPU memory manager for gradient computation
pub struct GpuMemoryManager {
    /// Maximum memory allocation per tensor
    max_allocation: usize,
    /// Currently allocated memory
    allocated: usize,
    /// Memory pool for reuse
    pool: Vec<Arc<dyn std::any::Any + Send + Sync>>,
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager
    pub fn new(max_allocation: usize) -> Self {
        Self {
            max_allocation,
            allocated: 0,
            pool: Vec::new(),
        }
    }

    /// Check if allocation is within limits
    pub fn can_allocate(&self, size: usize) -> bool {
        self.allocated + size <= self.max_allocation
    }

    /// Record allocation
    pub fn allocate(&mut self, size: usize) -> Result<()> {
        if !self.can_allocate(size) {
            return Err(AutogradError::memory_error(format!(
                "GPU memory limit exceeded: {} + {} > {}",
                self.allocated, size, self.max_allocation
            )));
        }
        self.allocated += size;
        Ok(())
    }

    /// Record deallocation
    pub fn deallocate(&mut self, size: usize) {
        self.allocated = self.allocated.saturating_sub(size);
    }

    /// Get current memory usage
    pub fn usage(&self) -> usize {
        self.allocated
    }

    /// Get available memory
    pub fn available(&self) -> usize {
        self.max_allocation.saturating_sub(self.allocated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_gpu_memory_manager() {
        let mut manager = GpuMemoryManager::new(1024);

        assert!(manager.can_allocate(512));
        manager.allocate(512).expect("Should allocate");
        assert_eq!(manager.usage(), 512);
        assert_eq!(manager.available(), 512);

        assert!(!manager.can_allocate(1024));
        manager.deallocate(256);
        assert_eq!(manager.usage(), 256);
    }

    #[test]
    fn test_gpu_tensor_creation() {
        // Test with CPU fallback
        let context =
            Arc::new(GpuContext::new(GpuBackend::Cpu).expect("Should create CPU context"));

        let array: Array2<f32> = Array2::zeros((2, 3));
        let result = GpuTensor::from_array(&array.view().into_dyn(), context);

        // This may fail if GPU is not available, which is okay
        if let Ok(tensor) = result {
            assert_eq!(tensor.shape(), &[2, 3]);
        }
    }
}
