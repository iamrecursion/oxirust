//! Distributed automatic differentiation
//!
//! This module provides support for distributed gradient computation across multiple
//! processes or machines, enabling data-parallel and model-parallel training.

use crate::{error::AutogradError, tensor::Tensor, Float, NdArray, Result};
use std::sync::{Arc, Mutex};

pub mod communication;
pub mod data_parallel;
pub mod model_parallel;

/// Distributed gradient accumulator
pub struct DistributedGradient<T: Float> {
    /// Local gradients
    local_gradients: Arc<Mutex<Vec<NdArray<T>>>>,
    /// Accumulated gradients across all workers
    accumulated: Arc<Mutex<Option<Vec<NdArray<T>>>>>,
    /// Number of workers
    num_workers: usize,
    /// Current worker rank
    rank: usize,
}

impl<T: Float + scirs2_core::ndarray::ScalarOperand> DistributedGradient<T> {
    /// Create a new distributed gradient accumulator
    pub fn new(num_workers: usize, rank: usize) -> Self {
        Self {
            local_gradients: Arc::new(Mutex::new(Vec::new())),
            accumulated: Arc::new(Mutex::new(None)),
            num_workers,
            rank,
        }
    }

    /// Add local gradient
    pub fn add_local(&self, gradient: NdArray<T>) -> Result<()> {
        let mut local = self
            .local_gradients
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock local gradients"))?;
        local.push(gradient);
        Ok(())
    }

    /// Synchronize gradients across all workers (allreduce)
    pub fn allreduce(&self) -> Result<Vec<NdArray<T>>> {
        // In a real implementation, this would communicate with other workers
        // For now, we simulate by averaging local gradients

        let local = self
            .local_gradients
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock local gradients"))?;

        // Average gradients (simulating allreduce)
        let num_grads = local.len();
        if num_grads == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(num_grads);
        for grad in local.iter() {
            let averaged = grad
                / T::from(self.num_workers).ok_or_else(|| {
                    AutogradError::compute_error("Failed to convert num_workers".to_string())
                })?;
            result.push(averaged);
        }

        // Store accumulated result
        let mut accumulated = self
            .accumulated
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock accumulated gradients"))?;
        *accumulated = Some(result.clone());

        Ok(result)
    }

    /// Get current worker rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get total number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Clear accumulated gradients
    pub fn clear(&self) -> Result<()> {
        let mut local = self
            .local_gradients
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock local gradients"))?;
        local.clear();

        let mut accumulated = self
            .accumulated
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock accumulated gradients"))?;
        *accumulated = None;

        Ok(())
    }
}

/// Data-parallel training strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// Data parallel - replicate model, split data
    DataParallel,
    /// Model parallel - split model across devices
    ModelParallel,
    /// Pipeline parallel - split model into stages
    PipelineParallel,
    /// Hybrid - combination of strategies
    Hybrid,
}

/// Distributed training configuration
pub struct DistributedConfig {
    /// Parallel strategy
    pub strategy: ParallelStrategy,
    /// Number of workers
    pub num_workers: usize,
    /// Current worker rank
    pub rank: usize,
    /// Gradient accumulation steps
    pub grad_accumulation_steps: usize,
    /// Use gradient compression
    pub compress_gradients: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelStrategy::DataParallel,
            num_workers: 1,
            rank: 0,
            grad_accumulation_steps: 1,
            compress_gradients: false,
        }
    }
}

/// Gradient synchronization backend
pub trait SyncBackend<T: Float>: Send + Sync {
    /// Perform allreduce operation on gradients
    fn allreduce(&self, gradients: &[NdArray<T>]) -> Result<Vec<NdArray<T>>>;

    /// Broadcast parameters from rank 0 to all workers
    fn broadcast(&self, parameters: &[NdArray<T>]) -> Result<Vec<NdArray<T>>>;

    /// Gather gradients from all workers to rank 0
    fn gather(&self, gradient: &NdArray<T>) -> Result<Vec<NdArray<T>>>;

    /// Scatter data from rank 0 to all workers
    fn scatter(&self, data: &[NdArray<T>]) -> Result<NdArray<T>>;
}

/// Local (single-process) sync backend for testing
pub struct LocalSyncBackend<T: Float> {
    num_workers: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> LocalSyncBackend<T> {
    /// Create a new local sync backend
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + scirs2_core::ndarray::ScalarOperand> SyncBackend<T> for LocalSyncBackend<T> {
    fn allreduce(&self, gradients: &[NdArray<T>]) -> Result<Vec<NdArray<T>>> {
        // Simulate allreduce by averaging (for testing)
        let divisor = T::from(self.num_workers).ok_or_else(|| {
            AutogradError::compute_error("Failed to convert num_workers".to_string())
        })?;

        Ok(gradients.iter().map(|g| g / divisor).collect())
    }

    fn broadcast(&self, parameters: &[NdArray<T>]) -> Result<Vec<NdArray<T>>> {
        // In local mode, just return copies
        Ok(parameters.to_vec())
    }

    fn gather(&self, gradient: &NdArray<T>) -> Result<Vec<NdArray<T>>> {
        // Simulate gather by replicating
        Ok(vec![gradient.clone(); self.num_workers])
    }

    fn scatter(&self, data: &[NdArray<T>]) -> Result<NdArray<T>> {
        // Return first element (simulating scatter to rank 0)
        data.first()
            .cloned()
            .ok_or_else(|| AutogradError::invalid_argument("Empty data for scatter".to_string()))
    }
}

/// Distributed optimizer wrapper
pub struct DistributedOptimizer<T: Float> {
    /// Synchronization backend
    backend: Arc<dyn SyncBackend<T>>,
    /// Configuration
    config: DistributedConfig,
    /// Gradient accumulation buffer
    grad_buffer: Arc<Mutex<Vec<Vec<NdArray<T>>>>>,
}

impl<T: Float + scirs2_core::ndarray::ScalarOperand> DistributedOptimizer<T> {
    /// Create a new distributed optimizer
    pub fn new(backend: Arc<dyn SyncBackend<T>>, config: DistributedConfig) -> Self {
        Self {
            backend,
            config,
            grad_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Accumulate gradient
    pub fn accumulate_gradient(&self, gradients: Vec<NdArray<T>>) -> Result<()> {
        let mut buffer = self
            .grad_buffer
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock gradient buffer"))?;
        buffer.push(gradients);
        Ok(())
    }

    /// Check if ready to synchronize
    pub fn should_sync(&self) -> Result<bool> {
        let buffer = self
            .grad_buffer
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock gradient buffer"))?;
        Ok(buffer.len() >= self.config.grad_accumulation_steps)
    }

    /// Synchronize accumulated gradients
    pub fn sync_gradients(&self) -> Result<Vec<NdArray<T>>> {
        let mut buffer = self
            .grad_buffer
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock gradient buffer"))?;

        if buffer.is_empty() {
            return Ok(Vec::new());
        }

        // Average accumulated gradients
        let num_grads = buffer[0].len();
        let num_steps = buffer.len();
        let mut averaged = Vec::with_capacity(num_grads);

        for i in 0..num_grads {
            let mut sum = buffer[0][i].clone();
            for step in buffer.iter().skip(1) {
                sum += &step[i];
            }
            let avg = sum
                / T::from(num_steps).ok_or_else(|| {
                    AutogradError::compute_error("Failed to convert num_steps".to_string())
                })?;
            averaged.push(avg);
        }

        // Synchronize across workers
        let synced = self.backend.allreduce(&averaged)?;

        // Clear buffer
        buffer.clear();

        Ok(synced)
    }

    /// Get configuration
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_distributed_gradient() {
        let grad_acc: DistributedGradient<f32> = DistributedGradient::new(4, 0);

        let grad1: Array1<f32> = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        grad_acc.add_local(grad1.into_dyn()).expect("Should add");

        let result = grad_acc.allreduce().expect("Should allreduce");
        assert_eq!(result.len(), 1);

        // Should be averaged by num_workers
        let result_vals = result[0].as_slice().expect("Should get slice");
        assert!((result_vals[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_strategy() {
        assert_eq!(
            ParallelStrategy::DataParallel,
            ParallelStrategy::DataParallel
        );
        assert_ne!(
            ParallelStrategy::DataParallel,
            ParallelStrategy::ModelParallel
        );
    }

    #[test]
    fn test_local_sync_backend() {
        let backend: LocalSyncBackend<f64> = LocalSyncBackend::new(2);

        let grad: Array1<f64> = Array1::from_vec(vec![4.0, 6.0]);
        let result = backend
            .allreduce(&[grad.into_dyn()])
            .expect("Should allreduce");

        // Should be divided by 2
        let result_vals = result[0].as_slice().expect("Should get slice");
        assert_eq!(result_vals[0], 2.0);
        assert_eq!(result_vals[1], 3.0);
    }

    #[test]
    fn test_distributed_optimizer() {
        let backend = Arc::new(LocalSyncBackend::<f32>::new(1));
        let config = DistributedConfig {
            grad_accumulation_steps: 2,
            ..Default::default()
        };

        let optimizer = DistributedOptimizer::new(backend, config);

        // Not ready after 1 accumulation
        let grad1: Array1<f32> = Array1::from_vec(vec![1.0]);
        optimizer
            .accumulate_gradient(vec![grad1.into_dyn()])
            .expect("Should accumulate");
        assert!(!optimizer.should_sync().expect("Should check"));

        // Ready after 2 accumulations
        let grad2: Array1<f32> = Array1::from_vec(vec![3.0]);
        optimizer
            .accumulate_gradient(vec![grad2.into_dyn()])
            .expect("Should accumulate");
        assert!(optimizer.should_sync().expect("Should check"));

        // Sync and check average
        let synced = optimizer.sync_gradients().expect("Should sync");
        let synced_val = synced[0].as_slice().expect("Should get slice")[0];
        assert_eq!(synced_val, 2.0); // (1 + 3) / 2
    }
}
