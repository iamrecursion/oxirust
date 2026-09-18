//! GPU memory management for automatic differentiation
//!
//! This module provides efficient memory management strategies for GPU-accelerated
//! gradient computation, including memory pooling and automatic garbage collection.

use crate::{error::AutogradError, Float, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// GPU memory pool for tensor buffers
pub struct GpuMemoryPool<T: Float> {
    /// Available buffers organized by size
    available: Arc<Mutex<HashMap<usize, Vec<Vec<T>>>>>,
    /// Total allocated memory in bytes
    total_allocated: Arc<Mutex<usize>>,
    /// Maximum memory limit
    max_memory: usize,
    /// Statistics
    stats: Arc<Mutex<PoolStatistics>>,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    /// Number of allocations
    pub allocations: usize,
    /// Number of deallocations
    pub deallocations: usize,
    /// Number of pool hits (reused buffers)
    pub pool_hits: usize,
    /// Number of pool misses (new allocations)
    pub pool_misses: usize,
    /// Peak memory usage
    pub peak_memory: usize,
}

impl<T: Float> GpuMemoryPool<T> {
    /// Create a new GPU memory pool
    pub fn new(max_memory: usize) -> Self {
        Self {
            available: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            max_memory,
            stats: Arc::new(Mutex::new(PoolStatistics::default())),
        }
    }

    /// Allocate a buffer from the pool or create a new one
    pub fn allocate(&self, size: usize) -> Result<Vec<T>> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock statistics"))?;
        stats.allocations += 1;

        let mut available = self
            .available
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock available buffers"))?;

        // Try to reuse from pool
        if let Some(buffers) = available.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                stats.pool_hits += 1;
                return Ok(buffer);
            }
        }

        // No available buffer, create new one
        stats.pool_misses += 1;

        let mut total = self
            .total_allocated
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock total allocated"))?;

        let bytes = size * std::mem::size_of::<T>();
        if *total + bytes > self.max_memory {
            return Err(AutogradError::memory_error(format!(
                "GPU memory limit exceeded: {} + {} > {}",
                *total, bytes, self.max_memory
            )));
        }

        *total += bytes;
        if *total > stats.peak_memory {
            stats.peak_memory = *total;
        }

        Ok(vec![T::zero(); size])
    }

    /// Return a buffer to the pool
    pub fn deallocate(&self, mut buffer: Vec<T>) -> Result<()> {
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock statistics"))?;
        stats.deallocations += 1;

        let size = buffer.len();

        // Clear buffer before returning to pool
        buffer.fill(T::zero());

        let mut available = self
            .available
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock available buffers"))?;

        available.entry(size).or_insert_with(Vec::new).push(buffer);

        Ok(())
    }

    /// Get current statistics
    pub fn statistics(&self) -> Result<PoolStatistics> {
        self.stats
            .lock()
            .map(|s| s.clone())
            .map_err(|_| AutogradError::internal_error("Failed to lock statistics"))
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> Result<usize> {
        self.total_allocated
            .lock()
            .map(|t| *t)
            .map_err(|_| AutogradError::internal_error("Failed to lock total allocated"))
    }

    /// Clear the pool and reset
    pub fn clear(&self) -> Result<()> {
        let mut available = self
            .available
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock available buffers"))?;
        available.clear();

        let mut total = self
            .total_allocated
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock total allocated"))?;
        *total = 0;

        Ok(())
    }
}

/// Memory allocation strategy for gradient computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Eager allocation - allocate all memory upfront
    Eager,
    /// Lazy allocation - allocate on demand
    Lazy,
    /// Pooled allocation - reuse memory from pool
    Pooled,
}

/// Memory manager for GPU gradient computation
pub struct GpuGradientMemory<T: Float> {
    /// Memory pool
    pool: Arc<GpuMemoryPool<T>>,
    /// Allocation strategy
    strategy: AllocationStrategy,
    /// Active allocations
    active: Arc<Mutex<HashMap<usize, Vec<T>>>>,
}

impl<T: Float> GpuGradientMemory<T> {
    /// Create a new GPU gradient memory manager
    pub fn new(max_memory: usize, strategy: AllocationStrategy) -> Self {
        Self {
            pool: Arc::new(GpuMemoryPool::new(max_memory)),
            strategy,
            active: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Allocate memory for gradient
    pub fn allocate_gradient(&self, id: usize, size: usize) -> Result<Vec<T>> {
        let buffer = match self.strategy {
            AllocationStrategy::Eager | AllocationStrategy::Lazy => {
                vec![T::zero(); size]
            }
            AllocationStrategy::Pooled => self.pool.allocate(size)?,
        };

        let mut active = self
            .active
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock active allocations"))?;
        active.insert(id, buffer.clone());

        Ok(buffer)
    }

    /// Free gradient memory
    pub fn free_gradient(&self, id: usize) -> Result<()> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock active allocations"))?;

        if let Some(buffer) = active.remove(&id) {
            if matches!(self.strategy, AllocationStrategy::Pooled) {
                self.pool.deallocate(buffer)?;
            }
        }

        Ok(())
    }

    /// Get memory statistics
    pub fn statistics(&self) -> Result<PoolStatistics> {
        self.pool.statistics()
    }

    /// Get active memory usage
    pub fn active_memory(&self) -> Result<usize> {
        let active = self
            .active
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock active allocations"))?;

        let bytes: usize = active
            .values()
            .map(|v| v.len() * std::mem::size_of::<T>())
            .sum();

        Ok(bytes)
    }
}

/// Gradient checkpointing strategy for memory efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// No checkpointing - store all intermediate values
    None,
    /// Checkpoint every N operations
    Periodic(usize),
    /// Dynamic checkpointing based on memory pressure
    Dynamic,
    /// Optimal checkpointing (Chen et al. 2016)
    Optimal,
}

/// GPU memory-efficient gradient checkpointing
pub struct GpuCheckpointing<T: Float> {
    /// Checkpointing strategy
    strategy: CheckpointStrategy,
    /// Checkpointed tensors
    checkpoints: Arc<Mutex<HashMap<usize, Vec<T>>>>,
    /// Memory budget
    memory_budget: usize,
}

impl<T: Float> GpuCheckpointing<T> {
    /// Create a new GPU checkpointing manager
    pub fn new(strategy: CheckpointStrategy, memory_budget: usize) -> Self {
        Self {
            strategy,
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
            memory_budget,
        }
    }

    /// Decide whether to checkpoint at given operation
    pub fn should_checkpoint(&self, operation_id: usize, memory_used: usize) -> bool {
        match self.strategy {
            CheckpointStrategy::None => false,
            CheckpointStrategy::Periodic(n) => operation_id.is_multiple_of(n),
            CheckpointStrategy::Dynamic => {
                // Checkpoint if memory usage exceeds 80% of budget
                memory_used > (self.memory_budget * 4) / 5
            }
            CheckpointStrategy::Optimal => {
                // Use square root checkpointing strategy
                let checkpoint_interval = (operation_id as f64).sqrt() as usize;
                operation_id.is_multiple_of(checkpoint_interval.max(1))
            }
        }
    }

    /// Save checkpoint
    pub fn save_checkpoint(&self, id: usize, data: Vec<T>) -> Result<()> {
        let mut checkpoints = self
            .checkpoints
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock checkpoints"))?;
        checkpoints.insert(id, data);
        Ok(())
    }

    /// Load checkpoint
    pub fn load_checkpoint(&self, id: usize) -> Result<Option<Vec<T>>> {
        let checkpoints = self
            .checkpoints
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock checkpoints"))?;
        Ok(checkpoints.get(&id).cloned())
    }

    /// Clear all checkpoints
    pub fn clear(&self) -> Result<()> {
        let mut checkpoints = self
            .checkpoints
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock checkpoints"))?;
        checkpoints.clear();
        Ok(())
    }

    /// Get number of checkpoints
    pub fn checkpoint_count(&self) -> Result<usize> {
        let checkpoints = self
            .checkpoints
            .lock()
            .map_err(|_| AutogradError::internal_error("Failed to lock checkpoints"))?;
        Ok(checkpoints.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool: GpuMemoryPool<f32> = GpuMemoryPool::new(1024 * 1024); // 1MB

        // Allocate buffer
        let buffer = pool.allocate(256).expect("Should allocate");
        assert_eq!(buffer.len(), 256);

        // Return to pool
        pool.deallocate(buffer).expect("Should deallocate");

        // Reuse from pool
        let buffer2 = pool.allocate(256).expect("Should allocate from pool");
        assert_eq!(buffer2.len(), 256);

        let stats = pool.statistics().expect("Should get stats");
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.pool_misses, 1);
    }

    #[test]
    fn test_checkpointing_strategy() {
        let checkpointing: GpuCheckpointing<f32> =
            GpuCheckpointing::new(CheckpointStrategy::Periodic(5), 1024);

        assert!(!checkpointing.should_checkpoint(1, 0));
        assert!(!checkpointing.should_checkpoint(4, 0));
        assert!(checkpointing.should_checkpoint(5, 0));
        assert!(checkpointing.should_checkpoint(10, 0));
    }

    #[test]
    fn test_allocation_strategies() {
        let memory = GpuGradientMemory::<f64>::new(1024 * 1024, AllocationStrategy::Pooled);

        let grad = memory.allocate_gradient(0, 100).expect("Should allocate");
        assert_eq!(grad.len(), 100);

        memory.free_gradient(0).expect("Should free");

        let active = memory.active_memory().expect("Should get active memory");
        assert_eq!(active, 0);
    }
}
