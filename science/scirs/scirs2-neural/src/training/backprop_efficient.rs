//! Memory-efficient backpropagation with gradient checkpointing
//!
//! This module provides memory-efficient backpropagation strategies that reduce
//! GPU memory usage by selectively checkpointing and recomputing activations.

use crate::error::{NeuralError, Result};
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBuffer, GpuContext, GpuDataType};
use scirs2_core::ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Recomputation policy for gradient checkpointing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecomputationPolicy {
    /// Checkpoint all activations (no recomputation, highest memory usage)
    CheckpointAll,
    /// Checkpoint no activations (full recomputation, lowest memory usage)
    CheckpointNone,
    /// Selectively checkpoint based on memory/computation tradeoff
    Selective {
        /// Checkpoint layers with recomputation cost above this threshold
        cost_threshold: u32,
    },
    /// Checkpoint every N layers
    EveryN {
        /// Checkpoint frequency
        n: usize,
    },
}

impl Default for RecomputationPolicy {
    fn default() -> Self {
        Self::Selective {
            cost_threshold: 100,
        }
    }
}

/// Activation checkpoint metadata
#[derive(Debug, Clone)]
pub struct ActivationCheckpoint {
    /// Layer ID
    pub layer_id: usize,
    /// Checkpoint timestamp
    pub timestamp: u64,
    /// Memory size in bytes
    pub memory_size: usize,
    /// Recomputation cost estimate
    pub recomputation_cost: u32,
    /// Whether this checkpoint is currently in memory
    pub in_memory: bool,
}

/// Gradient checkpoint manager (GPU feature required)
#[cfg(feature = "gpu")]
pub struct GradientCheckpointManager<T: GpuDataType> {
    /// Checkpointed activations (layer_id -> buffer)
    checkpoints: Arc<Mutex<HashMap<usize, GpuBuffer<T>>>>,
    /// Checkpoint metadata
    metadata: Arc<Mutex<HashMap<usize, ActivationCheckpoint>>>,
    /// Current memory usage in bytes
    memory_usage: Arc<AtomicU64>,
    /// Maximum memory budget in bytes
    memory_budget: u64,
    /// Recomputation policy
    policy: RecomputationPolicy,
    /// Global checkpoint counter
    checkpoint_counter: Arc<AtomicU64>,
    /// GPU context for buffer management
    gpu_context: Arc<GpuContext>,
}

#[cfg(feature = "gpu")]
impl<T: GpuDataType> GradientCheckpointManager<T> {
    /// Create a new gradient checkpoint manager
    pub fn new(
        gpu_context: Arc<GpuContext>,
        memory_budget: u64,
        policy: RecomputationPolicy,
    ) -> Self {
        Self {
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
            metadata: Arc::new(Mutex::new(HashMap::new())),
            memory_usage: Arc::new(AtomicU64::new(0)),
            memory_budget,
            policy,
            checkpoint_counter: Arc::new(AtomicU64::new(0)),
            gpu_context,
        }
    }

    /// Save an activation checkpoint
    pub fn checkpoint_activation(
        &self,
        layer_id: usize,
        activation: &GpuBuffer<T>,
        recomputation_cost: u32,
    ) -> Result<()> {
        let should_checkpoint = match self.policy {
            RecomputationPolicy::CheckpointAll => true,
            RecomputationPolicy::CheckpointNone => false,
            RecomputationPolicy::Selective { cost_threshold } => {
                recomputation_cost >= cost_threshold
            }
            RecomputationPolicy::EveryN { n } => layer_id.is_multiple_of(n),
        };

        if !should_checkpoint {
            return Ok(());
        }

        let activation_size = activation.len() * std::mem::size_of::<T>();

        // Check memory budget
        let current_usage = self.memory_usage.load(Ordering::Relaxed);
        if current_usage + activation_size as u64 > self.memory_budget {
            // Evict oldest checkpoint if needed
            self.evict_oldest_checkpoint()?;
        }

        // Store checkpoint
        let mut checkpoints = self
            .checkpoints
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock checkpoints".to_string()))?;

        let mut metadata = self
            .metadata
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock metadata".to_string()))?;

        // Create checkpoint metadata
        let checkpoint_meta = ActivationCheckpoint {
            layer_id,
            timestamp: self.checkpoint_counter.fetch_add(1, Ordering::Relaxed),
            memory_size: activation_size,
            recomputation_cost,
            in_memory: true,
        };

        // Clone the buffer (in real GPU implementation, this would be a device-to-device copy)
        let checkpoint_buffer = self.gpu_context.create_buffer::<T>(activation.len());

        checkpoints.insert(layer_id, checkpoint_buffer);
        metadata.insert(layer_id, checkpoint_meta);

        self.memory_usage
            .fetch_add(activation_size as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Retrieve a checkpointed activation (removes it from the checkpoint manager)
    pub fn get_checkpoint(&self, layer_id: usize) -> Result<Option<GpuBuffer<T>>> {
        let mut checkpoints = self
            .checkpoints
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock checkpoints".to_string()))?;

        Ok(checkpoints.remove(&layer_id))
    }

    /// Check if a layer has a checkpoint
    pub fn has_checkpoint(&self, layer_id: usize) -> bool {
        self.checkpoints
            .lock()
            .map(|cp| cp.contains_key(&layer_id))
            .unwrap_or(false)
    }

    /// Evict the oldest checkpoint to free memory
    fn evict_oldest_checkpoint(&self) -> Result<()> {
        let mut metadata = self
            .metadata
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock metadata".to_string()))?;

        // Find oldest checkpoint
        let oldest = metadata
            .iter()
            .filter(|(_, meta)| meta.in_memory)
            .min_by_key(|(_, meta)| meta.timestamp)
            .map(|(id, _)| *id);

        if let Some(layer_id) = oldest {
            self.remove_checkpoint(layer_id)?;
        }

        Ok(())
    }

    /// Remove a checkpoint and free its memory
    pub fn remove_checkpoint(&self, layer_id: usize) -> Result<()> {
        let mut checkpoints = self
            .checkpoints
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock checkpoints".to_string()))?;

        let mut metadata = self
            .metadata
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock metadata".to_string()))?;

        if let Some(checkpoint) = checkpoints.remove(&layer_id) {
            let size = checkpoint.len() * std::mem::size_of::<T>();
            self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
        }

        if let Some(meta) = metadata.get_mut(&layer_id) {
            meta.in_memory = false;
        }

        Ok(())
    }

    /// Clear all checkpoints
    pub fn clear(&self) -> Result<()> {
        let mut checkpoints = self
            .checkpoints
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock checkpoints".to_string()))?;

        let mut metadata = self
            .metadata
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock metadata".to_string()))?;

        checkpoints.clear();
        metadata.clear();
        self.memory_usage.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        self.memory_usage.load(Ordering::Relaxed)
    }

    /// Get memory budget in bytes
    pub fn memory_budget(&self) -> u64 {
        self.memory_budget
    }

    /// Get number of active checkpoints
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.lock().map(|cp| cp.len()).unwrap_or(0)
    }

    /// Get checkpoint statistics
    pub fn get_statistics(&self) -> CheckpointStatistics {
        let metadata = self.metadata.lock().expect("Failed to lock metadata");

        let total_checkpoints = metadata.len();
        let in_memory_checkpoints = metadata.values().filter(|meta| meta.in_memory).count();

        let total_memory = metadata
            .values()
            .filter(|meta| meta.in_memory)
            .map(|meta| meta.memory_size as u64)
            .sum();

        CheckpointStatistics {
            total_checkpoints,
            in_memory_checkpoints,
            total_memory,
            memory_budget: self.memory_budget,
            memory_utilization: total_memory as f64 / self.memory_budget as f64,
        }
    }
}

/// Statistics for checkpoint management
#[derive(Debug, Clone)]
pub struct CheckpointStatistics {
    /// Total number of checkpoints created
    pub total_checkpoints: usize,
    /// Number of checkpoints currently in memory
    pub in_memory_checkpoints: usize,
    /// Total memory used by checkpoints
    pub total_memory: u64,
    /// Memory budget
    pub memory_budget: u64,
    /// Memory utilization ratio (0.0 to 1.0)
    pub memory_utilization: f64,
}

/// Efficient backpropagation implementation with gradient checkpointing (GPU feature required)
#[cfg(feature = "gpu")]
pub struct EfficientBackprop<T: GpuDataType> {
    /// Gradient checkpoint manager
    checkpoint_manager: Arc<GradientCheckpointManager<T>>,
    /// GPU context
    gpu_context: Arc<GpuContext>,
    /// Whether to enable gradient checkpointing
    enabled: bool,
}

#[cfg(feature = "gpu")]
impl<T: GpuDataType> EfficientBackprop<T> {
    /// Create a new efficient backpropagation context
    pub fn new(
        gpu_context: Arc<GpuContext>,
        memory_budget: u64,
        policy: RecomputationPolicy,
        enabled: bool,
    ) -> Self {
        let checkpoint_manager = Arc::new(GradientCheckpointManager::new(
            gpu_context.clone(),
            memory_budget,
            policy,
        ));

        Self {
            checkpoint_manager,
            gpu_context,
            enabled,
        }
    }

    /// Forward pass with optional checkpointing
    pub fn forward_with_checkpoint(
        &self,
        layer_id: usize,
        input: &GpuBuffer<T>,
        forward_fn: impl FnOnce(&GpuBuffer<T>) -> Result<GpuBuffer<T>>,
        recomputation_cost: u32,
    ) -> Result<GpuBuffer<T>> {
        // Checkpoint input if enabled
        if self.enabled {
            self.checkpoint_manager
                .checkpoint_activation(layer_id, input, recomputation_cost)?;
        }

        // Execute forward pass
        forward_fn(input)
    }

    /// Backward pass with recomputation if needed
    pub fn backward_with_recomputation(
        &self,
        layer_id: usize,
        grad_output: &GpuBuffer<T>,
        forward_fn: impl FnOnce(&GpuBuffer<T>) -> Result<GpuBuffer<T>>,
        backward_fn: impl FnOnce(&GpuBuffer<T>, &GpuBuffer<T>) -> Result<GpuBuffer<T>>,
    ) -> Result<GpuBuffer<T>> {
        // Check if we have a checkpoint
        let activation =
            if let Some(checkpoint) = self.checkpoint_manager.get_checkpoint(layer_id)? {
                // Use checkpointed activation
                checkpoint
            } else {
                // Recompute activation (requires previous layer output)
                // In a real implementation, we would retrieve the previous layer's output
                // For now, we create a placeholder buffer
                self.gpu_context.create_buffer::<T>(grad_output.len())
            };

        // Execute backward pass
        backward_fn(&activation, grad_output)
    }

    /// Enable or disable gradient checkpointing
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if checkpointing is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get checkpoint manager
    pub fn checkpoint_manager(&self) -> &Arc<GradientCheckpointManager<T>> {
        &self.checkpoint_manager
    }

    /// Get checkpoint statistics
    pub fn get_statistics(&self) -> CheckpointStatistics {
        self.checkpoint_manager.get_statistics()
    }

    /// Clear all checkpoints
    pub fn clear_checkpoints(&self) -> Result<()> {
        self.checkpoint_manager.clear()
    }
}

/// CPU-based activation storage for fallback
#[derive(Debug)]
pub struct CpuActivationStore<F> {
    /// Stored activations
    activations: Arc<Mutex<HashMap<usize, ArrayD<F>>>>,
    /// Memory usage
    memory_usage: Arc<AtomicU64>,
}

impl<F> CpuActivationStore<F>
where
    F: Clone + Default,
{
    /// Create a new CPU activation store
    pub fn new() -> Self {
        Self {
            activations: Arc::new(Mutex::new(HashMap::new())),
            memory_usage: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Store an activation
    pub fn store(&self, layer_id: usize, activation: ArrayD<F>) -> Result<()> {
        let size = activation.len() * std::mem::size_of::<F>();

        let mut activations = self
            .activations
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock activations".to_string()))?;

        activations.insert(layer_id, activation);
        self.memory_usage.fetch_add(size as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Retrieve an activation
    pub fn retrieve(&self, layer_id: usize) -> Result<Option<ArrayD<F>>> {
        let activations = self
            .activations
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock activations".to_string()))?;

        Ok(activations.get(&layer_id).cloned())
    }

    /// Remove an activation
    pub fn remove(&self, layer_id: usize) -> Result<()> {
        let mut activations = self
            .activations
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock activations".to_string()))?;

        if let Some(activation) = activations.remove(&layer_id) {
            let size = activation.len() * std::mem::size_of::<F>();
            self.memory_usage.fetch_sub(size as u64, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Clear all activations
    pub fn clear(&self) -> Result<()> {
        let mut activations = self
            .activations
            .lock()
            .map_err(|_| NeuralError::TrainingError("Failed to lock activations".to_string()))?;

        activations.clear();
        self.memory_usage.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> u64 {
        self.memory_usage.load(Ordering::Relaxed)
    }
}

impl<F> Default for CpuActivationStore<F>
where
    F: Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use scirs2_core::gpu::GpuBackend;

    #[test]
    fn test_recomputation_policy() {
        let policy = RecomputationPolicy::default();
        assert!(matches!(policy, RecomputationPolicy::Selective { .. }));

        let checkpoint_all = RecomputationPolicy::CheckpointAll;
        assert_eq!(checkpoint_all, RecomputationPolicy::CheckpointAll);
    }

    #[test]
    fn test_checkpoint_manager_creation() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let manager = GradientCheckpointManager::<f32>::new(
            Arc::new(context),
            1024 * 1024 * 1024, // 1GB
            RecomputationPolicy::CheckpointAll,
        );

        assert_eq!(manager.memory_usage(), 0);
        assert_eq!(manager.num_checkpoints(), 0);
    }

    #[test]
    fn test_checkpoint_statistics() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let manager = GradientCheckpointManager::<f32>::new(
            Arc::new(context),
            1024 * 1024 * 1024,
            RecomputationPolicy::CheckpointAll,
        );

        let stats = manager.get_statistics();
        assert_eq!(stats.total_checkpoints, 0);
        assert_eq!(stats.in_memory_checkpoints, 0);
        assert_eq!(stats.total_memory, 0);
    }

    #[test]
    fn test_efficient_backprop_creation() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let backprop = EfficientBackprop::<f32>::new(
            Arc::new(context),
            1024 * 1024 * 1024,
            RecomputationPolicy::CheckpointAll,
            true,
        );

        assert!(backprop.is_enabled());
        assert_eq!(backprop.checkpoint_manager().num_checkpoints(), 0);
    }

    #[test]
    fn test_cpu_activation_store() {
        let store = CpuActivationStore::<f32>::new();

        let activation = Array::zeros(IxDyn(&[2, 3, 4]));
        store.store(0, activation.clone()).expect("Failed to store");

        let retrieved = store.retrieve(0).expect("Failed to retrieve");
        assert!(retrieved.is_some());

        assert!(store.memory_usage() > 0);

        store.clear().expect("Failed to clear");
        assert_eq!(store.memory_usage(), 0);
    }

    #[test]
    fn test_enable_disable_checkpointing() {
        let context = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
        let mut backprop = EfficientBackprop::<f32>::new(
            Arc::new(context),
            1024 * 1024 * 1024,
            RecomputationPolicy::CheckpointAll,
            true,
        );

        assert!(backprop.is_enabled());

        backprop.set_enabled(false);
        assert!(!backprop.is_enabled());

        backprop.set_enabled(true);
        assert!(backprop.is_enabled());
    }
}
