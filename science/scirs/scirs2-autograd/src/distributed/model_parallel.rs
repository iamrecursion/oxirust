//! Model-parallel training utilities

use crate::{Float, Result};

/// Model partition for model-parallel training
pub struct ModelPartition {
    /// Partition ID
    pub id: usize,
    /// Layer start index
    pub layer_start: usize,
    /// Layer end index
    pub layer_end: usize,
}

impl ModelPartition {
    /// Create a new model partition
    pub fn new(id: usize, layer_start: usize, layer_end: usize) -> Self {
        Self {
            id,
            layer_start,
            layer_end,
        }
    }

    /// Get number of layers in partition
    pub fn num_layers(&self) -> usize {
        self.layer_end - self.layer_start
    }
}

/// Partition a model into chunks for model-parallel training
pub fn partition_model(num_layers: usize, num_partitions: usize) -> Result<Vec<ModelPartition>> {
    let layers_per_partition = (num_layers + num_partitions - 1) / num_partitions;
    let mut partitions = Vec::with_capacity(num_partitions);

    for id in 0..num_partitions {
        let start = id * layers_per_partition;
        let end = ((id + 1) * layers_per_partition).min(num_layers);

        partitions.push(ModelPartition::new(id, start, end));
    }

    Ok(partitions)
}
