//! Data-parallel training utilities

use crate::{Float, NdArray, Result};

/// Data shard for distributed training
pub struct DataShard<T: Float> {
    /// Shard data
    pub data: Vec<NdArray<T>>,
    /// Shard rank
    pub rank: usize,
    /// Total shards
    pub total_shards: usize,
}

impl<T: Float> DataShard<T> {
    /// Create a new data shard
    pub fn new(data: Vec<NdArray<T>>, rank: usize, total_shards: usize) -> Self {
        Self {
            data,
            rank,
            total_shards,
        }
    }

    /// Get shard size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if shard is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Split data into shards for data-parallel training
pub fn shard_data<T: Float>(data: Vec<NdArray<T>>, num_shards: usize) -> Result<Vec<DataShard<T>>> {
    let shard_size = (data.len() + num_shards - 1) / num_shards;
    let mut shards = Vec::with_capacity(num_shards);

    for rank in 0..num_shards {
        let start = rank * shard_size;
        let end = ((rank + 1) * shard_size).min(data.len());

        let shard_data = data[start..end].to_vec();
        shards.push(DataShard::new(shard_data, rank, num_shards));
    }

    Ok(shards)
}
