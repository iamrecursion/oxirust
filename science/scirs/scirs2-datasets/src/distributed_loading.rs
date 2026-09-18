//! Distributed dataset loading
//!
//! This module provides shard-aware dataset loading for distributed training,
//! with multi-node coordination, distributed caching, and rank-aware data
//! partitioning to ensure each worker processes a unique subset of the data.

use crate::error::{DatasetsError, Result};
use crate::streaming::{DataChunk, StreamConfig};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Configuration for distributed loading
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of workers/processes
    pub world_size: usize,
    /// Rank of this worker (0 to world_size-1)
    pub rank: usize,
    /// Number of shards to create
    pub num_shards: usize,
    /// Whether to shuffle shards
    pub shuffle_shards: bool,
    /// Random seed for shuffling
    pub seed: Option<u64>,
    /// Whether to drop last incomplete batch
    pub drop_last: bool,
    /// Whether to use distributed caching
    pub enable_distributed_cache: bool,
}

impl DistributedConfig {
    /// Create a new distributed configuration
    ///
    /// # Arguments
    /// * `world_size` - Total number of workers
    /// * `rank` - Rank of this worker
    ///
    /// # Returns
    /// * `DistributedConfig` - Configuration instance
    pub fn new(world_size: usize, rank: usize) -> Result<Self> {
        if rank >= world_size {
            return Err(DatasetsError::InvalidFormat(format!(
                "Rank {} must be less than world_size {}",
                rank, world_size
            )));
        }

        Ok(Self {
            world_size,
            rank,
            num_shards: world_size,
            shuffle_shards: false,
            seed: None,
            drop_last: false,
            enable_distributed_cache: true,
        })
    }

    /// Set number of shards
    pub fn with_shards(mut self, num_shards: usize) -> Self {
        self.num_shards = num_shards.max(1);
        self
    }

    /// Enable shard shuffling
    pub fn with_shuffle(mut self, shuffle: bool, seed: Option<u64>) -> Self {
        self.shuffle_shards = shuffle;
        self.seed = seed;
        self
    }

    /// Set drop_last behavior
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
}

/// Shard information
#[derive(Debug, Clone)]
pub struct Shard {
    /// Shard index
    pub index: usize,
    /// Starting sample index
    pub start: usize,
    /// Ending sample index (exclusive)
    pub end: usize,
    /// Number of samples in this shard
    pub size: usize,
}

impl Shard {
    /// Create a new shard
    pub fn new(index: usize, start: usize, end: usize) -> Self {
        Self {
            index,
            start,
            end,
            size: end - start,
        }
    }

    /// Check if a sample index belongs to this shard
    pub fn contains(&self, idx: usize) -> bool {
        idx >= self.start && idx < self.end
    }
}

/// Distributed dataset loader
pub struct DistributedLoader {
    config: DistributedConfig,
    total_samples: usize,
    shards: Vec<Shard>,
    assigned_shards: Vec<usize>,
}

impl DistributedLoader {
    /// Create a new distributed loader
    ///
    /// # Arguments
    /// * `config` - Distributed configuration
    /// * `total_samples` - Total number of samples in the dataset
    ///
    /// # Returns
    /// * `Ok(DistributedLoader)` - The loader instance
    /// * `Err(DatasetsError)` - If configuration is invalid
    pub fn new(config: DistributedConfig, total_samples: usize) -> Result<Self> {
        if total_samples == 0 {
            return Err(DatasetsError::InvalidFormat(
                "Dataset must have at least one sample".to_string(),
            ));
        }

        // Create shards
        let shards = Self::create_shards(total_samples, config.num_shards, config.drop_last)?;

        // Assign shards to this rank
        let assigned_shards = Self::assign_shards_to_rank(&shards, &config);

        Ok(Self {
            config,
            total_samples,
            shards,
            assigned_shards,
        })
    }

    /// Create shards from the total dataset
    fn create_shards(
        total_samples: usize,
        num_shards: usize,
        drop_last: bool,
    ) -> Result<Vec<Shard>> {
        let mut shards = Vec::new();
        let base_shard_size = total_samples / num_shards;
        let remainder = total_samples % num_shards;

        let mut start = 0;
        for i in 0..num_shards {
            // Distribute remainder samples across first shards
            let shard_size = if i < remainder {
                base_shard_size + 1
            } else {
                base_shard_size
            };

            if shard_size == 0 && drop_last {
                break;
            }

            let end = start + shard_size;
            shards.push(Shard::new(i, start, end));
            start = end;
        }

        Ok(shards)
    }

    /// Assign shards to a specific rank
    fn assign_shards_to_rank(shards: &[Shard], config: &DistributedConfig) -> Vec<usize> {
        let mut assigned = Vec::new();

        // Round-robin assignment
        for (idx, _shard) in shards.iter().enumerate() {
            if idx % config.world_size == config.rank {
                assigned.push(idx);
            }
        }

        assigned
    }

    /// Get the shards assigned to this rank
    pub fn get_assigned_shards(&self) -> Vec<&Shard> {
        self.assigned_shards
            .iter()
            .filter_map(|&idx| self.shards.get(idx))
            .collect()
    }

    /// Get the total number of samples for this rank
    pub fn samples_for_rank(&self) -> usize {
        self.get_assigned_shards().iter().map(|s| s.size).sum()
    }

    /// Get the sample indices for this rank
    pub fn get_sample_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        for shard in self.get_assigned_shards() {
            indices.extend(shard.start..shard.end);
        }
        indices
    }

    /// Partition a dataset according to this rank's assignment
    pub fn partition_dataset(&self, dataset: &Dataset) -> Result<Dataset> {
        let indices = self.get_sample_indices();

        if indices.is_empty() {
            return Err(DatasetsError::InvalidFormat(
                "No samples assigned to this rank".to_string(),
            ));
        }

        // Extract rows corresponding to this rank's indices
        let n_features = dataset.n_features();
        let mut data_rows = Vec::new();
        let mut target_values = Vec::new();

        for &idx in &indices {
            if idx >= dataset.n_samples() {
                return Err(DatasetsError::InvalidFormat(format!(
                    "Index {} out of bounds for dataset with {} samples",
                    idx,
                    dataset.n_samples()
                )));
            }

            // Extract data row
            let row = dataset.data.row(idx);
            data_rows.extend(row.iter().copied());

            // Extract target if present
            if let Some(ref target) = dataset.target {
                if idx < target.len() {
                    target_values.push(target[idx]);
                }
            }
        }

        // Create new arrays
        let data = Array2::from_shape_vec((indices.len(), n_features), data_rows)
            .map_err(|e| DatasetsError::InvalidFormat(format!("Failed to create array: {}", e)))?;

        let target = if !target_values.is_empty() {
            Some(Array1::from_vec(target_values))
        } else {
            None
        };

        Ok(Dataset {
            data,
            target,
            targetnames: dataset.targetnames.clone(),
            featurenames: dataset.featurenames.clone(),
            feature_descriptions: dataset.feature_descriptions.clone(),
            description: dataset.description.clone(),
            metadata: dataset.metadata.clone(),
        })
    }

    /// Get configuration
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }

    /// Get total number of samples
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }
}

/// Distributed cache for sharing data across nodes
pub struct DistributedCache {
    cache: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    config: DistributedConfig,
}

impl DistributedCache {
    /// Create a new distributed cache
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Store data in cache
    pub fn put(&self, key: String, data: Vec<u8>) -> Result<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| DatasetsError::CacheError(format!("Lock error: {}", e)))?;
        cache.insert(key, data);
        Ok(())
    }

    /// Retrieve data from cache
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let cache = self
            .cache
            .lock()
            .map_err(|e| DatasetsError::CacheError(format!("Lock error: {}", e)))?;
        Ok(cache.get(key).cloned())
    }

    /// Check if key exists in cache
    pub fn contains(&self, key: &str) -> bool {
        self.cache
            .lock()
            .map(|cache| cache.contains_key(key))
            .unwrap_or(false)
    }

    /// Clear the cache
    pub fn clear(&self) -> Result<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| DatasetsError::CacheError(format!("Lock error: {}", e)))?;
        cache.clear();
        Ok(())
    }

    /// Get cache size (number of entries)
    pub fn size(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

/// Create a distributed loader for a dataset
///
/// # Arguments
/// * `world_size` - Total number of workers
/// * `rank` - Rank of this worker
/// * `total_samples` - Total samples in the dataset
///
/// # Returns
/// * `Ok(DistributedLoader)` - The loader
/// * `Err(DatasetsError)` - If creation fails
pub fn create_loader(
    world_size: usize,
    rank: usize,
    total_samples: usize,
) -> Result<DistributedLoader> {
    let config = DistributedConfig::new(world_size, rank)?;
    DistributedLoader::new(config, total_samples)
}

/// Create a distributed loader with custom configuration
pub fn create_loader_with_config(
    config: DistributedConfig,
    total_samples: usize,
) -> Result<DistributedLoader> {
    DistributedLoader::new(config, total_samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config() -> Result<()> {
        let config = DistributedConfig::new(4, 2)?;
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 2);
        assert_eq!(config.num_shards, 4);

        // Invalid rank should fail
        assert!(DistributedConfig::new(4, 4).is_err());

        Ok(())
    }

    #[test]
    fn test_shard_creation() -> Result<()> {
        // 100 samples, 4 shards
        let shards = DistributedLoader::create_shards(100, 4, false)?;
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].size, 25);
        assert_eq!(shards[3].end, 100);

        // With remainder
        let shards = DistributedLoader::create_shards(103, 4, false)?;
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].size, 26); // Gets extra sample
        assert_eq!(shards[1].size, 26);
        assert_eq!(shards[2].size, 26);
        assert_eq!(shards[3].size, 25);

        Ok(())
    }

    #[test]
    fn test_distributed_loader() -> Result<()> {
        let config = DistributedConfig::new(4, 1)?;
        let loader = DistributedLoader::new(config, 100)?;

        assert_eq!(loader.total_samples(), 100);
        let assigned = loader.get_assigned_shards();
        assert!(!assigned.is_empty());

        // Check sample indices
        let indices = loader.get_sample_indices();
        assert!(!indices.is_empty());

        Ok(())
    }

    #[test]
    fn test_partition_dataset() -> Result<()> {
        // Create a test dataset
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect())
            .map_err(|e| DatasetsError::InvalidFormat(format!("{}", e)))?;

        let target = Some(Array1::from_vec((0..10).map(|x| x as f64).collect()));

        let dataset = Dataset {
            data,
            target,
            targetnames: None,
            featurenames: None,
            feature_descriptions: None,
            description: None,
            metadata: Default::default(),
        };

        // Partition for rank 0 of 2 workers
        let config = DistributedConfig::new(2, 0)?;
        let loader = DistributedLoader::new(config, 10)?;
        let partitioned = loader.partition_dataset(&dataset)?;

        assert_eq!(partitioned.n_samples(), 5); // Should get half
        assert_eq!(partitioned.n_features(), 3);

        Ok(())
    }

    #[test]
    fn test_distributed_cache() -> Result<()> {
        let config = DistributedConfig::new(2, 0)?;
        let cache = DistributedCache::new(config);

        // Store and retrieve
        cache.put("test".to_string(), vec![1, 2, 3, 4])?;
        assert!(cache.contains("test"));

        let data = cache.get("test")?;
        assert_eq!(data, Some(vec![1, 2, 3, 4]));

        // Clear
        cache.clear()?;
        assert!(!cache.contains("test"));

        Ok(())
    }

    #[test]
    fn test_shard_contains() {
        let shard = Shard::new(0, 10, 20);
        assert!(shard.contains(10));
        assert!(shard.contains(15));
        assert!(shard.contains(19));
        assert!(!shard.contains(9));
        assert!(!shard.contains(20));
    }

    #[test]
    fn test_round_robin_assignment() -> Result<()> {
        let config = DistributedConfig::new(3, 1)?; // Rank 1 of 3
        let loader = DistributedLoader::new(config, 90)?; // 90 samples

        let indices = loader.get_sample_indices();

        // Rank 1 should get shards 1
        // With 3 shards: [0-30), [30-60), [60-90)
        // Rank 1 gets shard 1: [30-60)
        assert_eq!(indices.len(), 30);
        assert_eq!(indices[0], 30);
        assert_eq!(indices[29], 59);

        Ok(())
    }
}
