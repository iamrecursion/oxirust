//! Configuration for partitioned (memory-efficient) tensor reductions.

/// Strategy for accumulating partial results across chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulationStrategy {
    Sum,
    Max,
    Min,
    Mean,
    Product,
    LogSumExp,
}

/// Configuration for a partitioned reduction.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Number of elements per partition chunk.
    pub chunk_size: usize,
    /// Optional memory budget in bytes.
    pub max_memory_bytes: Option<usize>,
    /// How to accumulate across chunks.
    pub accumulation: AccumulationStrategy,
    /// Whether to run chunks in parallel (placeholder flag; actual parallelism
    /// requires a rayon dependency that is not present in this crate).
    pub parallel: bool,
    /// Numerical stability epsilon (used in log-sum-exp and division guards).
    pub epsilon: f64,
}

impl PartitionConfig {
    /// Create a new config with the given chunk size and default settings.
    pub fn new(chunk_size: usize) -> Self {
        PartitionConfig {
            chunk_size,
            ..Default::default()
        }
    }

    /// Derive chunk size from a memory budget and the element size in bytes.
    ///
    /// The computed chunk size is `max_bytes / element_size`, clamped to at
    /// least 1.
    pub fn memory_bounded(max_bytes: usize, element_size: usize) -> Self {
        let chunk_size = max_bytes.checked_div(element_size).unwrap_or(1).max(1);
        PartitionConfig {
            chunk_size,
            max_memory_bytes: Some(max_bytes),
            ..Default::default()
        }
    }

    /// Set the accumulation strategy.
    pub fn with_strategy(mut self, strategy: AccumulationStrategy) -> Self {
        self.accumulation = strategy;
        self
    }

    /// Enable or disable parallel chunk processing.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Return the number of chunks needed to process `total_elements` elements.
    pub fn chunks_for_size(&self, total_elements: usize) -> usize {
        if self.chunk_size == 0 {
            return 0;
        }
        total_elements.div_ceil(self.chunk_size)
    }
}

impl Default for PartitionConfig {
    fn default() -> Self {
        PartitionConfig {
            chunk_size: 4096,
            max_memory_bytes: None,
            accumulation: AccumulationStrategy::Sum,
            parallel: false,
            epsilon: 1e-12,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_config_new() {
        let cfg = PartitionConfig::new(1024);
        assert_eq!(cfg.chunk_size, 1024);
        assert!(cfg.max_memory_bytes.is_none());
        assert_eq!(cfg.accumulation, AccumulationStrategy::Sum);
        assert!(!cfg.parallel);
    }

    #[test]
    fn test_partition_config_memory_bounded() {
        // 64 bytes / 8 bytes per f64 = 8 elements per chunk
        let cfg = PartitionConfig::memory_bounded(64, 8);
        assert_eq!(cfg.chunk_size, 8);
        assert_eq!(cfg.max_memory_bytes, Some(64));
    }

    #[test]
    fn test_chunks_for_size() {
        let cfg = PartitionConfig::new(10);
        assert_eq!(cfg.chunks_for_size(0), 0);
        assert_eq!(cfg.chunks_for_size(10), 1);
        assert_eq!(cfg.chunks_for_size(11), 2);
        assert_eq!(cfg.chunks_for_size(100), 10);
        assert_eq!(cfg.chunks_for_size(101), 11);
    }
}
