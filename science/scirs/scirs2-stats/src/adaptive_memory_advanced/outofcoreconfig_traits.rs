//! # OutOfCoreConfig - Trait Implementations
//!
//! This module contains trait implementations for `OutOfCoreConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::{ChunkSchedulingStrategy, CompressionConfig, OutOfCoreConfig, StorageConfig};

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            enable_out_of_core: true,
            chunksize: 64 * 1024 * 1024,
            memory_chunks: 16,
            storage_config: StorageConfig::default(),
            compression_config: CompressionConfig::default(),
            scheduling_strategy: ChunkSchedulingStrategy::Adaptive,
        }
    }
}
