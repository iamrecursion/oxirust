//! # AdaptiveMemoryConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveMemoryConfig`.
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

use super::types::{
    AdaptiveMemoryConfig, AllocationStrategy, CacheOptimizationConfig, GarbageCollectionConfig,
    MemoryPressureConfig, NumaConfig, OutOfCoreConfig, PredictiveConfig,
};

impl Default for AdaptiveMemoryConfig {
    fn default() -> Self {
        Self {
            allocation_strategy: AllocationStrategy::Adaptive,
            cache_optimization: CacheOptimizationConfig::default(),
            numa_config: NumaConfig::default(),
            predictive_config: PredictiveConfig::default(),
            pressure_config: MemoryPressureConfig::default(),
            out_of_core_config: OutOfCoreConfig::default(),
            gc_config: GarbageCollectionConfig::default(),
        }
    }
}
