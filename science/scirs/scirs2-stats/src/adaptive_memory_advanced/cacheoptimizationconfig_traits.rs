//! # CacheOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `CacheOptimizationConfig`.
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
    AccessPatternConfig, CacheHierarchy, CacheOptimizationConfig, DataLayoutStrategy,
    PrefetchConfig,
};

impl Default for CacheOptimizationConfig {
    fn default() -> Self {
        Self {
            cache_hierarchy: CacheHierarchy::detect(),
            layout_strategy: DataLayoutStrategy::Adaptive,
            prefetch_config: PrefetchConfig::default(),
            cache_line_optimization: true,
            pattern_analysis: AccessPatternConfig::default(),
        }
    }
}
