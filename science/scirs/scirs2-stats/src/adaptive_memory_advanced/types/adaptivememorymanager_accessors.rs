//! # AdaptiveMemoryManager - accessors Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::MemoryPerformanceMetrics;

use super::adaptivememorymanager_type::AdaptiveMemoryManager;

impl<F> AdaptiveMemoryManager<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + 'static
        + std::fmt::Display,
{
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> MemoryPerformanceMetrics {
        self.performance_monitor.get_current_metrics()
    }

    /// Get the current memory configuration
    pub fn get_config(&self) -> &super::types::AdaptiveMemoryConfig {
        &self.config
    }
}
