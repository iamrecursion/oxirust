//! # AdaptiveMemoryManager - extract_allocation_features_group Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::AllocationContext;

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
    /// Extract features for allocation strategy prediction
    pub fn extract_allocation_features(&self, context: &AllocationContext) -> Vec<f64> {
        vec![
            context.size as f64,
            context.current_pressure,
            context.predicted_usage,
            context.numa_node.unwrap_or(0) as f64,
            context.allocation_type as u8 as f64,
            context.urgency as u8 as f64,
        ]
    }
}
