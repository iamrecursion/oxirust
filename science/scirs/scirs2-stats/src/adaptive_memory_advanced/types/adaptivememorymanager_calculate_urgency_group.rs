//! # AdaptiveMemoryManager - calculate_urgency_group Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::AllocationUrgency;

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
    /// Calculate allocation urgency
    pub fn calculate_urgency(&self, size: usize, pressure: f64) -> AllocationUrgency {
        if pressure > 0.95 {
            AllocationUrgency::Critical
        } else if pressure > 0.85 {
            AllocationUrgency::High
        } else if pressure > 0.7 {
            AllocationUrgency::Normal
        } else {
            AllocationUrgency::Low
        }
    }
}
