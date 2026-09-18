//! # AdaptiveMemoryManager - infer_allocation_type_group Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::AllocationType;

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
    /// Infer allocation type from size and context
    pub fn infer_allocation_type(&self, size: usize) -> AllocationType {
        #[cfg(target_pointer_width = "32")]
        let huge_threshold = 256 * 1024 * 1024; // 256MB for 32-bit
        #[cfg(target_pointer_width = "64")]
        let huge_threshold = 1024 * 1024 * 1024; // 1GB for 64-bit

        if size < 1024 {
            AllocationType::SmallObject
        } else if size < 1024 * 1024 {
            AllocationType::LargeObject
        } else if size < huge_threshold {
            AllocationType::HugeObject
        } else {
            AllocationType::HugeObject
        }
    }
}
