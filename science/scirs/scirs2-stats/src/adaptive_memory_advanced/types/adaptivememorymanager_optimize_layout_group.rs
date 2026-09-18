//! # AdaptiveMemoryManager - optimize_layout_group Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

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
    /// Optimize memory layout for better cache performance
    pub fn optimize_layout<T>(&self, data: &mut ArrayView2<T>) -> StatsResult<()>
    where
        T: Clone + Send + Sync,
    {
        self.cache_manager.optimize_layout(data)
    }
}
