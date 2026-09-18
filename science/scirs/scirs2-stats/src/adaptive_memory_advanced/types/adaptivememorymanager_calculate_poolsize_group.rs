//! # AdaptiveMemoryManager - calculate_poolsize_group Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult};
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
    /// Calculate appropriate pool size for allocation
    pub fn calculate_poolsize(&self, size: usize) -> usize {
        let mut poolsize = 1;
        while poolsize < size {
            poolsize *= 2;
        }
        poolsize
    }
    /// Pool deallocation
    pub fn deallocate_pool(&self, ptr: *mut u8, size: usize) -> StatsResult<()> {
        let poolsize = self.calculate_poolsize(size);
        if let Some(pool) = self
            .memory_pools
            .read()
            .expect("Operation failed")
            .get(&poolsize)
        {
            pool.deallocate(ptr)
        } else {
            Err(StatsError::InvalidArgument("Pool not found".to_string()))
        }
    }
}
