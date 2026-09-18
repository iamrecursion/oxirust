//! # AdaptiveMemoryManager - update_config_group Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::AdaptiveMemoryConfig;

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
    /// Update configuration
    pub fn update_config(&mut self, config: AdaptiveMemoryConfig) {
        self.config = config;
    }
}
