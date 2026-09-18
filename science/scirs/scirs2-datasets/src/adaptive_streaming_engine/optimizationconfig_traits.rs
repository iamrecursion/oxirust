//! # OptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::{MemoryStrategy, OptimizationConfig};

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            optimal_batch_size: 1000,
            optimal_buffer_size: 10000,
            num_workers: num_cpus::get(),
            memory_strategy: MemoryStrategy::Balanced,
        }
    }
}
