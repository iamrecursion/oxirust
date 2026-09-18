//! # NumaConfig - Trait Implementations
//!
//! This module contains trait implementations for `NumaConfig`.
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

use super::types::{NumaBindingStrategy, NumaConfig, NumaMigrationPolicy};

impl Default for NumaConfig {
    fn default() -> Self {
        Self {
            enable_numa: true,
            auto_detect_topology: true,
            binding_strategy: NumaBindingStrategy::Adaptive,
            thread_affinity: true,
            optimize_communication: true,
            migration_policy: NumaMigrationPolicy::OnPatternChange,
        }
    }
}
