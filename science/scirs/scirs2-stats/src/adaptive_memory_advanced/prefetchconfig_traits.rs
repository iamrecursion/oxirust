//! # PrefetchConfig - Trait Implementations
//!
//! This module contains trait implementations for `PrefetchConfig`.
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

use super::types::PrefetchConfig;

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            enable_software_prefetch: true,
            enable_hardware_hints: true,
            prefetch_distance: 8,
            temporal_awareness: true,
            spatial_awareness: true,
            predictive_prefetch: true,
        }
    }
}
