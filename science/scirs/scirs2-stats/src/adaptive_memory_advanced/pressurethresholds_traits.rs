//! # PressureThresholds - Trait Implementations
//!
//! This module contains trait implementations for `PressureThresholds`.
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

use super::types::PressureThresholds;

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            low_threshold: 0.7,
            medium_threshold: 0.8,
            high_threshold: 0.9,
            critical_threshold: 0.95,
            swap_threshold: 0.1,
        }
    }
}
