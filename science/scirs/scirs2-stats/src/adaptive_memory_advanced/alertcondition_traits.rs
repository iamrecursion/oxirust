//! # AlertCondition - Trait Implementations
//!
//! This module contains trait implementations for `AlertCondition`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::AlertCondition;

impl std::fmt::Debug for AlertCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertCondition::MemoryUsageThreshold(v) => {
                f.debug_tuple("MemoryUsageThreshold").field(v).finish()
            }
            AlertCondition::CacheHitRatioThreshold(v) => {
                f.debug_tuple("CacheHitRatioThreshold").field(v).finish()
            }
            AlertCondition::GCOverheadThreshold(v) => {
                f.debug_tuple("GCOverheadThreshold").field(v).finish()
            }
            AlertCondition::FragmentationThreshold(v) => {
                f.debug_tuple("FragmentationThreshold").field(v).finish()
            }
            AlertCondition::PressureLevelThreshold(v) => {
                f.debug_tuple("PressureLevelThreshold").field(v).finish()
            }
            AlertCondition::PerformanceDegradation(v) => {
                f.debug_tuple("PerformanceDegradation").field(v).finish()
            }
            AlertCondition::Custom(_) => f.debug_tuple("Custom").field(&"<function>").finish(),
        }
    }
}
