//! # GCTriggerConditions - Trait Implementations
//!
//! This module contains trait implementations for `GCTriggerConditions`.
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
use std::time::{Duration, Instant, SystemTime};

use super::types::GCTriggerConditions;

impl Default for GCTriggerConditions {
    fn default() -> Self {
        Self {
            memory_threshold: 0.8,
            timebased: Some(Duration::from_secs(60)),
            allocation_threshold: 1000000,
            pressure_trigger: true,
            predictive_trigger: true,
        }
    }
}
