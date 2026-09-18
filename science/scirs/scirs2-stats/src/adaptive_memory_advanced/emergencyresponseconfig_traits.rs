//! # EmergencyResponseConfig - Trait Implementations
//!
//! This module contains trait implementations for `EmergencyResponseConfig`.
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

use super::types::{EmergencyRecoveryStrategy, EmergencyResponseConfig};

impl Default for EmergencyResponseConfig {
    fn default() -> Self {
        Self {
            enable_emergency: true,
            evacuation_threshold: 0.98,
            compression_ratio: 0.5,
            enable_spillover: true,
            recovery_strategy: EmergencyRecoveryStrategy::Adaptive,
        }
    }
}
