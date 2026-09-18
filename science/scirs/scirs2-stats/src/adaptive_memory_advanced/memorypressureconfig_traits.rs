//! # MemoryPressureConfig - Trait Implementations
//!
//! This module contains trait implementations for `MemoryPressureConfig`.
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

use super::types::{
    EmergencyResponseConfig, MemoryPressureConfig, PressureThresholds, ResponseStrategies,
};

impl Default for MemoryPressureConfig {
    fn default() -> Self {
        Self {
            pressure_thresholds: PressureThresholds::default(),
            response_strategies: ResponseStrategies::default(),
            monitoring_frequency: Duration::from_millis(500),
            emergency_config: EmergencyResponseConfig::default(),
        }
    }
}
