//! # ResponseStrategies - Trait Implementations
//!
//! This module contains trait implementations for `ResponseStrategies`.
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

use super::types::{PressureResponse, ResponseStrategies};

impl Default for ResponseStrategies {
    fn default() -> Self {
        Self {
            low_pressure: vec![PressureResponse::ReduceCache],
            medium_pressure: vec![PressureResponse::TriggerGC, PressureResponse::CompressData],
            high_pressure: vec![
                PressureResponse::TriggerGC,
                PressureResponse::CompressData,
                PressureResponse::MoveToDisk,
            ],
            critical_pressure: vec![
                PressureResponse::EmergencyEvacuation,
                PressureResponse::PauseOperations,
            ],
        }
    }
}
