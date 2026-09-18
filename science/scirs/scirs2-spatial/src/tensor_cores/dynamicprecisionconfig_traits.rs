//! # DynamicPrecisionConfig - Trait Implementations
//!
//! This module contains trait implementations for `DynamicPrecisionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant};

use super::types::{DynamicPrecisionConfig, PrecisionMode, ScalingStrategy};

impl Default for DynamicPrecisionConfig {
    fn default() -> Self {
        Self {
            strategy: ScalingStrategy::Balanced,
            min_precision: PrecisionMode::Int8Dynamic,
            max_precision: PrecisionMode::Full32,
            stability_threshold_up: 1e-6,
            stability_threshold_down: 1e-9,
            performance_weight: 0.6,
            accuracy_weight: 0.4,
            max_changes_per_operation: 3,
            change_cooldown: Duration::from_millis(100),
        }
    }
}
