//! # StabilityMetrics - Trait Implementations
//!
//! This module contains trait implementations for `StabilityMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StabilityMetrics;

impl Default for StabilityMetrics {
    fn default() -> Self {
        Self {
            condition_number: 1.0,
            relative_error: 0.0,
            nan_count: 0,
            infinite_count: 0,
            normal_count: 0,
            subnormal_count: 0,
        }
    }
}

