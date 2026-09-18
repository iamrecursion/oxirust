//! # RegressionThresholds - Trait Implementations
//!
//! This module contains trait implementations for `RegressionThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RegressionThresholds;

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            performance_threshold: 0.05,
            accuracy_threshold: 1e-10,
            stability_threshold: 0.01,
        }
    }
}

