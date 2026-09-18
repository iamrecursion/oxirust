//! # StabilityTolerance - Trait Implementations
//!
//! This module contains trait implementations for `StabilityTolerance`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StabilityTolerance;

impl Default for StabilityTolerance {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-14,
            relative_tolerance: 1e-12,
            condition_number_threshold: 1e12,
            cancellation_threshold: 1e-10,
            convergence_tolerance: 1e-10,
            monte_carlo_confidence_level: 0.95,
        }
    }
}

