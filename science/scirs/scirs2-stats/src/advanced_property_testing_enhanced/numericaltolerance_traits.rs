//! # NumericalTolerance - Trait Implementations
//!
//! This module contains trait implementations for `NumericalTolerance`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NumericalTolerance;

impl Default for NumericalTolerance {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-12,
            relative_tolerance: 1e-10,
            ulp_tolerance: 4,
            adaptive_tolerance: true,
        }
    }
}

