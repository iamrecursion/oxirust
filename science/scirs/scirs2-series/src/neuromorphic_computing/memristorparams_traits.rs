//! # MemristorParams - Trait Implementations
//!
//! This module contains trait implementations for `MemristorParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::utils::const_f64;

use super::types::MemristorParams;

impl<F: Float + scirs2_core::numeric::FromPrimitive> Default for MemristorParams<F> {
    fn default() -> Self {
        Self {
            r_min: const_f64::<F>(100.0),
            r_max: const_f64::<F>(10000.0),
            alpha: const_f64::<F>(0.1),
            beta: const_f64::<F>(1.0),
        }
    }
}
