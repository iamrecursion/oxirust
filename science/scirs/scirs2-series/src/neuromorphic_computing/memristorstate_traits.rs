//! # MemristorState - Trait Implementations
//!
//! This module contains trait implementations for `MemristorState`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::utils::const_f64;

use super::types::{MemristorParams, MemristorState};

impl<F: Float + scirs2_core::numeric::FromPrimitive> Default for MemristorState<F> {
    fn default() -> Self {
        let resistance = const_f64::<F>(1000.0);
        Self {
            resistance,
            conductance: const_f64::<F>(1.0) / resistance,
            state: const_f64::<F>(0.0),
            params: MemristorParams::default(),
        }
    }
}
