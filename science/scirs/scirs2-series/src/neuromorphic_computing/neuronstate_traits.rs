//! # NeuronState - Trait Implementations
//!
//! This module contains trait implementations for `NeuronState`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::utils::const_f64;

use super::types::NeuronState;

impl<F: Float + FromPrimitive> Default for NeuronState<F> {
    fn default() -> Self {
        Self {
            v: const_f64::<F>(-70.0),
            u: F::zero(),
            last_spike: None,
            refractory: 0.0,
            input_current: F::zero(),
        }
    }
}
