//! # OrderedFloat - Trait Implementations
//!
//! This module contains trait implementations for `OrderedFloat`.
//!
//! ## Implemented Traits
//!
//! - `Eq`
//! - `PartialOrd`
//! - `Ord`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use std::cmp::Ordering;

use super::types::OrderedFloat;

impl<F: Float> Eq for OrderedFloat<F> {}

impl<F: Float> PartialOrd for OrderedFloat<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Float> Ord for OrderedFloat<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}
