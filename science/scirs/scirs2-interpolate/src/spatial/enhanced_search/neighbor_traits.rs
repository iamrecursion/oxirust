//! # Neighbor - Trait Implementations
//!
//! This module contains trait implementations for `Neighbor`.
//!
//! ## Implemented Traits
//!
//! - `PartialEq`
//! - `Eq`
//! - `PartialOrd`
//! - `Ord`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use std::cmp::Ordering;

use super::types::Neighbor;

impl<F: Float> PartialEq for Neighbor<F> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<F: Float> Eq for Neighbor<F> {}

impl<F: Float> PartialOrd for Neighbor<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Float> Ord for Neighbor<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}
