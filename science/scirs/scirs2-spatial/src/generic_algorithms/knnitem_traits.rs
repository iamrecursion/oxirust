//! # KNNItem - Trait Implementations
//!
//! This module contains trait implementations for `KNNItem`.
//!
//! ## Implemented Traits
//!
//! - `PartialEq`
//! - `Eq`
//! - `PartialOrd`
//! - `Ord`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::generic_traits::{DistanceMetric, Point, SpatialPoint, SpatialScalar};
use scirs2_core::parallel_ops::*;
use std::cmp::Ordering;

use super::types::KNNItem;

impl<T: SpatialScalar> PartialEq for KNNItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: SpatialScalar> Eq for KNNItem<T> {}

impl<T: SpatialScalar> PartialOrd for KNNItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: SpatialScalar> Ord for KNNItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}
