//! Sequence random operations for SCIRS2 ecosystem
//!
//! This module provides sequence-based random operations that are fully compatible
//! with the rand::seq module while maintaining SCIRS2 POLICY compliance.

// Re-export the core SliceRandom trait for direct compatibility
pub use rand::seq::SliceRandom;

// Re-export other sequence operations from rand::seq
pub use rand::seq::index;

// Re-export our enhanced slice operations
pub use crate::random::slice_ops::{ScientificSliceRandom, SliceRandomExt};
