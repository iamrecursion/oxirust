//! # GCWorkloadAwareness - Trait Implementations
//!
//! This module contains trait implementations for `GCWorkloadAwareness`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::GCWorkloadAwareness;

impl Default for GCWorkloadAwareness {
    fn default() -> Self {
        Self {
            operation_type_aware: true,
            lifecycle_analysis: true,
            phase_awareness: true,
            pattern_integration: true,
        }
    }
}
