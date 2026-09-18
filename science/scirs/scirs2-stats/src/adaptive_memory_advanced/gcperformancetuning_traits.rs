//! # GCPerformanceTuning - Trait Implementations
//!
//! This module contains trait implementations for `GCPerformanceTuning`.
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
use std::time::{Duration, Instant, SystemTime};

use super::types::GCPerformanceTuning;

impl Default for GCPerformanceTuning {
    fn default() -> Self {
        Self {
            parallel_threads: num_threads().max(2),
            pause_time_target: Duration::from_millis(10),
            incremental_chunksize: 1024,
            concurrent_enabled: true,
            background_enabled: true,
        }
    }
}
