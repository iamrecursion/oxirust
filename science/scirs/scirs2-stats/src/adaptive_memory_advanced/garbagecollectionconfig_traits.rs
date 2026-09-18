//! # GarbageCollectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `GarbageCollectionConfig`.
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

use super::types::{
    GCPerformanceTuning, GCStrategy, GCTriggerConditions, GCWorkloadAwareness,
    GarbageCollectionConfig,
};

impl Default for GarbageCollectionConfig {
    fn default() -> Self {
        Self {
            gc_strategy: GCStrategy::StatisticalAware,
            trigger_conditions: GCTriggerConditions::default(),
            performance_tuning: GCPerformanceTuning::default(),
            workload_awareness: GCWorkloadAwareness::default(),
        }
    }
}
