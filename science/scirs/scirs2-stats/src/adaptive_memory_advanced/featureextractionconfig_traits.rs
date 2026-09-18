//! # FeatureExtractionConfig - Trait Implementations
//!
//! This module contains trait implementations for `FeatureExtractionConfig`.
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

use super::types::FeatureExtractionConfig;

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            access_frequency: true,
            temporal_patterns: true,
            spatial_locality: true,
            data_characteristics: true,
            computation_type: true,
            system_resources: true,
        }
    }
}
