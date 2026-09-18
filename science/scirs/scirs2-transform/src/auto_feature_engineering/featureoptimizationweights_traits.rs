//! # FeatureOptimizationWeights - Trait Implementations
//!
//! This module contains trait implementations for `FeatureOptimizationWeights`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "auto-feature-engineering")]
use super::types::FeatureOptimizationWeights;

#[cfg(feature = "auto-feature-engineering")]
impl Default for FeatureOptimizationWeights {
    fn default() -> Self {
        FeatureOptimizationWeights {
            performance_weight: 0.5,
            efficiency_weight: 0.3,
            interpretability_weight: 0.1,
            robustness_weight: 0.1,
        }
    }
}
