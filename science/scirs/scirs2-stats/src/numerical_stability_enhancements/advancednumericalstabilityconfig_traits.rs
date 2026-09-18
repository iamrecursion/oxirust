//! # AdvancedNumericalStabilityConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdvancedNumericalStabilityConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{AdvancedNumericalStabilityConfig, EdgeCaseGenerationApproach, NumericalStabilityThoroughness, PrecisionTestingStrategy, StabilityTolerance};

impl Default for AdvancedNumericalStabilityConfig {
    fn default() -> Self {
        Self {
            enable_edge_case_testing: true,
            enable_precision_analysis: true,
            enable_invariant_validation: true,
            enable_cancellation_detection: true,
            enable_overflow_monitoring: true,
            enable_condition_analysis: true,
            enable_differentiation_testing: true,
            enable_convergence_testing: true,
            enable_monte_carlo_testing: true,
            enable_regression_testing: true,
            thoroughness_level: NumericalStabilityThoroughness::Comprehensive,
            precision_strategy: PrecisionTestingStrategy::MultiPrecision,
            edge_case_approach: EdgeCaseGenerationApproach::Systematic,
            stability_tolerance: StabilityTolerance::default(),
            test_timeout: Duration::from_secs(600),
            max_convergence_iterations: 10000,
            monte_carlo_samples: 100000,
        }
    }
}

