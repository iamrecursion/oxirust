//! # AdvancedPropertyConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdvancedPropertyConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant};

use super::types::{AdvancedPropertyConfig, EdgeCaseGenerationStrategy, NumericalTolerance, PropertyGenerationStrategy, TestingThoroughnessLevel};

impl Default for AdvancedPropertyConfig {
    fn default() -> Self {
        Self {
            enable_mathematical_invariants: true,
            enable_statistical_properties: true,
            enable_numerical_stability: true,
            enable_cross_implementation: true,
            enable_edge_case_generation: true,
            enable_performance_properties: false,
            enable_fuzzing: true,
            enable_regression_detection: true,
            thoroughness_level: TestingThoroughnessLevel::Comprehensive,
            property_generation_strategy: PropertyGenerationStrategy::Intelligent,
            edge_case_strategy: EdgeCaseGenerationStrategy::Adaptive,
            numerical_tolerance: NumericalTolerance::default(),
            test_timeout: Duration::from_secs(300),
            max_iterations: 10000,
        }
    }
}

