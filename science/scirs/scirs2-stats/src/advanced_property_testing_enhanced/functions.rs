//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult};
use std::time::{Duration, Instant};

use super::types::{AdvancedPropertyConfig, AdvancedPropertyTester, EdgeCaseGenerationStrategy, NumericalTolerance, PropertyGenerationStrategy, TestData, TestingThoroughnessLevel};

pub trait TestDataGenerator<F> {
    fn generate(&self) -> StatsResult<TestData<F>>;
}
pub trait Implementation<F> {
    fn execute(&self, input: &TestData<F>) -> StatsResult<F>;
    fn name(&self) -> &str;
    fn version(&self) -> &str;
}
/// Create default advanced property tester
#[allow(dead_code)]
pub fn create_advanced_think_property_tester() -> AdvancedPropertyTester {
    AdvancedPropertyTester::new(AdvancedPropertyConfig::default())
}
/// Create configured advanced property tester
#[allow(dead_code)]
pub fn create_configured_advanced_think_property_tester(
    config: AdvancedPropertyConfig,
) -> AdvancedPropertyTester {
    AdvancedPropertyTester::new(config)
}
/// Create comprehensive property tester for production use
#[allow(dead_code)]
pub fn create_comprehensive_property_tester() -> AdvancedPropertyTester {
    let config = AdvancedPropertyConfig {
        enable_mathematical_invariants: true,
        enable_statistical_properties: true,
        enable_numerical_stability: true,
        enable_cross_implementation: true,
        enable_edge_case_generation: true,
        enable_performance_properties: true,
        enable_fuzzing: true,
        enable_regression_detection: true,
        thoroughness_level: TestingThoroughnessLevel::Comprehensive,
        property_generation_strategy: PropertyGenerationStrategy::Intelligent,
        edge_case_strategy: EdgeCaseGenerationStrategy::AIGuided,
        numerical_tolerance: NumericalTolerance::default(),
        test_timeout: Duration::from_secs(600),
        max_iterations: 50000,
    };
    AdvancedPropertyTester::new(config)
}
/// Create fast property tester for development
#[allow(dead_code)]
pub fn create_fast_property_tester() -> AdvancedPropertyTester {
    let config = AdvancedPropertyConfig {
        enable_mathematical_invariants: true,
        enable_statistical_properties: false,
        enable_numerical_stability: true,
        enable_cross_implementation: false,
        enable_edge_case_generation: true,
        enable_performance_properties: false,
        enable_fuzzing: false,
        enable_regression_detection: false,
        thoroughness_level: TestingThoroughnessLevel::Standard,
        property_generation_strategy: PropertyGenerationStrategy::Predefined,
        edge_case_strategy: EdgeCaseGenerationStrategy::Manual,
        numerical_tolerance: NumericalTolerance {
            absolute_tolerance: 1e-8,
            relative_tolerance: 1e-6,
            ulp_tolerance: 8,
            adaptive_tolerance: false,
        },
        test_timeout: Duration::from_secs(60),
        max_iterations: 1000,
    };
    AdvancedPropertyTester::new(config)
}
