//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crate::testing::{TestConfig, TestResult, TestSuite};
use std::time::{Duration, Instant};

use super::types::{EcosystemTestConfig, EcosystemTestRunner};

/// Create a comprehensive ecosystem test suite
#[allow(dead_code)]
pub fn create_ecosystem_test_suite(config: EcosystemTestConfig) -> CoreResult<TestSuite> {
    let base_config = config.base.clone();
    let mut suite = TestSuite::new("SciRS2 Ecosystem Integration - 1.0 Release", base_config);
    suite.add_test("ecosystem_integration_1_0", move |_runner| {
        let ecosystem_runner = EcosystemTestRunner::new(config.clone());
        let result = ecosystem_runner.run_ecosystem_tests()?;
        if result.base.passed {
            Ok(TestResult::success(
                std::time::Duration::from_secs(1),
                result.discovered_modules.len(),
            )
            .with_metadata(
                "health_score".to_string(),
                format!("{:.1}", result.health_score),
            )
            .with_metadata(
                "ready_for_release".to_string(),
                result.release_readiness.ready_for_release.to_string(),
            ))
        } else {
            Ok(TestResult::failure(
                std::time::Duration::from_secs(1),
                result.discovered_modules.len(),
                result
                    .base
                    .error
                    .unwrap_or_else(|| "Ecosystem integration failed".to_string()),
            ))
        }
    });
    Ok(suite)
}
