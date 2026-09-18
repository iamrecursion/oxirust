//! # EcosystemTestConfig - Trait Implementations
//!
//! This module contains trait implementations for `EcosystemTestConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::testing::{TestConfig, TestResult, TestSuite};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::types::{ApiComplianceLevel, DeploymentTarget, EcosystemTestConfig};

impl Default for EcosystemTestConfig {
    fn default() -> Self {
        let workspace_path = std::env::var("SCIRS2_WORKSPACE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        Self {
            base: TestConfig::default().with_timeout(Duration::from_secs(300)),
            workspace_path,
            auto_discover_modules: true,
            included_modules: HashSet::new(),
            excluded_modules: HashSet::new(),
            test_performance: true,
            test_api_stability: true,
            test_production_readiness: true,
            test_long_term_stability: true,
            max_performance_degradation: 5.0,
            min_modules_required: 20,
            api_compliance_level: ApiComplianceLevel::Stable,
            deployment_targets: vec![
                DeploymentTarget::Linux,
                DeploymentTarget::MacOS,
                DeploymentTarget::Windows,
            ],
        }
    }
}
