use super::*;

use super::*;

#[test]
fn test_ecosystem_config_creation() {
    let config = EcosystemTestConfig::default();
    assert!(config.auto_discover_modules);
    assert_eq!(config.min_modules_required, 20);
    assert_eq!(config.max_performance_degradation, 5.0);
}

#[test]
fn test_module_type_classification() {
    let runner = EcosystemTestRunner::new(EcosystemTestConfig::default());

    assert_eq!(runner.classify_module_type("scirs2-core"), ModuleType::Core);
    assert_eq!(
        runner.classify_module_type("scirs2-linalg"),
        ModuleType::Computational
    );
    assert_eq!(
        runner.classify_module_type("scirs2-neural"),
        ModuleType::MachineLearning
    );
    assert_eq!(runner.classify_module_type("scirs2-io"), ModuleType::DataIO);
    assert_eq!(
        runner.classify_module_type("scirs2"),
        ModuleType::Integration
    );
}

#[test]
fn test_semver_compliance_check() {
    let runner = EcosystemTestRunner::new(EcosystemTestConfig::default());

    assert!(runner.is_semver_compliant("1.0.0"));
    assert!(runner.is_semver_compliant("0.1.0"));
    assert!(!runner.is_semver_compliant("1.0"));
    assert!(!runner.is_semver_compliant("invalid"));
}
