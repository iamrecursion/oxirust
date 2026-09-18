use super::*;

    use super::*;

    #[test]
    fn test_advanced_think_property_tester_creation() {
        let tester = create_advanced_think_property_tester();
        assert!(tester.config.enable_mathematical_invariants);
        assert!(tester.config.enable_statistical_properties);
    }

    #[test]
    fn test_numerical_tolerance_default() {
        let tolerance = NumericalTolerance::default();
        assert_eq!(tolerance.absolute_tolerance, 1e-12);
        assert_eq!(tolerance.relative_tolerance, 1e-10);
        assert_eq!(tolerance.ulp_tolerance, 4);
        assert!(tolerance.adaptive_tolerance);
    }

    #[test]
    fn test_mathematical_property_registry() {
        let config = AdvancedPropertyConfig::default();
        let registry = MathematicalPropertyRegistry::new(&config);

        let mean_properties = registry.get_properties_for_function("mean").expect("Test: operation failed");
        assert!(!mean_properties.is_empty());
    }

    #[test]
    fn test_statistical_property_registry() {
        let config = AdvancedPropertyConfig::default();
        let registry = StatisticalPropertyRegistry::new(&config);

        let mean_properties = registry
            .get_properties_for_operation(&StatisticalOperationType::Mean)
            .expect("Test: operation failed");
        assert!(!mean_properties.is_empty());
    }

    #[test]
    fn test_fuzzing_config_creation() {
        let config = FuzzingConfig {
            fuzzing_strategy: FuzzingStrategy::Guided,
            input_mutation_rate: 0.1,
            max_iterations: 10000,
            crash_detection: true,
            anomaly_detection: true,
            coverage_tracking: true,
        };

        assert_eq!(config.fuzzing_strategy, FuzzingStrategy::Guided);
        assert_eq!(config.input_mutation_rate, 0.1);
        assert!(config.crash_detection);
    }

    #[test]
    fn test_edge_case_criticality_ordering() {
        assert!(EdgeCaseCriticality::Critical as u8 >, EdgeCaseCriticality::High as u8);
        assert!(EdgeCaseCriticality::High as u8 >, EdgeCaseCriticality::Medium as u8);
        assert!(EdgeCaseCriticality::Medium as u8 >, EdgeCaseCriticality::Low as u8);
    }

    #[test]
    fn test_complexity_class_hierarchy() {
        // Test that complexity classes are properly ordered
        let linear = ComplexityClass::Linear;
        let quadratic = ComplexityClass::Quadratic;

        assert_ne!(linear, quadratic);
        assert_eq!(linear, ComplexityClass::Linear);
    }

    #[test]
    fn test_regression_thresholds_default() {
        let thresholds = RegressionThresholds::default();
        assert_eq!(thresholds.performance_threshold, 0.05);
        assert_eq!(thresholds.accuracy_threshold, 1e-10);
        assert_eq!(thresholds.stability_threshold, 0.01);
    }

    #[test]
    fn test_specialized_property_tester_creation() {
        let comprehensive_tester = create_comprehensive_property_tester();
        assert_eq!(
            comprehensive_tester.config.thoroughness_level,
            TestingThoroughnessLevel::Comprehensive
        );
        assert!(comprehensive_tester.config.enable_performance_properties);

        let fast_tester = create_fast_property_tester();
        assert_eq!(
            fast_tester.config.thoroughness_level,
            TestingThoroughnessLevel::Standard
        );
        assert!(!fast_tester.config.enable_performance_properties);
    }

    #[test]
    fn test_coverage_tracker() {
        let tracker = CoverageTracker::new();
        assert_eq!(tracker.code_coverage, 0.0);
        assert_eq!(tracker.branch_coverage, 0.0);
        assert_eq!(tracker.path_coverage, 0.0);
    }

    #[test]
    fn test_vulnerability_severity_ordering() {
        assert!(VulnerabilitySeverity::Critical as u8 >, VulnerabilitySeverity::High as u8);
        assert!(VulnerabilitySeverity::High as u8 >, VulnerabilitySeverity::Medium as u8);
        assert!(VulnerabilitySeverity::Medium as u8 >, VulnerabilitySeverity::Low as u8);
    }
