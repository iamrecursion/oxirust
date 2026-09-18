use super::*;

    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_advanced_think_numerical_stability_tester_creation() {
        let tester = create_advanced_think_numerical_stability_tester();
        assert!(tester.config.enable_edge_case_testing);
        assert!(tester.config.enable_precision_analysis);
        assert!(tester.config.enable_invariant_validation);
    }

    #[test]
    fn test_stability_tolerance_default() {
        let tolerance = StabilityTolerance::default();
        assert_eq!(tolerance.absolute_tolerance, 1e-14);
        assert_eq!(tolerance.relative_tolerance, 1e-12);
        assert_eq!(tolerance.condition_number_threshold, 1e12);
    }

    #[test]
    fn test_edge_case_generation() {
        let config = AdvancedNumericalStabilityConfig::default();
        let generator = EdgeCaseGenerator::new(&config);
        let testdata = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let edge_cases = generator
            .generate_comprehensive_edge_cases(&testdata)
            .expect("Test: operation failed");
        assert!(edge_cases.len() > 0);
    }

    #[test]
    fn test_stability_assessment_levels() {
        assert_eq!(StabilityAssessment::Excellent as u8, 0);
        assert!(matches!(
            StabilityAssessment::Good,
            StabilityAssessment::Good
        ));
        assert!(matches!(
            StabilityAssessment::Critical,
            StabilityAssessment::Critical
        ));
    }

    #[test]
    fn test_edge_case_types() {
        assert_eq!(EdgeCaseType::EmptyArray as u8, 0);
        assert!(matches!(EdgeCaseType::AllZeros, EdgeCaseType::AllZeros));
        assert!(matches!(
            EdgeCaseType::ContainsNaN,
            EdgeCaseType::ContainsNaN
        ));
    }

    #[test]
    fn test_comprehensive_stability_result_creation() {
        let result = ComprehensiveStabilityResult::new("test_function".to_string());
        assert_eq!(result.function_name, "test_function");
        assert_eq!(result.overall_stability_score, 0.0);
        assert!(matches!(
            result.stability_assessment,
            StabilityAssessment::Unknown
        ));
    }

    #[test]
    fn test_stability_metrics_default() {
        let metrics = StabilityMetrics::default();
        assert_eq!(metrics.condition_number, 1.0);
        assert_eq!(metrics.relative_error, 0.0);
        assert_eq!(metrics.nan_count, 0);
    }

    #[test]
    fn test_monte_carlo_stability_result_creation() {
        let result = MonteCarloStabilityResult::new();
        assert_eq!(result.sample_count, 0);
        assert_eq!(result.mean_value, 0.0);
        assert!(matches!(
            result.stability_assessment,
            MonteCarloStabilityAssessment::Unknown
        ));
    }

    #[test]
    fn test_fast_stability_tester_config() {
        let tester = create_fast_numerical_stability_tester();
        assert!(tester.config.enable_edge_case_testing);
        assert!(!tester.config.enable_monte_carlo_testing);
        assert_eq!(
            tester.config.thoroughness_level,
            NumericalStabilityThoroughness::Basic
        );
    }

    #[test]
    fn test_exhaustive_stability_tester_config() {
        let tester = create_exhaustive_numerical_stability_tester();
        assert!(tester.config.enable_edge_case_testing);
        assert!(tester.config.enable_monte_carlo_testing);
        assert_eq!(
            tester.config.thoroughness_level,
            NumericalStabilityThoroughness::Exhaustive
        );
        assert_eq!(tester.config.monte_carlo_samples, 1000000);
    }
