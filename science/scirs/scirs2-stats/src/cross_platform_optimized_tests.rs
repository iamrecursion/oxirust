use super::*;

    use super::*;

    #[test]
    fn test_advanced_think_cross_platform_tester_creation() {
        let tester = create_advanced_think_cross_platform_tester();
        assert!(tester.config.enable_platform_detection);
        assert!(tester.config.enable_consistency_testing);
    }

    #[test]
    fn test_consistency_tolerance_default() {
        let tolerance = ConsistencyTolerance::default();
        assert_eq!(tolerance.numerical_absolute_tolerance, 1e-12);
        assert_eq!(tolerance.numerical_relative_tolerance, 1e-10);
        assert_eq!(tolerance.performance_variance_threshold, 0.20);
    }

    #[test]
    fn test_performance_variance_tolerance_default() {
        let tolerance = PerformanceVarianceTolerance::default();
        assert_eq!(tolerance.max_performance_degradation, 0.30);
        assert_eq!(tolerance.max_memory_variance, 0.25);
        assert_eq!(tolerance.max_throughput_variance, 0.20);
    }

    #[test]
    fn test_platform_type_variants() {
        assert_eq!(PlatformType::X86_64Linux as u8, 0);
        assert!(matches!(PlatformType::X86_64Linux, PlatformType::X86_64Linux));
        assert!(matches!(PlatformType::AArch64MacOS, PlatformType::AArch64MacOS));
    }

    #[test]
    fn test_optimization_type_ordering() {
        let simd = OptimizationType::SIMD;
        let parallel = OptimizationType::Parallel;
        let numa = OptimizationType::NUMA;
        
        assert_ne!(simd, parallel);
        assert_ne!(parallel, numa);
        assert_eq!(simd, OptimizationType::SIMD);
    }

    #[test]
    fn test_platform_detector() {
        let config = AdvancedCrossPlatformConfig::default();
        let detector = PlatformDetector::new(&config);
        
        let platform = detector.detect_current_platform().expect("Operation failed");
        assert!(!platform.hardware_profile.cpu_architecture.is_empty());
        assert!(platform.hardware_profile.cpu_cores > 0);
    }

    #[test]
    fn test_hardware_profile_creation() {
        let hardware_profile = HardwareProfile {
            cpu_architecture: "x86_64".to_string(),
            cpu_vendor: "Intel".to_string(),
            cpu_model: "Core i7".to_string(),
            cpu_cores: 8,
            cpu_threads: 16,
            cpu_features: vec!["AVX2".to_string()],
            memorysize_gb: 32.0,
            memory_speed_mhz: 3200.0,
            cache_hierarchy: CacheHierarchy {
                l1data_kb: 32,
                l1_instruction_kb: 32,
                l2_kb: 256,
                l3_kb: 8192,
                cache_linesize: 64,
            },
            numa_topology: None,
            accelerators: vec![],
        };
        
        assert_eq!(hardware_profile.cpu_cores, 8);
        assert_eq!(hardware_profile.cpu_threads, 16);
        assert_eq!(hardware_profile.memorysize_gb, 32.0);
    }

    #[test]
    fn test_edge_case_type_variants() {
        assert!(matches!(EdgeCaseType::NumericalLimits, EdgeCaseType::NumericalLimits));
        assert!(matches!(EdgeCaseType::MemoryLimits, EdgeCaseType::MemoryLimits));
        assert!(matches!(EdgeCaseType::PlatformSpecific, EdgeCaseType::PlatformSpecific));
    }

    #[test]
    fn test_regression_severity_ordering() {
        assert!(RegressionSeverity::Critical as u8 >, RegressionSeverity::Major as u8);
        assert!(RegressionSeverity::Major as u8 >, RegressionSeverity::Moderate as u8);
        assert!(RegressionSeverity::Moderate as u8 >, RegressionSeverity::Minor as u8);
    }

    #[test]
    fn test_specialized_cross_platform_tester_creation() {
        let comprehensive_tester = create_comprehensive_cross_platform_tester();
        assert_eq!(
            comprehensive_tester.config.thoroughness_level,
            PlatformTestingThoroughness::Comprehensive
        );
        assert!(comprehensive_tester.config.enable_continuous_monitoring);
        
        let fast_tester = create_fast_cross_platform_tester();
        assert_eq!(
            fast_tester.config.thoroughness_level,
            PlatformTestingThoroughness::Basic
        );
        assert!(!fast_tester.config.enable_performance_analysis);
    }

    #[test]
    fn test_monitoring_session_creation() {
        let config = ContinuousMonitoringConfig {
            monitoring_interval: Duration::from_secs(60),
            platforms_to_monitor: vec!["linux".to_string()],
            metrics_to_track: vec!["performance".to_string()],
            alert_thresholds: HashMap::new(),
            auto_remediation: false,
        };
        
        let mut monitoring_system = ContinuousMonitoringSystem::new(&AdvancedCrossPlatformConfig::default());
        let session = monitoring_system.start_monitoring_session(config).expect("Operation failed");
        
        assert!(!session.session_id.is_empty());
        assert_eq!(session.current_status, MonitoringStatus::Starting);
    }
