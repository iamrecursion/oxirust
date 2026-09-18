use super::*;

    use super::*;

    #[test]
    fn test_advanced_coordinator_creation() {
        let coordinator = create_advanced_fft_coordinator::<f64>();
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_advanced_config_default() {
        let config = advancedFftConfig::default();
        assert!(config.enable_method_selection);
        assert!(config.enable_adaptive_optimization);
        assert!(config.enable_quantum_optimization);
    }

    #[test]
    fn test_algorithm_types() {
        let algorithms = [
            FftAlgorithmType::CooleyTukeyRadix2,
            FftAlgorithmType::CooleyTukeyMixedRadix,
            FftAlgorithmType::BluesteinAlgorithm,
            FftAlgorithmType::GpuAcceleratedFft,
        ];
        assert_eq!(algorithms.len(), 4);
    }

    #[test]
    fn test_hardware_capabilities_detection() {
        let capabilities = HardwareCapabilities::detect();
        assert!(capabilities.is_ok());
    }
