//! # QuantumConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{parallel_ops::*, simd_ops::SimdUnifiedOps, validation::*};
use std::collections::HashMap;

use super::functions::const_f64;
use super::types::{
    AnnealingSchedule, BlockEncodingConfig, ClassicalOptimizer, ContractionStrategy,
    DataEncodingMethod, DecoherenceConfig, EntanglementType, MeasurementStrategy, NoiseModel,
    ParameterConfig, ParameterInitialization, QuantumAmplitudeEstimationConfig,
    QuantumAnnealingConfig, QuantumClusteringAlgorithm, QuantumClusteringConfig, QuantumConfig,
    QuantumFeatureMap, QuantumKernelType, QuantumLayerConfig, QuantumLayerType,
    QuantumNeuralNetworkConfig, QuantumPCAConfig, QuantumSVMConfig, QuantumTrainingConfig,
    RotationAxis, TensorNetworkConfig, TensorNetworkType, VQEAnsatz, VQEConfig,
};

impl<F> Default for QuantumConfig<F>
where
    F: Float + NumCast + Copy + std::fmt::Display,
{
    fn default() -> Self {
        Self {
            num_qubits: 10,
            circuit_depth: 5,
            qae_config: QuantumAmplitudeEstimationConfig {
                evaluation_qubits: 3,
                target_accuracy: const_f64::<F>(0.01),
                max_iterations: 100,
                use_mlae: true,
                use_iqae: false,
            },
            qpca_config: QuantumPCAConfig {
                num_components: 5,
                matrix_exp_precision: const_f64::<F>(1e-6),
                use_variational: true,
                block_encoding: BlockEncodingConfig {
                    precision: const_f64::<F>(1e-8),
                    alpha: F::one(),
                    ancilla_qubits: 2,
                },
            },
            qsvm_config: QuantumSVMConfig {
                kernel_type: QuantumKernelType::FidelityKernel,
                feature_map: QuantumFeatureMap::ZZFeatureMap {
                    repetitions: 2,
                    entanglement: EntanglementType::Linear,
                },
                c_parameter: F::one(),
                use_vqc: true,
                ansatz_layers: 3,
            },
            qclustering_config: QuantumClusteringConfig {
                algorithm: QuantumClusteringAlgorithm::QuantumKMeans,
                num_clusters: 3,
                annealing_config: QuantumAnnealingConfig {
                    annealing_schedule: AnnealingSchedule::Linear {
                        duration: const_f64::<F>(100.0),
                    },
                    num_runs: 100,
                    temperature_range: (const_f64::<F>(0.01), const_f64::<F>(10.0)),
                    use_simulated_fallback: true,
                },
                use_qaoa: false,
            },
            vqe_config: VQEConfig {
                ansatz_type: VQEAnsatz::HardwareEfficient { layers: 3 },
                optimizer: ClassicalOptimizer::COBYLA,
                tolerance: const_f64::<F>(1e-6),
                max_iterations: 1000,
                measurement_shots: 1024,
            },
            tensor_network_config: TensorNetworkConfig {
                network_type: TensorNetworkType::MPS,
                max_bond_dim: 50,
                truncation_threshold: const_f64::<F>(1e-12),
                use_gpu: false,
                contraction_strategy: ContractionStrategy::Optimal,
            },
            qnn_config: QuantumNeuralNetworkConfig {
                quantum_layers: vec![QuantumLayerConfig {
                    layer_type: QuantumLayerType::RotationLayer {
                        axes: vec![RotationAxis::Y, RotationAxis::Z],
                    },
                    num_qubits: 4,
                    parameters: ParameterConfig {
                        num_parameters: 8,
                        initialization: ParameterInitialization::Random,
                        bounds: Some((-std::f64::consts::PI, std::f64::consts::PI)),
                    },
                }],
                data_encoding: DataEncodingMethod::AngleEncoding,
                measurement_strategy: MeasurementStrategy::PauliExpectation {
                    operators: vec!["Z".to_string()],
                },
                classical_layers: vec![],
                training_config: QuantumTrainingConfig {
                    learning_rate: const_f64::<F>(0.01),
                    epochs: 100,
                    batchsize: 32,
                    use_parameter_shift: true,
                    regularization: const_f64::<F>(0.001),
                },
            },
            noise_model: NoiseModel {
                gate_errors: HashMap::new(),
                decoherence_times: DecoherenceConfig {
                    t1: const_f64::<F>(100.0),
                    t2: const_f64::<F>(50.0),
                    t2_star: const_f64::<F>(30.0),
                },
                readout_errors: const_f64::<F>(0.01),
                enable_noise: false,
            },
        }
    }
}
