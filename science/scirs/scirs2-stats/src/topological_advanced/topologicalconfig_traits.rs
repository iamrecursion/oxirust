//! # TopologicalConfig - Trait Implementations
//!
//! This module contains trait implementations for `TopologicalConfig`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Clone`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
use scirs2_linalg::parallel_dispatch::ParallelConfig;

use super::functions::const_f64;
use super::types::{
    ClusteringMethod, CoeffientField, CoverConfig, CoverType, DistanceMetric, FiltrationConfig,
    FiltrationType, MapperConfig, MergerStrategy, MultipleComparisonsCorrection, MultiscaleConfig,
    NullModel, PersistenceAlgorithm, PersistenceConfig, ScaleDistribution, SimplificationConfig,
    TopologicalConfig, TopologicalInferenceConfig, TopologicalTest,
};

impl<F> std::fmt::Debug for TopologicalConfig<F>
where
    F: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopologicalConfig")
            .field("max_dimension", &self.max_dimension)
            .field("filtration_config", &self.filtration_config)
            .field("persistence_config", &self.persistence_config)
            .field("mapper_config", &self.mapper_config)
            .field("multiscale_config", &self.multiscale_config)
            .field("inference_config", &self.inference_config)
            .field("parallel_config", &"<ParallelConfig>")
            .finish()
    }
}

impl<F> Clone for TopologicalConfig<F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            max_dimension: self.max_dimension,
            filtration_config: self.filtration_config.clone(),
            persistence_config: self.persistence_config.clone(),
            mapper_config: self.mapper_config.clone(),
            multiscale_config: self.multiscale_config.clone(),
            inference_config: self.inference_config.clone(),
            parallel_config: ParallelConfig::default(),
        }
    }
}

impl<F> Default for TopologicalConfig<F>
where
    F: Float + NumCast + Copy + std::fmt::Display + SimdUnifiedOps + Send + Sync,
{
    fn default() -> Self {
        Self {
            max_dimension: 2,
            filtration_config: FiltrationConfig {
                filtration_type: FiltrationType::VietorisRips,
                distance_metric: DistanceMetric::Euclidean,
                max_epsilon: const_f64::<F>(1.0),
                num_steps: 100,
                adaptive_steps: false,
            },
            persistence_config: PersistenceConfig {
                algorithm: PersistenceAlgorithm::StandardReduction,
                coefficient_field: CoeffientField::Z2,
                persistence_threshold: const_f64::<F>(0.01),
                compute_entropy: true,
                stability_analysis: false,
            },
            mapper_config: MapperConfig {
                filter_functions: Vec::new(),
                cover_config: CoverConfig {
                    num_intervals: vec![10],
                    overlap_percent: const_f64::<F>(0.3),
                    cover_type: CoverType::UniformInterval,
                },
                clustering_method: ClusteringMethod::SingleLinkage,
                overlap_threshold: const_f64::<F>(0.1),
                simplification: SimplificationConfig {
                    edge_contraction: false,
                    vertex_removal: false,
                    threshold: 0.01,
                },
            },
            multiscale_config: MultiscaleConfig {
                scale_range: (const_f64::<F>(0.1), const_f64::<F>(2.0)),
                num_scales: 10,
                scale_distribution: ScaleDistribution::Linear,
                merger_strategy: MergerStrategy::Union,
            },
            inference_config: TopologicalInferenceConfig {
                bootstrap_samples: 0,
                confidence_level: const_f64::<F>(0.95),
                null_model: NullModel::UniformRandom,
                test_type: TopologicalTest::PersistentRankTest,
                multiple_comparisons: MultipleComparisonsCorrection::BenjaminiHochberg,
            },
            parallel_config: ParallelConfig::default(),
        }
    }
}
