//! Advanced analysis tools for dynamical systems
//!
//! This module provides tools for analyzing the behavior of dynamical systems,
//! including bifurcation analysis and stability assessment.

pub mod advanced;
pub mod bifurcation;
pub mod ml_prediction;
pub mod stability;
pub mod types;
pub mod utils;

// Re-export all types and analyzers for backward compatibility
pub use types::*;

// Bifurcation analysis
pub use bifurcation::BifurcationAnalyzer;

// Stability analysis
pub use stability::StabilityAnalyzer;

// Advanced analysis components
pub use advanced::{
    BifurcationPointData, BoxCountingMethod, ContinuationAnalyzer, ContinuationResult,
    FixedPointData, FractalAnalyzer, FractalDimension, LineType, LyapunovCalculator,
    MonodromyAnalyzer, MonodromyResult, PeriodicStabilityType, PoincareAnalyzer, PoincareMap,
    RQAMeasures, RecurrenceAnalysis, RecurrenceAnalyzer, RecurrentLine, ReturnMap,
};

// Note: advanced module has its own BifurcationType and DistanceMetric enums
// that may conflict with types.rs - users should use fully qualified names if needed
pub use advanced::BifurcationType as AdvancedBifurcationType;
pub use advanced::DistanceMetric as RecurrenceDistanceMetric;

// ML prediction components
pub use ml_prediction::{
    AccuracyMetrics, ActivationFunction, AdaptiveThresholdSystem, AlertAction, AlertMetrics,
    AlertSuppressionConfig, AlertSystemConfig, AnomalyDetectionConfig, AnomalyDetectionMethod,
    BaseClassifier, BatchNormParams, BayesianConfig, BifurcationEnsembleClassifier,
    BifurcationPrediction, BifurcationPredictionNetwork, ConformalConfig, ConformityScore,
    ConnectionType, ConservationLaw, Constraint, CrossValidationConfig, DataAugmentation,
    DiversityMethod, EarlyStoppingConfig, EnsembleAggregation, EnsembleConfig,
    EnsembleTrainingStrategy, EpochMetrics, EscalationLevel, FeatureExtraction,
    FeatureNormalization, FeatureSelectionConfig, FeatureSelectionMethod, FeedbackMechanism,
    FilterType, FrequencyFeatures, InterpolationMethod, KNNWeights, LatencyMetrics,
    LearningRateSchedule, LogFormat, LossFunction, MCDropoutConfig, MetaLearner, ModelParameters,
    MonitoringConfig, MonitoringEnsembleConfig, MultiStepStrategy, NetworkArchitecture,
    NotificationMethod, Optimizer, OutlierDetectionMethod, PerformanceMetrics, PerformanceTracker,
    PhaseSpaceFeatures, PreprocessingPipeline, PreprocessingStep, PriorParams, QualityCheck,
    RealTimeBifurcationMonitor, RegularizationConfig, ResourceMetrics, SVMKernel, ScoreFunction,
    SkipConnection, SmoothingMethod, StatisticalFeatures, StatisticalTest, TestMetrics,
    ThresholdAdaptationMethod, TimeSeriesBifurcationForecaster, TimeSeriesFeatures,
    TimeSeriesModel, TopologicalFeatures, TrainingConfiguration, TrendAnalysisConfig,
    TrendDetectionMethod, UncertaintyEstimate, UncertaintyQuantification, ValidationRule,
    VariationalMethod, VotingStrategy, WaveletType,
};

// Note: ml_prediction module has its own DistanceMetric enum that may conflict
pub use ml_prediction::DistanceMetric as MLDistanceMetric;

// Utility functions
pub use utils::{
    compute_determinant, compute_trace, estimate_matrix_rank, is_bifurcation_point,
    solve_linear_system,
};
