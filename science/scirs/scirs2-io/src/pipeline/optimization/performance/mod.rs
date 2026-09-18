//! Performance optimization and monitoring for pipeline operations
//!
//! This module provides comprehensive performance optimization capabilities including:
//! - Auto-tuning system with machine learning-based parameter optimization
//! - Predictive modeling for performance forecasting
//! - Real-time monitoring and regression detection
//! - Historical performance analysis and trend tracking

pub mod auto_tuning;
pub mod predictive_modeling;
pub mod monitoring;

// Re-export key types for easy access
pub use auto_tuning::{
    AutoTuner, OptimalParameters, OptimizedPipelineConfig, PrefetchStrategy, 
    BatchProcessingMode, CacheStrategy,
};

pub use predictive_modeling::{
    ParameterOptimizationModel, PerformancePredictionEngine, PerformancePrediction,
    TrainingExample, ModelStatistics, ModelCheckpoint,
};

pub use monitoring::{
    PerformanceHistory, ExecutionRecord, PipelineProfile, RegressionDetector,
    PipelinePerformanceMetrics, StagePerformance, RealTimeMonitor, 
    PerformanceTrends, TrendDirection, RegressionAlert, AlertSeverity,
};