//! Advanced pipeline optimization with machine learning, quantum, and neuromorphic algorithms
//!
//! This module provides sophisticated optimization techniques for data processing pipelines,
//! including machine learning-based auto-tuning, quantum-inspired optimization, and
//! neuromorphic computing approaches.

pub mod config;
pub mod monitoring;
pub mod neuromorphic;
pub mod performance;
pub mod quantum;

// Re-export commonly used types
pub use config::{
    AutoTuningParameters, BatchProcessingMode, CacheConfiguration, CachePerformance,
    CacheReplacementPolicy, ConsciousnessConfig, ExecutionRecord, MemoryStrategy, MemoryUsage,
    NeuromorphicConfig, NumaNode, NumaTopology, OptimizedPipelineConfig,
    PipelinePerformanceMetrics, PrefetchStrategy, QuantumOptimizationConfig, RegressionDetector,
    SystemMetrics,
};

pub use monitoring::ResourceMonitor;

pub use performance::{
    AutoTuner, PerformanceHistory, PerformancePredictor, PerformanceTrend, PipelineProfile,
    TrendDirection,
};

pub use quantum::{QuantumAnnealer, QuantumOptimizer, QuantumState};

pub use neuromorphic::{
    AdaptationEvent, NeuromorphicMemory, NeuromorphicOptimizer, PlasticityRule, SpikePattern,
    SpikingNeuralNetwork, SpikingNeuron, SynapticConnection,
};
