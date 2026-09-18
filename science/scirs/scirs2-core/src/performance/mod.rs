//! Performance optimization and monitoring for SciRS2 Core
//!
//! This module provides comprehensive performance optimization capabilities
//! including advanced SIMD operations, cache-aware algorithms, adaptive
//! optimization, and production-ready resource management.

pub mod advanced_optimization;
pub mod benchmarking;
pub mod cache_optimization;

/// Re-export key AI optimization types and functions
pub use advanced_optimization::{
    AIOptimizationEngine,
    AcceleratorType,
    AdvancedOptimizationConfig,
    CpuCharacteristics,
    ExecutionContext,
    OptimizationAnalytics,
    // Legacy compatibility types
    OptimizationSettings,
    PerformanceProfile,
    PerformanceTarget,
    SimdInstructionSet,
    SystemLoad,
    WorkloadType,
};

/// Re-export cache optimization functions
pub use cache_optimization::{
    adaptive_memcpy, adaptive_sort, cache_aware_reduce, cache_aware_transpose,
    matrix_multiply_cache_aware,
};

/// Re-export benchmarking framework types and functions
pub use benchmarking::{
    BenchmarkConfig, BenchmarkMeasurement, BenchmarkResults, BenchmarkRunner, BottleneckType,
    MemoryScaling, PerformanceBottleneck, ScalabilityAnalysis, StrategyPerformance,
};

/// Initialize the AI optimization engine
#[allow(dead_code)]
pub fn initialize_ai_optimization_engine() -> crate::error::CoreResult<AIOptimizationEngine> {
    Ok(AIOptimizationEngine::new())
}

/// Get AI-driven optimization analytics
#[allow(dead_code)]
pub fn get_optimization_analytics() -> OptimizationAnalytics {
    let engine = AIOptimizationEngine::new();
    engine.get_optimization_analytics()
}
