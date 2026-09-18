//! Advanced pipeline optimization with comprehensive resource management
//!
//! This module provides a complete optimization framework for I/O pipelines including:
//! - Resource allocation and system monitoring
//! - Memory pool management and allocation strategies  
//! - Performance monitoring, auto-tuning, and predictive modeling
//! - Cache optimization and out-of-core processing

pub mod resource;
pub mod memory;
pub mod performance;

// Re-export key optimization components
pub use resource::allocation::{
    ResourceMonitor, SystemMetrics, MemoryUsage, CachePerformance, NumaTopology, NumaNode,
};

pub use memory::pool_management::{
    MemoryPoolManager, MemoryStrategy, MemoryPool, MemoryBlock, 
    AllocationStrategy, FragmentationMonitor, MemoryPoolStats,
};

pub use performance::{
    AutoTuner, OptimalParameters, OptimizedPipelineConfig, PrefetchStrategy,
    BatchProcessingMode, CacheStrategy, PerformanceHistory, ExecutionRecord,
    PipelineProfile, RegressionDetector, PipelinePerformanceMetrics, 
    StagePerformance, RealTimeMonitor,
};

use crate::error::Result;

/// Advanced pipeline optimizer that coordinates all optimization subsystems
#[derive(Debug)]
pub struct AdvancedPipelineOptimizer {
    resource_monitor: ResourceMonitor,
    memory_manager: MemoryPoolManager,
    auto_tuner: AutoTuner,
    performance_monitor: RealTimeMonitor,
}

impl AdvancedPipelineOptimizer {
    pub fn new() -> Self {
        Self {
            resource_monitor: ResourceMonitor::new(),
            memory_manager: MemoryPoolManager::new(),
            auto_tuner: AutoTuner::new(),
            performance_monitor: RealTimeMonitor::new(),
        }
    }

    /// Optimize pipeline configuration based on current system state and data size
    pub fn optimize_pipeline_configuration(
        &mut self,
        pipeline_id: &str,
        data_size: usize,
    ) -> Result<OptimizedPipelineConfig> {
        // Get current system metrics
        let system_metrics = self.resource_monitor.get_current_metrics()?;

        // Determine optimal memory strategy
        let memory_strategy = self.memory_manager.determine_optimal_strategy(
            data_size,
            &system_metrics,
        )?;

        // Get historical data for the pipeline
        let similar_configs = self.performance_monitor.performance_history
            .get_similar_configurations(pipeline_id, data_size);

        // Use auto-tuner to optimize parameters
        let optimal_params = self.auto_tuner.optimize_parameters(
            &system_metrics,
            &similar_configs,
            data_size,
        )?;

        // Create optimized configuration
        Ok(OptimizedPipelineConfig {
            thread_count: optimal_params.thread_count,
            chunk_size: optimal_params.chunk_size,
            simd_optimization: optimal_params.simd_enabled,
            gpu_acceleration: optimal_params.gpu_enabled,
            compression_level: optimal_params.compression_level,
            io_buffer_size: optimal_params.io_buffer_size,
            memory_strategy,
            auto_scaling: true,
            cache_strategy: CacheStrategy::Adaptive { initial_capacity: 1000 },
            prefetch_strategy: optimal_params.prefetch_strategy,
            batch_processing: optimal_params.batch_processing,
        })
    }

    /// Update the optimizer with performance feedback
    pub fn update_performance(
        &mut self,
        pipeline_id: &str,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) -> Result<Vec<performance::RegressionAlert>> {
        // Update auto-tuner model
        self.auto_tuner.update_model(config, metrics)?;

        // Update performance monitoring
        let alerts = self.performance_monitor.update_metrics(pipeline_id, metrics)?;

        // Check if memory compaction is needed
        self.memory_manager.compact_if_needed()?;

        Ok(alerts)
    }

    /// Start monitoring a pipeline
    pub fn start_monitoring(&mut self, pipeline_id: &str, config: &OptimizedPipelineConfig) {
        self.performance_monitor.start_monitoring(pipeline_id, config);
    }

    /// Stop monitoring a pipeline
    pub fn stop_monitoring(&mut self, pipeline_id: &str) {
        self.performance_monitor.stop_monitoring(pipeline_id);
    }

    /// Get optimization statistics and health information
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let dashboard = self.performance_monitor.get_dashboard_data();
        
        OptimizationStats {
            active_pipelines: dashboard.active_pipeline_count,
            total_executions: dashboard.total_executions,
            average_throughput: dashboard.avg_throughput,
            memory_pools_active: self.memory_manager.pools.len(),
            system_cpu_usage: self.resource_monitor.system_metrics.cpu_usage,
            system_memory_utilization: self.resource_monitor.system_metrics.memory_usage.utilization,
        }
    }
}

impl Default for AdvancedPipelineOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization statistics for monitoring and reporting
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub active_pipelines: usize,
    pub total_executions: usize,
    pub average_throughput: f64,
    pub memory_pools_active: usize,
    pub system_cpu_usage: f64,
    pub system_memory_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_pipeline_optimizer_creation() {
        let optimizer = AdvancedPipelineOptimizer::new();
        assert_eq!(optimizer.memory_manager.pools.len(), 0);
    }

    #[test]
    fn test_pipeline_configuration_optimization() {
        let mut optimizer = AdvancedPipelineOptimizer::new();
        let result = optimizer.optimize_pipeline_configuration("test_pipeline", 1024 * 1024);
        
        assert!(result.is_ok());
        let config = result.expect("Operation failed");
        assert!(config.thread_count > 0);
        assert!(config.chunk_size > 0);
    }

    #[test]
    fn test_optimization_stats() {
        let optimizer = AdvancedPipelineOptimizer::new();
        let stats = optimizer.get_optimization_stats();
        
        assert_eq!(stats.active_pipelines, 0);
        assert_eq!(stats.memory_pools_active, 0);
    }
}