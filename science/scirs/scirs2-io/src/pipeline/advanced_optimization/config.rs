//! Configuration types and structures for advanced pipeline optimization
//!
//! This module contains all the configuration enums, structs, and result types
//! used throughout the advanced optimization system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

/// Optimized pipeline configuration with advanced settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPipelineConfig {
    pub thread_count: usize,
    pub chunk_size: usize,
    pub memory_strategy: MemoryStrategy,
    pub cache_config: CacheConfiguration,
    pub simd_optimization: bool,
    pub gpu_acceleration: bool,
    pub prefetch_strategy: PrefetchStrategy,
    pub compression_level: u8,
    pub io_buffer_size: usize,
    pub batch_processing: BatchProcessingMode,
}

/// Memory allocation and management strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryStrategy {
    /// Standard allocation with GC
    Standard,
    /// Memory pool allocation for reduced fragmentation
    MemoryPool { pool_size: usize },
    /// Memory mapping for large datasets
    MemoryMapped { chunk_size: usize },
    /// Streaming processing for very large datasets
    Streaming { buffer_size: usize },
    /// Hybrid approach combining multiple strategies
    Hybrid {
        small_data_threshold: usize,
        memory_pool_size: usize,
        streaming_threshold: usize,
    },
}

/// Cache configuration for optimal data locality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfiguration {
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub prefetch_distance: usize,
    pub cache_line_size: usize,
    pub temporal_locality_weight: f64,
    pub spatial_locality_weight: f64,
    pub replacement_policy: CacheReplacementPolicy,
}

impl Default for CacheConfiguration {
    fn default() -> Self {
        Self {
            l1_cache_size: 32 * 1024,  // 32KB
            l2_cache_size: 256 * 1024, // 256KB
            prefetch_distance: 64,
            cache_line_size: 64,
            temporal_locality_weight: 0.7,
            spatial_locality_weight: 0.3,
            replacement_policy: CacheReplacementPolicy::LRU,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheReplacementPolicy {
    LRU,
    LFU,
    ARC, // Adaptive Replacement Cache
    CLOCK,
}

/// Data prefetch strategy for reducing memory latency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    None,
    Sequential { distance: usize },
    Adaptive { learning_window: usize },
    Pattern { pattern_length: usize },
}

/// Batch processing mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchProcessingMode {
    Disabled,
    Fixed {
        batch_size: usize,
    },
    Dynamic {
        min_batch_size: usize,
        max_batch_size: usize,
        latency_target: Duration,
    },
    Adaptive {
        target_throughput: f64,
        adjustment_factor: f64,
    },
}

/// Individual execution record for performance tracking
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub timestamp: DateTime<Utc>,
    pub pipeline_id: String,
    pub config: OptimizedPipelineConfig,
    pub metrics: PipelinePerformanceMetrics,
}

/// Pipeline performance metrics for optimization feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelinePerformanceMetrics {
    pub throughput: f64,
    pub latency: Duration,
    pub peak_memory_usage: usize,
    pub avg_memory_usage: usize,
    pub cpu_utilization: f64,
    pub gpu_utilization: f64,
    pub io_wait_time: Duration,
    pub cache_hit_ratio: f64,
    pub data_size: usize,
    pub error_rate: f64,
    pub power_consumption: f64,
}

impl Default for PipelinePerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency: Duration::from_millis(100),
            peak_memory_usage: 0,
            avg_memory_usage: 0,
            cpu_utilization: 0.0,
            gpu_utilization: 0.0,
            io_wait_time: Duration::from_millis(0),
            cache_hit_ratio: 0.0,
            data_size: 0,
            error_rate: 0.0,
            power_consumption: 0.0,
        }
    }
}

/// System resource metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: MemoryUsage,
    pub io_utilization: f64,
    pub network_bandwidth_usage: f64,
    pub cache_performance: CachePerformance,
    pub numa_topology: NumaTopology,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.5,
            memory_usage: MemoryUsage {
                total: 8 * 1024 * 1024 * 1024,
                available: 4 * 1024 * 1024 * 1024,
                used: 4 * 1024 * 1024 * 1024,
                utilization: 0.5,
            },
            io_utilization: 0.3,
            network_bandwidth_usage: 0.2,
            cache_performance: CachePerformance {
                l1_hit_rate: 0.95,
                l2_hit_rate: 0.85,
                l3_hit_rate: 0.75,
                tlb_hit_rate: 0.99,
            },
            numa_topology: NumaTopology::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub total: u64,
    pub available: u64,
    pub used: u64,
    pub utilization: f64,
}

#[derive(Debug, Clone)]
pub struct CachePerformance {
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub tlb_hit_rate: f64,
}

#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
    pub preferred_node: usize,
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self {
            nodes: vec![NumaNode {
                id: 0,
                memory_size: 8 * 1024 * 1024 * 1024,
                cpu_cores: vec![0, 1, 2, 3],
            }],
            preferred_node: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NumaNode {
    pub id: usize,
    pub memory_size: u64,
    pub cpu_cores: Vec<usize>,
}

/// Auto-tuning parameters for optimization
#[derive(Debug, Clone)]
pub struct AutoTuningParameters {
    pub thread_count: usize,
    pub chunk_size: usize,
    pub simd_enabled: bool,
    pub gpu_enabled: bool,
    pub prefetch_strategy: PrefetchStrategy,
    pub compression_level: u8,
    pub io_buffer_size: usize,
    pub batch_processing: BatchProcessingMode,
}

impl Default for AutoTuningParameters {
    fn default() -> Self {
        Self {
            thread_count: num_cpus::get(),
            chunk_size: 1024,
            simd_enabled: true,
            gpu_enabled: false,
            prefetch_strategy: PrefetchStrategy::Sequential { distance: 64 },
            compression_level: 6,
            io_buffer_size: 64 * 1024,
            batch_processing: BatchProcessingMode::Disabled,
        }
    }
}

/// Performance regression detector using statistical methods
#[derive(Debug)]
pub struct RegressionDetector {
    recent_metrics: VecDeque<f64>,
    baseline_performance: f64,
    detection_window: usize,
    regression_threshold: f64,
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl RegressionDetector {
    pub fn new() -> Self {
        Self {
            recent_metrics: VecDeque::with_capacity(50),
            baseline_performance: 0.0,
            detection_window: 20,
            regression_threshold: 0.1, // 10% performance drop
        }
    }

    pub fn check_regression(&mut self, metrics: &PipelinePerformanceMetrics) {
        let performance_score = metrics.throughput;

        self.recent_metrics.push_back(performance_score);
        if self.recent_metrics.len() > self.detection_window {
            self.recent_metrics.pop_front();
        }

        // Update baseline if we have enough data
        if self.recent_metrics.len() >= self.detection_window {
            let avg_recent: f64 =
                self.recent_metrics.iter().sum::<f64>() / self.recent_metrics.len() as f64;

            if self.baseline_performance == 0.0 {
                self.baseline_performance = avg_recent;
            } else {
                // Check for regression
                let relative_change =
                    (avg_recent - self.baseline_performance) / self.baseline_performance;
                if relative_change < -self.regression_threshold {
                    // Performance regression detected
                    eprintln!(
                        "Performance regression detected: {:.2}% decrease from baseline",
                        -relative_change * 100.0
                    );
                }

                // Update baseline with exponential moving average
                self.baseline_performance = 0.9 * self.baseline_performance + 0.1 * avg_recent;
            }
        }
    }
}

/// Quantum optimization configuration
#[derive(Debug, Clone)]
pub struct QuantumOptimizationConfig {
    pub num_qubits: usize,
    pub annealing_steps: usize,
    pub temperature_schedule: Vec<f64>,
    pub tunneling_probability: f64,
}

impl Default for QuantumOptimizationConfig {
    fn default() -> Self {
        Self {
            num_qubits: 10,
            annealing_steps: 1000,
            temperature_schedule: (0..1000)
                .map(|i| 10.0 * (-5.0 * i as f64 / 1000.0).exp())
                .collect(),
            tunneling_probability: 0.1,
        }
    }
}

/// Neuromorphic optimization configuration
#[derive(Debug, Clone)]
pub struct NeuromorphicConfig {
    pub num_neurons: usize,
    pub num_outputs: usize,
    pub memory_capacity: usize,
    pub learning_rate: f64,
    pub adaptation_rate: f64,
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            num_neurons: 1000,
            num_outputs: 100,
            memory_capacity: 10000,
            learning_rate: 0.01,
            adaptation_rate: 0.001,
        }
    }
}

/// Consciousness-inspired optimization configuration
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    pub awareness_level: f64,
    pub attention_focus: f64,
    pub metacognitive_strength: f64,
    pub intentionality_weight: f64,
    pub max_cycles: usize,
    pub convergence_threshold: f64,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            awareness_level: 0.5,
            attention_focus: 0.7,
            metacognitive_strength: 0.6,
            intentionality_weight: 0.8,
            max_cycles: 100,
            convergence_threshold: 1e-6,
        }
    }
}
