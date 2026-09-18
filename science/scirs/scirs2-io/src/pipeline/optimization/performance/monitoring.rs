//! Performance monitoring and metrics tracking for pipeline optimization
//!
//! This module provides comprehensive performance tracking, regression detection,
//! and historical analysis for pipeline optimization and tuning.

use crate::error::Result;
use super::auto_tuning::OptimizedPipelineConfig;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Performance history tracker for machine learning optimization
#[derive(Debug)]
pub struct PerformanceHistory {
    executions: Vec<ExecutionRecord>,
    pipeline_profiles: HashMap<String, PipelineProfile>,
    max_history_size: usize,
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceHistory {
    pub fn new() -> Self {
        Self {
            executions: Vec::new(),
            pipeline_profiles: HashMap::new(),
            max_history_size: 10000,
        }
    }

    pub fn record_execution(
        &mut self,
        pipeline_id: &str,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) -> Result<()> {
        let record = ExecutionRecord {
            timestamp: Utc::now(),
            pipeline_id: pipeline_id.to_string(),
            config: config.clone(),
            metrics: metrics.clone(),
        };

        self.executions.push(record);

        // Maintain history size limit
        if self.executions.len() > self.max_history_size {
            self.executions.remove(0);
        }

        // Update or create pipeline profile
        self.update_pipeline_profile(pipeline_id, config, metrics);

        Ok(())
    }

    pub fn get_similar_configurations(
        &self,
        pipeline_id: &str,
        data_size: usize,
    ) -> Vec<&ExecutionRecord> {
        let size_threshold = 0.2; // 20% size difference tolerance

        self.executions
            .iter()
            .filter(|record| {
                record.pipeline_id == pipeline_id
                    && (record.metrics.data_size as f64 - data_size as f64).abs()
                        / (data_size as f64)
                        < size_threshold
            })
            .collect()
    }

    pub fn get_performance_trends(&self, pipeline_id: &str) -> Option<PerformanceTrends> {
        self.pipeline_profiles.get(pipeline_id).map(|profile| {
            let recent_executions: Vec<_> = self.executions
                .iter()
                .rev()
                .take(100)
                .filter(|e| e.pipeline_id == pipeline_id)
                .collect();

            if recent_executions.len() < 2 {
                return PerformanceTrends::default();
            }

            let throughput_trend = self.calculate_trend(
                &recent_executions.iter().map(|e| e.metrics.throughput).collect::<Vec<_>>()
            );
            let memory_trend = self.calculate_trend(
                &recent_executions.iter().map(|e| e.metrics.peak_memory_usage as f64).collect::<Vec<_>>()
            );

            PerformanceTrends {
                throughput_trend,
                memory_trend,
                execution_count: recent_executions.len(),
                time_span: recent_executions.first().expect("Operation failed").timestamp 
                    - recent_executions.last().expect("Operation failed").timestamp,
            }
        })
    }

    fn calculate_trend(&self, values: &[f64]) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Stable;
        }

        let first_half = &values[0..values.len()/2];
        let second_half = &values[values.len()/2..];

        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;

        let change_ratio = (second_avg - first_avg) / first_avg;

        if change_ratio > 0.05 {
            TrendDirection::Improving
        } else if change_ratio < -0.05 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    fn update_pipeline_profile(
        &mut self,
        pipeline_id: &str,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) {
        let profile = self
            .pipeline_profiles
            .entry(pipeline_id.to_string())
            .or_insert_with(|| PipelineProfile::new(pipeline_id));

        profile.update(config, metrics);
    }

    pub fn get_best_configurations(&self, pipeline_id: &str) -> Vec<OptimizedPipelineConfig> {
        self.pipeline_profiles
            .get(pipeline_id)
            .map(|profile| profile.optimal_configurations.clone())
            .unwrap_or_default()
    }
}

/// Individual execution record for performance tracking
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub timestamp: DateTime<Utc>,
    pub pipeline_id: String,
    pub config: OptimizedPipelineConfig,
    pub metrics: PipelinePerformanceMetrics,
}

/// Pipeline performance profile with statistical analysis
#[derive(Debug)]
pub struct PipelineProfile {
    pub pipeline_id: String,
    pub execution_count: usize,
    pub avg_throughput: f64,
    pub avg_memory_usage: f64,
    pub avg_cpu_utilization: f64,
    pub optimal_configurations: Vec<OptimizedPipelineConfig>,
    pub performance_regression_detector: RegressionDetector,
}

impl PipelineProfile {
    pub fn new(pipeline_id: &str) -> Self {
        Self {
            pipeline_id: pipeline_id.to_string(),
            execution_count: 0,
            avg_throughput: 0.0,
            avg_memory_usage: 0.0,
            avg_cpu_utilization: 0.0,
            optimal_configurations: Vec::new(),
            performance_regression_detector: RegressionDetector::new(),
        }
    }

    pub fn update(
        &mut self,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) {
        self.execution_count += 1;

        // Update running averages
        let weight = 1.0 / self.execution_count as f64;
        self.avg_throughput += weight * (metrics.throughput - self.avg_throughput);
        self.avg_memory_usage +=
            weight * (metrics.peak_memory_usage as f64 - self.avg_memory_usage);
        self.avg_cpu_utilization += weight * (metrics.cpu_utilization - self.avg_cpu_utilization);

        // Check for performance regression
        self.performance_regression_detector
            .check_regression(metrics);

        // Update optimal configurations if this is better
        if self.is_better_configuration(config, metrics) {
            self.optimal_configurations.push(config.clone());
            // Keep only top 5 configurations
            if self.optimal_configurations.len() > 5 {
                self.optimal_configurations.remove(0);
            }
        }
    }

    fn is_better_configuration(
        &self,
        _config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) -> bool {
        // Score based on throughput, memory efficiency, and CPU utilization
        let score = metrics.throughput * 0.5
            + (1.0 / metrics.peak_memory_usage as f64) * 0.3
            + metrics.cpu_utilization * 0.2;

        // Compare with average performance
        let avg_score = self.avg_throughput * 0.5
            + (1.0 / self.avg_memory_usage) * 0.3
            + self.avg_cpu_utilization * 0.2;

        score > avg_score * 1.1 // 10% improvement threshold
    }

    pub fn get_performance_statistics(&self) -> PerformanceStatistics {
        PerformanceStatistics {
            execution_count: self.execution_count,
            avg_throughput: self.avg_throughput,
            avg_memory_usage: self.avg_memory_usage,
            avg_cpu_utilization: self.avg_cpu_utilization,
            optimal_config_count: self.optimal_configurations.len(),
            has_regression: self.performance_regression_detector.recent_metrics.len() > 5,
        }
    }
}

/// Performance regression detector using statistical methods
#[derive(Debug)]
pub struct RegressionDetector {
    pub recent_metrics: VecDeque<f64>,
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
            recent_metrics: VecDeque::new(),
            baseline_performance: 0.0,
            detection_window: 10,
            regression_threshold: 0.1, // 10% degradation
        }
    }

    pub fn check_regression(&mut self, metrics: &PipelinePerformanceMetrics) -> Option<RegressionAlert> {
        let performance_score = metrics.throughput / (metrics.peak_memory_usage as f64).max(1.0);

        self.recent_metrics.push_back(performance_score);
        if self.recent_metrics.len() > self.detection_window {
            self.recent_metrics.pop_front();
        }

        if self.baseline_performance == 0.0 {
            self.baseline_performance = performance_score;
            return None;
        }

        // Check for statistically significant regression
        if self.recent_metrics.len() >= self.detection_window {
            let recent_avg: f64 =
                self.recent_metrics.iter().sum::<f64>() / self.recent_metrics.len() as f64;
            let regression_ratio =
                (self.baseline_performance - recent_avg) / self.baseline_performance;

            if regression_ratio > self.regression_threshold {
                return Some(RegressionAlert {
                    severity: if regression_ratio > 0.2 {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    regression_percentage: regression_ratio * 100.0,
                    baseline_performance: self.baseline_performance,
                    current_performance: recent_avg,
                    detection_window: self.detection_window,
                });
            }
        }

        None
    }

    pub fn reset_baseline(&mut self) {
        if let Some(&last_metric) = self.recent_metrics.back() {
            self.baseline_performance = last_metric;
        }
        self.recent_metrics.clear();
    }
}

/// Comprehensive performance metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelinePerformanceMetrics {
    pub execution_time: Duration,
    pub throughput: f64, // items per second
    pub peak_memory_usage: usize,
    pub avg_memory_usage: usize,
    pub cpu_utilization: f64,
    pub cache_hit_rate: f64,
    pub io_wait_time: Duration,
    pub network_io_bytes: usize,
    pub disk_io_bytes: usize,
    pub data_size: usize,
    pub error_count: usize,
    pub stage_performance: Vec<StagePerformance>,
}

impl Default for PipelinePerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time: Duration::from_millis(100),
            throughput: 1000.0,
            peak_memory_usage: 1024 * 1024,
            avg_memory_usage: 512 * 1024,
            cpu_utilization: 0.5,
            cache_hit_rate: 0.8,
            io_wait_time: Duration::from_millis(10),
            network_io_bytes: 0,
            disk_io_bytes: 1024 * 1024,
            data_size: 10000,
            error_count: 0,
            stage_performance: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePerformance {
    pub stage_name: String,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub cpu_utilization: f64,
    pub cache_misses: usize,
    pub simd_efficiency: f64,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub throughput_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub execution_count: usize,
    pub time_span: chrono::Duration,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            throughput_trend: TrendDirection::Stable,
            memory_trend: TrendDirection::Stable,
            execution_count: 0,
            time_span: chrono::Duration::zero(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Performance statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    pub execution_count: usize,
    pub avg_throughput: f64,
    pub avg_memory_usage: f64,
    pub avg_cpu_utilization: f64,
    pub optimal_config_count: usize,
    pub has_regression: bool,
}

/// Regression alert information
#[derive(Debug, Clone)]
pub struct RegressionAlert {
    pub severity: AlertSeverity,
    pub regression_percentage: f64,
    pub baseline_performance: f64,
    pub current_performance: f64,
    pub detection_window: usize,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Warning,
    Critical,
}

/// Real-time performance monitor
#[derive(Debug)]
pub struct RealTimeMonitor {
    performance_history: PerformanceHistory,
    active_pipelines: HashMap<String, ActivePipelineMonitor>,
    alert_threshold: f64,
}

impl RealTimeMonitor {
    pub fn new() -> Self {
        Self {
            performance_history: PerformanceHistory::new(),
            active_pipelines: HashMap::new(),
            alert_threshold: 0.15, // 15% degradation triggers alert
        }
    }

    pub fn start_monitoring(&mut self, pipeline_id: &str, config: &OptimizedPipelineConfig) {
        let monitor = ActivePipelineMonitor::new(pipeline_id, config);
        self.active_pipelines.insert(pipeline_id.to_string(), monitor);
    }

    pub fn update_metrics(&mut self, pipeline_id: &str, metrics: &PipelinePerformanceMetrics) -> Result<Vec<RegressionAlert>> {
        let mut alerts = Vec::new();

        // Update history
        if let Some(monitor) = self.active_pipelines.get(pipeline_id) {
            self.performance_history.record_execution(pipeline_id, &monitor.config, metrics)?;
        }

        // Check for regressions
        if let Some(profile) = self.performance_history.pipeline_profiles.get_mut(pipeline_id) {
            if let Some(alert) = profile.performance_regression_detector.check_regression(metrics) {
                alerts.push(alert);
            }
        }

        Ok(alerts)
    }

    pub fn stop_monitoring(&mut self, pipeline_id: &str) {
        self.active_pipelines.remove(pipeline_id);
    }

    pub fn get_dashboard_data(&self) -> MonitoringDashboard {
        let active_pipeline_count = self.active_pipelines.len();
        let total_executions = self.performance_history.executions.len();
        
        let avg_throughput = if !self.performance_history.executions.is_empty() {
            self.performance_history.executions.iter()
                .map(|e| e.metrics.throughput)
                .sum::<f64>() / self.performance_history.executions.len() as f64
        } else {
            0.0
        };

        MonitoringDashboard {
            active_pipeline_count,
            total_executions,
            avg_throughput,
            pipeline_profiles: self.performance_history.pipeline_profiles.keys().cloned().collect(),
        }
    }
}

impl Default for RealTimeMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ActivePipelineMonitor {
    pub pipeline_id: String,
    pub config: OptimizedPipelineConfig,
    pub start_time: DateTime<Utc>,
}

impl ActivePipelineMonitor {
    pub fn new(pipeline_id: &str, config: &OptimizedPipelineConfig) -> Self {
        Self {
            pipeline_id: pipeline_id.to_string(),
            config: config.clone(),
            start_time: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringDashboard {
    pub active_pipeline_count: usize,
    pub total_executions: usize,
    pub avg_throughput: f64,
    pub pipeline_profiles: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::auto_tuning::{CacheStrategy, PrefetchStrategy, BatchProcessingMode};

    fn create_test_config() -> OptimizedPipelineConfig {
        OptimizedPipelineConfig {
            thread_count: 4,
            chunk_size: 8192,
            simd_optimization: true,
            gpu_acceleration: false,
            compression_level: 6,
            io_buffer_size: 64 * 1024,
            memory_strategy: crate::pipeline::optimization::memory::pool_management::MemoryStrategy::Standard,
            auto_scaling: true,
            cache_strategy: CacheStrategy::LRU { capacity: 1000 },
            prefetch_strategy: PrefetchStrategy::Sequential { distance: 4 },
            batch_processing: BatchProcessingMode::Fixed { batch_size: 100 },
        }
    }

    fn create_test_metrics() -> PipelinePerformanceMetrics {
        PipelinePerformanceMetrics {
            execution_time: Duration::from_millis(100),
            throughput: 1500.0,
            peak_memory_usage: 2 * 1024 * 1024,
            avg_memory_usage: 1024 * 1024,
            cpu_utilization: 0.7,
            cache_hit_rate: 0.9,
            io_wait_time: Duration::from_millis(5),
            network_io_bytes: 0,
            disk_io_bytes: 1024 * 1024,
            data_size: 10000,
            error_count: 0,
            stage_performance: Vec::new(),
        }
    }

    #[test]
    fn test_performance_history_creation() {
        let history = PerformanceHistory::new();
        assert_eq!(history.max_history_size, 10000);
        assert!(history.executions.is_empty());
        assert!(history.pipeline_profiles.is_empty());
    }

    #[test]
    fn test_execution_recording() {
        let mut history = PerformanceHistory::new();
        let config = create_test_config();
        let metrics = create_test_metrics();

        let result = history.record_execution("test_pipeline", &config, &metrics);
        assert!(result.is_ok());
        assert_eq!(history.executions.len(), 1);
        assert_eq!(history.pipeline_profiles.len(), 1);
    }

    #[test]
    fn test_regression_detector() {
        let mut detector = RegressionDetector::new();
        let metrics = create_test_metrics();

        // First execution sets baseline
        let alert = detector.check_regression(&metrics);
        assert!(alert.is_none());

        // Simulate performance degradation
        let mut degraded_metrics = metrics;
        degraded_metrics.throughput = 500.0; // Significant drop

        // Add several degraded measurements
        for _ in 0..10 {
            detector.check_regression(&degraded_metrics);
        }

        let alert = detector.check_regression(&degraded_metrics);
        assert!(alert.is_some());
        if let Some(alert) = alert {
            assert!(alert.regression_percentage > 10.0);
        }
    }

    #[test]
    fn test_real_time_monitor() {
        let mut monitor = RealTimeMonitor::new();
        let config = create_test_config();
        let metrics = create_test_metrics();

        monitor.start_monitoring("test_pipeline", &config);
        assert_eq!(monitor.active_pipelines.len(), 1);

        let alerts = monitor.update_metrics("test_pipeline", &metrics).expect("Operation failed");
        assert!(alerts.is_empty()); // No alerts on first update

        monitor.stop_monitoring("test_pipeline");
        assert_eq!(monitor.active_pipelines.len(), 0);
    }

    #[test]
    fn test_pipeline_profile_update() {
        let mut profile = PipelineProfile::new("test");
        let config = create_test_config();
        let metrics = create_test_metrics();

        profile.update(&config, &metrics);
        assert_eq!(profile.execution_count, 1);
        assert_eq!(profile.avg_throughput, metrics.throughput);
    }
}