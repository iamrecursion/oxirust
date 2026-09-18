//! Advanced Performance Monitoring and Optimization
//!
//! This module provides comprehensive performance monitoring, analysis, and optimization
//! capabilities for Advanced mode text processing operations.

use crate::error::{Result, TextError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Comprehensive performance monitor for Advanced operations
#[derive(Debug)]
pub struct AdvancedPerformanceMonitor {
    /// Historical performance data
    metricshistory: Arc<RwLock<Vec<PerformanceDataPoint>>>,
    /// Real-time performance aggregator
    realtime_aggregator: Arc<Mutex<RealtimeAggregator>>,
    /// Performance alert thresholds
    alert_thresholds: PerformanceThresholds,
    /// System resource monitor
    resource_monitor: Arc<Mutex<SystemResourceMonitor>>,
    /// Optimization recommendations engine
    optimization_engine: Arc<Mutex<OptimizationEngine>>,
}

/// Single performance data point
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Timestamp of the measurement
    pub timestamp: Instant,
    /// Operation type that was measured
    pub operationtype: String,
    /// Processing time for the operation
    pub processing_time: Duration,
    /// Number of items processed
    pub itemsprocessed: usize,
    /// Memory usage during operation (bytes)
    pub memory_usage: usize,
    /// CPU utilization percentage (0-100)
    pub cpu_utilization: f64,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization: f64,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Real-time performance aggregator
#[derive(Debug)]
struct RealtimeAggregator {
    /// Current operation start time
    current_operation: Option<Instant>,
    /// Running statistics
    running_stats: HashMap<String, RunningStatistics>,
    /// Alert counters
    alert_counts: HashMap<String, usize>,
}

/// Running statistics for performance metrics
#[derive(Debug, Clone)]
struct RunningStatistics {
    /// Number of samples
    count: usize,
    /// Sum of values
    sum: f64,
    /// Sum of squared values (for variance calculation)
    sum_squared: f64,
    /// Minimum value seen
    min: f64,
    /// Maximum value seen
    max: f64,
    /// Moving average (exponential)
    moving_average: f64,
}

/// Performance alert thresholds
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable processing time (milliseconds)
    pub max_processing_time_ms: u64,
    /// Minimum acceptable throughput (items/second)
    pub min_throughput: f64,
    /// Maximum acceptable memory usage (MB)
    pub max_memory_usage_mb: usize,
    /// Maximum acceptable CPU utilization (percentage)
    pub max_cpu_utilization: f64,
    /// Minimum acceptable cache hit rate
    pub min_cache_hit_rate: f64,
}

/// System resource monitor
#[derive(Debug)]
struct SystemResourceMonitor {
    /// Memory usage tracker
    memory_tracker: MemoryTracker,
    /// CPU usage tracker
    cpu_tracker: CpuUsageTracker,
    /// GPU usage tracker (if available)
    #[allow(dead_code)]
    gpu_tracker: Option<GpuUsageTracker>,
    /// Network I/O tracker
    network_tracker: NetworkTracker,
}

/// Memory usage tracking
#[derive(Debug)]
struct MemoryTracker {
    /// Peak memory usage (bytes)
    peak_usage: usize,
    /// Current memory usage (bytes)
    #[allow(dead_code)]
    current_usage: usize,
    /// Memory allocation events
    #[allow(dead_code)]
    allocations: Vec<AllocationEvent>,
}

/// Memory allocation event
#[derive(Debug, Clone)]
struct AllocationEvent {
    /// Timestamp of allocation
    #[allow(dead_code)]
    timestamp: Instant,
    /// Size allocated (bytes)
    #[allow(dead_code)]
    size: usize,
    /// Allocation type
    #[allow(dead_code)]
    allocation_type: String,
}

/// CPU usage tracking
#[derive(Debug)]
struct CpuUsageTracker {
    /// CPU usage samples
    #[allow(dead_code)]
    usage_samples: Vec<CpuUsageSample>,
    /// Current load average
    load_average: f64,
}

/// CPU usage sample
#[derive(Debug, Clone)]
struct CpuUsageSample {
    /// Timestamp of sample
    #[allow(dead_code)]
    timestamp: Instant,
    /// CPU utilization percentage
    #[allow(dead_code)]
    utilization: f64,
}

/// GPU usage tracking
#[derive(Debug)]
struct GpuUsageTracker {
    /// GPU utilization samples
    #[allow(dead_code)]
    utilization_samples: Vec<GpuUsageSample>,
    /// GPU memory usage (bytes)
    #[allow(dead_code)]
    memory_usage: usize,
}

/// GPU usage sample
#[derive(Debug, Clone)]
struct GpuUsageSample {
    /// Timestamp of sample
    #[allow(dead_code)]
    timestamp: Instant,
    /// GPU utilization percentage
    #[allow(dead_code)]
    utilization: f64,
    /// Memory utilization percentage
    #[allow(dead_code)]
    memory_utilization: f64,
}

/// Network I/O tracking
#[derive(Debug)]
struct NetworkTracker {
    /// Bytes sent
    bytes_sent: usize,
    /// Bytes received
    bytes_received: usize,
    /// Network latency samples
    #[allow(dead_code)]
    latency_samples: Vec<NetworkLatencySample>,
}

/// Network latency sample
#[derive(Debug, Clone)]
struct NetworkLatencySample {
    /// Timestamp of sample
    #[allow(dead_code)]
    timestamp: Instant,
    /// Latency in milliseconds
    #[allow(dead_code)]
    latency_ms: f64,
}

/// Optimization recommendations engine
#[derive(Debug)]
struct OptimizationEngine {
    /// Performance patterns database
    patterndatabase: Vec<PerformancePattern>,
    /// Current optimization recommendations
    current_recommendations: Vec<OptimizationRecommendation>,
    /// Optimization history
    optimizationhistory: Vec<OptimizationApplication>,
}

/// Performance pattern for optimization
#[derive(Debug, Clone)]
struct PerformancePattern {
    /// Pattern identifier
    #[allow(dead_code)]
    id: String,
    /// Pattern description
    #[allow(dead_code)]
    description: String,
    /// Conditions that trigger this pattern
    conditions: Vec<PerformanceCondition>,
    /// Recommended optimizations
    recommendations: Vec<OptimizationRecommendation>,
}

/// Performance condition
#[derive(Debug, Clone)]
struct PerformanceCondition {
    /// Metric name
    metric: String,
    /// Comparison operator
    operator: ComparisonOperator,
    /// Threshold value
    threshold: f64,
}

/// Comparison operators for conditions
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ComparisonOperator {
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Equal to
    EqualTo,
    /// Greater than or equal to
    GreaterOrEqual,
    /// Less than or equal to
    LessOrEqual,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation identifier
    pub id: String,
    /// Category of optimization
    pub category: String,
    /// Detailed recommendation
    pub recommendation: String,
    /// Estimated performance impact (0.0-1.0)
    pub impact_estimate: f64,
    /// Implementation complexity (1-5)
    pub complexity: u8,
    /// Prerequisites for implementation
    pub prerequisites: Vec<String>,
}

/// Applied optimization record
#[derive(Debug, Clone)]
pub struct OptimizationApplication {
    /// Timestamp of application
    #[allow(dead_code)]
    timestamp: Instant,
    /// Optimization that was applied
    #[allow(dead_code)]
    optimization: OptimizationRecommendation,
    /// Performance before optimization
    #[allow(dead_code)]
    performance_before: PerformanceSnapshot,
    /// Performance after optimization
    #[allow(dead_code)]
    performance_after: Option<PerformanceSnapshot>,
}

/// Performance snapshot
#[derive(Debug, Clone)]
struct PerformanceSnapshot {
    /// Average processing time
    #[allow(dead_code)]
    avg_processing_time: Duration,
    /// Average throughput
    #[allow(dead_code)]
    avg_throughput: f64,
    /// Average memory usage
    #[allow(dead_code)]
    avg_memory_usage: usize,
    /// Average CPU utilization
    #[allow(dead_code)]
    avg_cpu_utilization: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_processing_time_ms: 1000, // 1 second
            min_throughput: 100.0,        // 100 items/sec
            max_memory_usage_mb: 8192,    // 8GB
            max_cpu_utilization: 90.0,    // 90%
            min_cache_hit_rate: 0.8,      // 80%
        }
    }
}

impl AdvancedPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            metricshistory: Arc::new(RwLock::new(Vec::new())),
            realtime_aggregator: Arc::new(Mutex::new(RealtimeAggregator::new())),
            alert_thresholds: PerformanceThresholds::default(),
            resource_monitor: Arc::new(Mutex::new(SystemResourceMonitor::new())),
            optimization_engine: Arc::new(Mutex::new(OptimizationEngine::new())),
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: PerformanceThresholds) -> Self {
        Self {
            metricshistory: Arc::new(RwLock::new(Vec::new())),
            realtime_aggregator: Arc::new(Mutex::new(RealtimeAggregator::new())),
            alert_thresholds: thresholds,
            resource_monitor: Arc::new(Mutex::new(SystemResourceMonitor::new())),
            optimization_engine: Arc::new(Mutex::new(OptimizationEngine::new())),
        }
    }

    /// Start monitoring an operation
    pub fn start_operation(&self, operationtype: &str) -> Result<OperationMonitor> {
        let mut aggregator = self.realtime_aggregator.lock().expect("Operation failed");
        aggregator.start_operation(operationtype)?;

        Ok(OperationMonitor {
            operationtype: operationtype.to_string(),
            start_time: Instant::now(),
            monitor: self,
        })
    }

    /// Record a performance data point
    pub fn record_performance(&self, datapoint: PerformanceDataPoint) -> Result<()> {
        // Add to history
        let mut history = self.metricshistory.write().expect("Operation failed");
        history.push(datapoint.clone());

        // Limit history size
        if history.len() > 10000 {
            history.drain(0..1000); // Remove oldest 1000 entries
        }
        drop(history);

        // Update real-time aggregator
        let mut aggregator = self.realtime_aggregator.lock().expect("Operation failed");
        aggregator.update_statistics(&datapoint)?;
        drop(aggregator);

        // Check for alerts
        self.check_alerts(&datapoint)?;

        // Update optimization recommendations
        let mut optimizer = self.optimization_engine.lock().expect("Operation failed");
        optimizer.update_recommendations(&datapoint)?;
        drop(optimizer);

        Ok(())
    }

    /// Get current performance summary
    pub fn get_performance_summary(&self) -> Result<PerformanceSummary> {
        let history = self.metricshistory.read().expect("Operation failed");
        let aggregator = self.realtime_aggregator.lock().expect("Operation failed");

        let recent_window = std::cmp::min(100, history.len());
        let recentdata = if recent_window > 0 {
            &history[history.len() - recent_window..]
        } else {
            &[]
        };

        let summary = PerformanceSummary {
            total_operations: history.len(),
            recent_avg_processing_time: Self::calculate_avg_processing_time(recentdata),
            recent_avg_throughput: Self::calculate_avg_throughput(recentdata),
            recent_avg_memory_usage: Self::calculate_avg_memory_usage(recentdata),
            cache_hit_rate: Self::calculate_avg_cache_hit_rate(recentdata),
            active_alerts: aggregator.get_active_alerts(),
            optimization_opportunities: self.get_optimization_opportunities()?,
        };

        Ok(summary)
    }

    /// Get optimization recommendations
    pub fn get_optimization_opportunities(&self) -> Result<Vec<OptimizationRecommendation>> {
        let optimizer = self.optimization_engine.lock().expect("Operation failed");
        Ok(optimizer.current_recommendations.clone())
    }

    /// Apply an optimization
    pub fn apply_optimization(&self, optimizationid: &str) -> Result<()> {
        let mut optimizer = self.optimization_engine.lock().expect("Operation failed");
        optimizer.apply_optimization(optimizationid)?;
        Ok(())
    }

    /// Get detailed performance report
    pub fn generate_performance_report(&self) -> Result<DetailedPerformanceReport> {
        // Get the summary first to avoid nested locking
        let summary = self.get_performance_summary()?;

        // Then acquire other locks
        let history = self.metricshistory.read().expect("Operation failed");
        let resource_monitor = self.resource_monitor.lock().expect("Operation failed");
        let optimization_engine = self.optimization_engine.lock().expect("Operation failed");

        let report = DetailedPerformanceReport {
            summary,
            historical_trends: Self::analyze_trends(&history),
            resource_utilization: resource_monitor.get_utilization_summary(),
            bottleneck_analysis: Self::identify_bottlenecks(&history),
            optimizationhistory: optimization_engine.optimizationhistory.clone(),
            recommendations: optimization_engine.current_recommendations.clone(),
        };

        Ok(report)
    }

    // Helper methods
    fn check_alerts(&self, datapoint: &PerformanceDataPoint) -> Result<()> {
        let mut aggregator = self.realtime_aggregator.lock().expect("Operation failed");

        if datapoint.processing_time.as_millis()
            > self.alert_thresholds.max_processing_time_ms as u128
        {
            aggregator.increment_alert("high_processing_time");
        }

        let throughput = datapoint.itemsprocessed as f64 / datapoint.processing_time.as_secs_f64();
        if throughput < self.alert_thresholds.min_throughput {
            aggregator.increment_alert("low_throughput");
        }

        if datapoint.memory_usage > self.alert_thresholds.max_memory_usage_mb * 1024 * 1024 {
            aggregator.increment_alert("high_memory_usage");
        }

        if datapoint.cpu_utilization > self.alert_thresholds.max_cpu_utilization {
            aggregator.increment_alert("high_cpu_utilization");
        }

        if datapoint.cache_hit_rate < self.alert_thresholds.min_cache_hit_rate {
            aggregator.increment_alert("low_cache_hit_rate");
        }

        Ok(())
    }

    fn calculate_avg_processing_time(data: &[PerformanceDataPoint]) -> Duration {
        if data.is_empty() {
            return Duration::from_millis(0);
        }

        let total_ms: u128 = data.iter().map(|d| d.processing_time.as_millis()).sum();
        Duration::from_millis((total_ms / data.len() as u128) as u64)
    }

    fn calculate_avg_throughput(data: &[PerformanceDataPoint]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let total_throughput: f64 = data
            .iter()
            .map(|d| d.itemsprocessed as f64 / d.processing_time.as_secs_f64())
            .sum();
        total_throughput / data.len() as f64
    }

    fn calculate_avg_memory_usage(data: &[PerformanceDataPoint]) -> usize {
        if data.is_empty() {
            return 0;
        }

        data.iter().map(|d| d.memory_usage).sum::<usize>() / data.len()
    }

    fn calculate_avg_cache_hit_rate(data: &[PerformanceDataPoint]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        data.iter().map(|d| d.cache_hit_rate).sum::<f64>() / data.len() as f64
    }

    fn analyze_trends(history: &[PerformanceDataPoint]) -> TrendAnalysis {
        TrendAnalysis {
            processing_time_trend: Self::calculate_trend(
                &history
                    .iter()
                    .map(|d| d.processing_time.as_millis() as f64)
                    .collect::<Vec<_>>(),
            ),
            throughput_trend: Self::calculate_trend(
                &history
                    .iter()
                    .map(|d| d.itemsprocessed as f64 / d.processing_time.as_secs_f64())
                    .collect::<Vec<_>>(),
            ),
            memory_usage_trend: Self::calculate_trend(
                &history
                    .iter()
                    .map(|d| d.memory_usage as f64)
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn calculate_trend(values: &[f64]) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Stable;
        }

        let mid_point = values.len() / 2;
        let first_half_avg = values[..mid_point].iter().sum::<f64>() / mid_point as f64;
        let second_half_avg =
            values[mid_point..].iter().sum::<f64>() / (values.len() - mid_point) as f64;

        let change_rate = (second_half_avg - first_half_avg) / first_half_avg;

        if change_rate > 0.1 {
            TrendDirection::Increasing
        } else if change_rate < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    fn identify_bottlenecks(history: &[PerformanceDataPoint]) -> Vec<BottleneckAnalysis> {
        let mut bottlenecks = Vec::new();

        // Analyze processing time bottlenecks
        let avg_processing_time = Self::calculate_avg_processing_time(history);
        if avg_processing_time.as_millis() > 500 {
            bottlenecks.push(BottleneckAnalysis {
                component: "Processing Time".to_string(),
                severity: if avg_processing_time.as_millis() > 1000 {
                    "High"
                } else {
                    "Medium"
                }
                .to_string(),
                description: format!(
                    "Average processing time is {}ms",
                    avg_processing_time.as_millis()
                ),
                recommendations: vec![
                    "Enable SIMD optimizations".to_string(),
                    "Increase parallel processing".to_string(),
                    "Optimize memory allocation".to_string(),
                ],
            });
        }

        // Analyze memory usage bottlenecks
        let avg_memory = Self::calculate_avg_memory_usage(history);
        if avg_memory > 4 * 1024 * 1024 * 1024 {
            // 4GB
            bottlenecks.push(BottleneckAnalysis {
                component: "Memory Usage".to_string(),
                severity: "High".to_string(),
                description: {
                    let avg_memory_mb = avg_memory / (1024 * 1024);
                    format!("Average memory usage is {avg_memory_mb} MB")
                },
                recommendations: vec![
                    "Implement memory pooling".to_string(),
                    "Use streaming processing".to_string(),
                    "Optimize data structures".to_string(),
                ],
            });
        }

        bottlenecks
    }
}

/// Operation monitor for tracking individual operations
pub struct OperationMonitor<'a> {
    operationtype: String,
    start_time: Instant,
    monitor: &'a AdvancedPerformanceMonitor,
}

impl<'a> OperationMonitor<'a> {
    /// Complete the operation and record performance
    pub fn complete(self, itemsprocessed: usize) -> Result<()> {
        let processing_time = self.start_time.elapsed();

        // Get current resource usage (simplified)
        let data_point = PerformanceDataPoint {
            timestamp: self.start_time,
            operationtype: self.operationtype,
            processing_time,
            itemsprocessed,
            memory_usage: 0,      // Would be measured in real implementation
            cpu_utilization: 0.0, // Would be measured in real implementation
            gpu_utilization: 0.0, // Would be measured in real implementation
            cache_hit_rate: 0.9,  // Would be measured in real implementation
            custom_metrics: HashMap::new(),
        };

        self.monitor.record_performance(data_point)
    }
}

/// Performance summary
#[derive(Debug)]
pub struct PerformanceSummary {
    /// Total number of operations recorded
    pub total_operations: usize,
    /// Recent average processing time
    pub recent_avg_processing_time: Duration,
    /// Recent average throughput
    pub recent_avg_throughput: f64,
    /// Recent average memory usage
    pub recent_avg_memory_usage: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Active performance alerts
    pub active_alerts: Vec<String>,
    /// Available optimization opportunities
    pub optimization_opportunities: Vec<OptimizationRecommendation>,
}

/// Detailed performance report
#[derive(Debug)]
pub struct DetailedPerformanceReport {
    /// Performance summary
    pub summary: PerformanceSummary,
    /// Historical trend analysis
    pub historical_trends: TrendAnalysis,
    /// Resource utilization summary
    pub resource_utilization: ResourceUtilizationSummary,
    /// Bottleneck analysis
    pub bottleneck_analysis: Vec<BottleneckAnalysis>,
    /// History of applied optimizations
    pub optimizationhistory: Vec<OptimizationApplication>,
    /// Current recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Trend analysis results
#[derive(Debug)]
pub struct TrendAnalysis {
    /// Processing time trend
    pub processing_time_trend: TrendDirection,
    /// Throughput trend
    pub throughput_trend: TrendDirection,
    /// Memory usage trend
    pub memory_usage_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug)]
pub enum TrendDirection {
    /// Metric is increasing
    Increasing,
    /// Metric is decreasing
    Decreasing,
    /// Metric is stable
    Stable,
}

/// Resource utilization summary
#[derive(Debug)]
pub struct ResourceUtilizationSummary {
    /// Average CPU utilization
    pub avg_cpu_utilization: f64,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Network I/O summary
    pub network_io: NetworkIOSummary,
}

/// Network I/O summary
#[derive(Debug)]
pub struct NetworkIOSummary {
    /// Total bytes sent
    pub bytes_sent: usize,
    /// Total bytes received
    pub bytes_received: usize,
    /// Average latency
    pub avg_latency_ms: f64,
}

/// Bottleneck analysis
#[derive(Debug)]
pub struct BottleneckAnalysis {
    /// Component with bottleneck
    pub component: String,
    /// Severity level
    pub severity: String,
    /// Description of the bottleneck
    pub description: String,
    /// Recommendations to address it
    pub recommendations: Vec<String>,
}

// Implementation stubs for supporting structures
impl RealtimeAggregator {
    fn new() -> Self {
        Self {
            current_operation: None,
            running_stats: HashMap::new(),
            alert_counts: HashMap::new(),
        }
    }

    fn start_operation(&mut self, _operationtype: &str) -> Result<()> {
        self.current_operation = Some(Instant::now());
        Ok(())
    }

    fn update_statistics(&mut self, datapoint: &PerformanceDataPoint) -> Result<()> {
        let key = &datapoint.operationtype;
        let stats = self
            .running_stats
            .entry(key.clone())
            .or_insert_with(RunningStatistics::new);
        stats.update(datapoint.processing_time.as_millis() as f64);
        Ok(())
    }

    fn increment_alert(&mut self, alerttype: &str) {
        *self.alert_counts.entry(alerttype.to_string()).or_insert(0) += 1;
    }

    fn get_active_alerts(&self) -> Vec<String> {
        self.alert_counts.keys().cloned().collect()
    }
}

impl RunningStatistics {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_squared: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            moving_average: 0.0,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squared += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Update moving average with exponential decay
        let alpha = 0.1;
        self.moving_average = alpha * value + (1.0 - alpha) * self.moving_average;
    }
}

impl SystemResourceMonitor {
    fn new() -> Self {
        Self {
            memory_tracker: MemoryTracker::new(),
            cpu_tracker: CpuUsageTracker::new(),
            gpu_tracker: None,
            network_tracker: NetworkTracker::new(),
        }
    }

    fn get_utilization_summary(&self) -> ResourceUtilizationSummary {
        ResourceUtilizationSummary {
            avg_cpu_utilization: self.cpu_tracker.load_average,
            peak_memory_usage: self.memory_tracker.peak_usage,
            network_io: NetworkIOSummary {
                bytes_sent: self.network_tracker.bytes_sent,
                bytes_received: self.network_tracker.bytes_received,
                avg_latency_ms: 5.0, // Placeholder
            },
        }
    }
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            allocations: Vec::new(),
        }
    }
}

impl CpuUsageTracker {
    fn new() -> Self {
        Self {
            usage_samples: Vec::new(),
            load_average: 0.0,
        }
    }
}

impl NetworkTracker {
    fn new() -> Self {
        Self {
            bytes_sent: 0,
            bytes_received: 0,
            latency_samples: Vec::new(),
        }
    }
}

impl OptimizationEngine {
    fn new() -> Self {
        Self {
            patterndatabase: Self::initialize_patterns(),
            current_recommendations: Vec::new(),
            optimizationhistory: Vec::new(),
        }
    }

    fn initialize_patterns() -> Vec<PerformancePattern> {
        vec![PerformancePattern {
            id: "high_processing_time".to_string(),
            description: "Processing time is consistently high".to_string(),
            conditions: vec![PerformanceCondition {
                metric: "processing_time_ms".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 1000.0,
            }],
            recommendations: vec![
                OptimizationRecommendation {
                    id: "enable_simd".to_string(),
                    category: "Performance".to_string(),
                    recommendation: "Enable SIMD optimizations for string operations".to_string(),
                    impact_estimate: 0.3,
                    complexity: 2,
                    prerequisites: vec!["SIMD-capable hardware".to_string()],
                },
                OptimizationRecommendation {
                    id: "increase_parallelism".to_string(),
                    category: "Performance".to_string(),
                    recommendation: "Increase parallel processing threads".to_string(),
                    impact_estimate: 0.25,
                    complexity: 1,
                    prerequisites: vec!["Multi-core CPU".to_string()],
                },
            ],
        }]
    }

    fn update_recommendations(&mut self, datapoint: &PerformanceDataPoint) -> Result<()> {
        // Analyze current performance against patterns
        for pattern in &self.patterndatabase {
            if self.matches_pattern(datapoint, pattern) {
                // Add recommendations if not already present
                for recommendation in &pattern.recommendations {
                    if !self
                        .current_recommendations
                        .iter()
                        .any(|r| r.id == recommendation.id)
                    {
                        self.current_recommendations.push(recommendation.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn matches_pattern(
        &self,
        data_point: &PerformanceDataPoint,
        pattern: &PerformancePattern,
    ) -> bool {
        pattern.conditions.iter().all(|condition| {
            let value = match condition.metric.as_str() {
                "processing_time_ms" => data_point.processing_time.as_millis() as f64,
                "cpu_utilization" => data_point.cpu_utilization,
                "memory_usage_mb" => data_point.memory_usage as f64 / (1024.0 * 1024.0),
                "cache_hit_rate" => data_point.cache_hit_rate,
                _ => return false,
            };

            match condition.operator {
                ComparisonOperator::GreaterThan => value > condition.threshold,
                ComparisonOperator::LessThan => value < condition.threshold,
                ComparisonOperator::EqualTo => (value - condition.threshold).abs() < 0.001,
                ComparisonOperator::GreaterOrEqual => value >= condition.threshold,
                ComparisonOperator::LessOrEqual => value <= condition.threshold,
            }
        })
    }

    fn apply_optimization(&mut self, optimizationid: &str) -> Result<()> {
        if let Some(optimization) = self
            .current_recommendations
            .iter()
            .find(|r| r.id == optimizationid)
        {
            let application = OptimizationApplication {
                timestamp: Instant::now(),
                optimization: optimization.clone(),
                performance_before: PerformanceSnapshot {
                    avg_processing_time: Duration::from_millis(100),
                    avg_throughput: 1000.0,
                    avg_memory_usage: 1024 * 1024 * 1024,
                    avg_cpu_utilization: 75.0,
                },
                performance_after: None, // Would be filled in later
            };

            self.optimizationhistory.push(application);

            // Remove from current recommendations
            self.current_recommendations
                .retain(|r| r.id != optimizationid);

            Ok(())
        } else {
            Err(TextError::InvalidInput(format!(
                "Optimization not found: {optimizationid}"
            )))
        }
    }
}

impl Default for AdvancedPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = AdvancedPerformanceMonitor::new();
        let summary = monitor.get_performance_summary().expect("Operation failed");
        assert_eq!(summary.total_operations, 0);
    }

    #[test]
    fn test_operation_monitoring() {
        let monitor = AdvancedPerformanceMonitor::new();
        let op_monitor = monitor
            .start_operation("test_operation")
            .expect("Operation failed");

        // Simulate some work
        std::thread::sleep(Duration::from_millis(10));

        op_monitor.complete(100).expect("Operation failed");

        let summary = monitor.get_performance_summary().expect("Operation failed");
        assert_eq!(summary.total_operations, 1);
    }

    #[test]
    fn test_performance_thresholds() {
        let thresholds = PerformanceThresholds {
            max_processing_time_ms: 500,
            min_throughput: 200.0,
            max_memory_usage_mb: 4096,
            max_cpu_utilization: 80.0,
            min_cache_hit_rate: 0.9,
        };

        let monitor = AdvancedPerformanceMonitor::with_thresholds(thresholds);

        // Test with data point that should trigger alerts
        let data_point = PerformanceDataPoint {
            timestamp: Instant::now(),
            operationtype: "test".to_string(),
            processing_time: Duration::from_millis(1000), // Above threshold
            itemsprocessed: 10,
            memory_usage: 6 * 1024 * 1024 * 1024, // 6GB - above threshold
            cpu_utilization: 95.0,                // Above threshold
            gpu_utilization: 50.0,
            cache_hit_rate: 0.7, // Below threshold
            custom_metrics: HashMap::new(),
        };

        monitor
            .record_performance(data_point)
            .expect("Operation failed");

        let summary = monitor.get_performance_summary().expect("Operation failed");
        assert!(!summary.active_alerts.is_empty());
    }

    #[test]
    fn test_optimization_recommendations() {
        let monitor = AdvancedPerformanceMonitor::new();

        // Add a data point that should trigger optimization recommendations
        let data_point = PerformanceDataPoint {
            timestamp: Instant::now(),
            operationtype: "slow_operation".to_string(),
            processing_time: Duration::from_millis(2000), // High processing time
            itemsprocessed: 50,
            memory_usage: 1024 * 1024 * 1024, // 1GB
            cpu_utilization: 80.0,
            gpu_utilization: 0.0,
            cache_hit_rate: 0.9,
            custom_metrics: HashMap::new(),
        };

        monitor
            .record_performance(data_point)
            .expect("Operation failed");

        let recommendations = monitor
            .get_optimization_opportunities()
            .expect("Operation failed");
        assert!(!recommendations.is_empty());

        // Apply an optimization
        if let Some(first_rec) = recommendations.first() {
            monitor
                .apply_optimization(&first_rec.id)
                .expect("Operation failed");
        }
    }

    #[test]
    fn test_trend_analysis() {
        let monitor = AdvancedPerformanceMonitor::new();

        // Add multiple data points to create a trend
        for i in 1..=10 {
            let data_point = PerformanceDataPoint {
                timestamp: Instant::now(),
                operationtype: "trend_test".to_string(),
                processing_time: Duration::from_millis(100 + i * 10), // Increasing trend
                itemsprocessed: 100,
                memory_usage: 1024 * 1024 * i as usize, // Increasing memory
                cpu_utilization: 50.0 + i as f64,
                gpu_utilization: 0.0,
                cache_hit_rate: 0.9,
                custom_metrics: HashMap::new(),
            };

            monitor
                .record_performance(data_point)
                .expect("Operation failed");
        }

        let report = monitor
            .generate_performance_report()
            .expect("Operation failed");
        assert!(matches!(
            report.historical_trends.processing_time_trend,
            TrendDirection::Increasing
        ));
        assert!(matches!(
            report.historical_trends.memory_usage_trend,
            TrendDirection::Increasing
        ));
    }
}
