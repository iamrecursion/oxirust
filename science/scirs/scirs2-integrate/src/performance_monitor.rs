//! Performance monitoring and profiling for numerical algorithms
//!
//! This module provides comprehensive performance monitoring tools for analyzing
//! the behavior and efficiency of numerical integration algorithms. It tracks
//! various metrics including computation time, memory usage, convergence rates,
//! and algorithm-specific statistics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Comprehensive performance metrics for numerical algorithms
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total computation time
    pub total_time: Duration,
    /// Time spent in different algorithm phases
    pub phase_times: HashMap<String, Duration>,
    /// Number of function evaluations
    pub function_evaluations: usize,
    /// Number of jacobian evaluations
    pub jacobian_evaluations: usize,
    /// Number of linear system solves
    pub linear_solves: usize,
    /// Memory usage statistics
    pub memory_stats: MemoryStatistics,
    /// Algorithm-specific metrics
    pub algorithmmetrics: HashMap<String, f64>,
    /// Convergence history
    pub convergence_history: Vec<f64>,
    /// Step size adaptation history
    pub step_size_history: Vec<f64>,
    /// Error estimates over time
    pub error_estimates: Vec<f64>,
    /// Cache performance statistics
    pub cache_stats: CacheStatistics,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Current memory usage (bytes)
    pub current_memory: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Memory bandwidth utilization
    pub bandwidth_utilization: Option<f64>,
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Estimated cache hit rate
    pub cache_hit_rate: Option<f64>,
    /// Memory access patterns
    pub access_patterns: HashMap<String, usize>,
    /// FLOPS (Floating Point Operations Per Second)
    pub flops: Option<f64>,
}

/// Performance profiler for tracking algorithm behavior
pub struct PerformanceProfiler {
    start_time: Instant,
    phase_timers: HashMap<String, Instant>,
    metrics: Arc<Mutex<PerformanceMetrics>>,
    is_active: bool,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_time: Duration::ZERO,
            phase_times: HashMap::new(),
            function_evaluations: 0,
            jacobian_evaluations: 0,
            linear_solves: 0,
            memory_stats: MemoryStatistics {
                peak_memory: 0,
                current_memory: 0,
                allocation_count: 0,
                deallocation_count: 0,
                bandwidth_utilization: None,
            },
            algorithmmetrics: HashMap::new(),
            convergence_history: Vec::new(),
            step_size_history: Vec::new(),
            error_estimates: Vec::new(),
            cache_stats: CacheStatistics {
                cache_hit_rate: None,
                access_patterns: HashMap::new(),
                flops: None,
            },
        }
    }
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            phase_timers: HashMap::new(),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            is_active: true,
        }
    }

    /// Start profiling a specific phase
    pub fn start_phase(&mut self, phasename: &str) {
        if !self.is_active {
            return;
        }
        self.phase_timers
            .insert(phasename.to_string(), Instant::now());
    }

    /// End profiling a specific phase
    pub fn end_phase(&mut self, phasename: &str) {
        if !self.is_active {
            return;
        }

        if let Some(start_time) = self.phase_timers.remove(phasename) {
            let duration = start_time.elapsed();
            if let Ok(mut metrics) = self.metrics.lock() {
                *metrics
                    .phase_times
                    .entry(phasename.to_string())
                    .or_insert(Duration::ZERO) += duration;
            }
        }
    }

    /// Record a function evaluation
    pub fn record_function_evaluation(&mut self) {
        if !self.is_active {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.function_evaluations += 1;
        }
    }

    /// Record a Jacobian evaluation
    pub fn record_jacobian_evaluation(&mut self) {
        if !self.is_active {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.jacobian_evaluations += 1;
        }
    }

    /// Record a linear system solve
    pub fn record_linear_solve(&mut self) {
        if !self.is_active {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.linear_solves += 1;
        }
    }

    /// Record convergence information
    pub fn record_convergence(&mut self, residualnorm: f64) {
        if !self.is_active {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.convergence_history.push(residualnorm);
        }
    }

    /// Record step size adaptation
    pub fn record_step_size(&mut self, stepsize: f64) {
        if !self.is_active {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.step_size_history.push(stepsize);
        }
    }

    /// Record error estimate
    pub fn record_error_estimate(&mut self, error: f64) {
        if !self.is_active {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.error_estimates.push(error);
        }
    }

    /// Record algorithm-specific metric
    pub fn record_metric(&mut self, name: &str, value: f64) {
        if !self.is_active {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.algorithmmetrics.insert(name.to_string(), value);
        }
    }

    /// Update memory statistics
    pub fn update_memory_stats(&mut self, current_memory: usize, peak_memory: usize) {
        if !self.is_active {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.memory_stats.current_memory = current_memory;
            if peak_memory > metrics.memory_stats.peak_memory {
                metrics.memory_stats.peak_memory = peak_memory;
            }
            metrics.memory_stats.allocation_count += 1;
        }
    }

    /// Estimate FLOPS based on operations and time
    pub fn estimate_flops(&mut self, operations: usize, time: Duration) {
        if !self.is_active || time.is_zero() {
            return;
        }

        let flops = operations as f64 / time.as_secs_f64();
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.cache_stats.flops = Some(flops);
        }
    }

    /// Finalize profiling and get metrics
    pub fn finalize(&self) -> PerformanceMetrics {
        let total_time = self.start_time.elapsed();

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.total_time = total_time;

            // Compute derived metrics
            self.compute_efficiencymetrics(&mut metrics);

            metrics.clone()
        } else {
            PerformanceMetrics::default()
        }
    }

    /// Compute efficiency and derived metrics
    fn compute_efficiencymetrics(&self, metrics: &mut PerformanceMetrics) {
        // Compute convergence rate if we have history
        if metrics.convergence_history.len() > 1 {
            let rates: Vec<f64> = metrics
                .convergence_history
                .windows(2)
                .map(|window| {
                    if window[0] > 0.0 && window[1] > 0.0 {
                        (window[1] / window[0]).log10()
                    } else {
                        0.0
                    }
                })
                .collect();

            if !rates.is_empty() {
                let avg_rate = rates.iter().sum::<f64>() / rates.len() as f64;
                metrics
                    .algorithmmetrics
                    .insert("convergence_rate".to_string(), avg_rate);
            }
        }

        // Compute function evaluation efficiency
        if metrics.function_evaluations > 0 && !metrics.total_time.is_zero() {
            let eval_rate = metrics.function_evaluations as f64 / metrics.total_time.as_secs_f64();
            metrics
                .algorithmmetrics
                .insert("evaluations_per_second".to_string(), eval_rate);
        }

        // Compute step size statistics
        if !metrics.step_size_history.is_empty() {
            let min_step = metrics
                .step_size_history
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_step = metrics
                .step_size_history
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let avg_step = metrics.step_size_history.iter().sum::<f64>()
                / metrics.step_size_history.len() as f64;

            metrics
                .algorithmmetrics
                .insert("min_step_size".to_string(), min_step);
            metrics
                .algorithmmetrics
                .insert("max_step_size".to_string(), max_step);
            metrics
                .algorithmmetrics
                .insert("avg_step_size".to_string(), avg_step);
        }

        // Memory efficiency metrics
        if metrics.memory_stats.peak_memory > 0 {
            let memory_mb = metrics.memory_stats.peak_memory as f64 / (1024.0 * 1024.0);
            metrics
                .algorithmmetrics
                .insert("peak_memory_mb".to_string(), memory_mb);
        }
    }

    /// Disable profiling for performance-critical sections
    pub fn disable(&mut self) {
        self.is_active = false;
    }

    /// Re-enable profiling
    pub fn enable(&mut self) {
        self.is_active = true;
    }

    /// Check if profiling is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }
}

/// Performance analysis utilities
pub struct PerformanceAnalyzer;

impl PerformanceAnalyzer {
    /// Analyze convergence characteristics
    pub fn analyze_convergence(metrics: &PerformanceMetrics) -> ConvergenceAnalysis {
        let mut analysis = ConvergenceAnalysis::default();

        if metrics.convergence_history.len() >= 2 {
            // Linear convergence detection
            let log_residuals: Vec<f64> = metrics
                .convergence_history
                .iter()
                .filter(|&&r| r > 0.0)
                .map(|&r| r.log10())
                .collect();

            if log_residuals.len() >= 3 {
                // Estimate convergence order using least squares
                let n = log_residuals.len();
                let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

                // Simple linear regression: log(r_n) = a + b*n
                let x_mean = x.iter().sum::<f64>() / n as f64;
                let y_mean = log_residuals.iter().sum::<f64>() / n as f64;

                let numerator: f64 = x
                    .iter()
                    .zip(&log_residuals)
                    .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
                    .sum();
                let denominator: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();

                if denominator.abs() > 1e-10 {
                    analysis.convergence_rate = Some(-numerator / denominator);
                }
            }

            analysis.final_residual = metrics.convergence_history.last().copied();
            analysis.initial_residual = metrics.convergence_history.first().copied();
        }

        analysis
    }

    /// Analyze performance bottlenecks
    pub fn identify_bottlenecks(metrics: &PerformanceMetrics) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // Phase time analysis
        let total_phase_time: Duration = metrics.phase_times.values().sum();
        if !total_phase_time.is_zero() {
            for (phase, duration) in &metrics.phase_times {
                let percentage = duration.as_secs_f64() / total_phase_time.as_secs_f64() * 100.0;
                if percentage > 30.0 {
                    bottlenecks.push(PerformanceBottleneck {
                        category: BottleneckCategory::ComputationPhase,
                        description: format!(
                            "Phase '{phase}' takes {percentage:.1}% of computation time"
                        ),
                        severity: if percentage > 50.0 {
                            Severity::High
                        } else {
                            Severity::Medium
                        },
                        suggested_improvements: vec![
                            "Consider algorithm optimization".to_string(),
                            "Check for unnecessary computations".to_string(),
                            "Consider parallelization".to_string(),
                        ],
                    });
                }
            }
        }

        // Memory usage analysis
        if metrics.memory_stats.peak_memory > 1024 * 1024 * 1024 {
            // > 1GB
            bottlenecks.push(PerformanceBottleneck {
                category: BottleneckCategory::Memory,
                description: format!(
                    "High memory usage: {:.1} MB",
                    metrics.memory_stats.peak_memory as f64 / (1024.0 * 1024.0)
                ),
                severity: Severity::Medium,
                suggested_improvements: vec![
                    "Consider using memory pooling".to_string(),
                    "Implement iterative refinement".to_string(),
                    "Use sparse data structures".to_string(),
                ],
            });
        }

        // Function evaluation efficiency
        if metrics.function_evaluations > 0 && !metrics.total_time.is_zero() {
            let eval_rate = metrics.function_evaluations as f64 / metrics.total_time.as_secs_f64();
            if eval_rate < 100.0 {
                // Less than 100 evaluations per second
                bottlenecks.push(PerformanceBottleneck {
                    category: BottleneckCategory::FunctionEvaluation,
                    description: format!("Low function evaluation rate: {eval_rate:.1} evals/sec"),
                    severity: Severity::Low,
                    suggested_improvements: vec![
                        "Optimize function implementation".to_string(),
                        "Consider SIMD vectorization".to_string(),
                        "Cache expensive computations".to_string(),
                    ],
                });
            }
        }

        bottlenecks
    }

    /// Generate performance report
    pub fn generate_report(metrics: &PerformanceMetrics) -> PerformanceReport {
        let convergence_analysis = Self::analyze_convergence(metrics);
        let bottlenecks = Self::identify_bottlenecks(metrics);

        PerformanceReport {
            metrics: metrics.clone(),
            convergence_analysis,
            bottlenecks,
            recommendations: Self::generate_recommendations(metrics),
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(metrics: &PerformanceMetrics) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check step size adaptation
        if !metrics.step_size_history.is_empty() {
            let min_step = metrics
                .step_size_history
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_step = metrics
                .step_size_history
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if max_step / min_step > 1000.0 {
                recommendations.push(OptimizationRecommendation {
                    category: "Step Size Control".to_string(),
                    description: "Large step size variations detected".to_string(),
                    suggestion: "Consider more aggressive step size adaptation or better initial step size estimation".to_string(),
                    expected_improvement: 15.0,
                });
            }
        }

        // Check convergence efficiency
        if let Some(rate) = metrics.algorithmmetrics.get("convergence_rate") {
            if *rate < 1.0 {
                recommendations.push(OptimizationRecommendation {
                    category: "Convergence".to_string(),
                    description: "Slow convergence detected".to_string(),
                    suggestion: "Consider using higher-order methods or better preconditioning"
                        .to_string(),
                    expected_improvement: 25.0,
                });
            }
        }

        recommendations
    }
}

/// Convergence analysis results
#[derive(Debug, Clone, Default)]
pub struct ConvergenceAnalysis {
    pub convergence_rate: Option<f64>,
    pub initial_residual: Option<f64>,
    pub final_residual: Option<f64>,
    pub convergence_order: Option<f64>,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub category: BottleneckCategory,
    pub description: String,
    pub severity: Severity,
    pub suggested_improvements: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum BottleneckCategory {
    ComputationPhase,
    Memory,
    FunctionEvaluation,
    LinearSolver,
    StepSizeControl,
}

#[derive(Debug, Clone)]
pub enum Severity {
    Low,
    Medium,
    High,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub category: String,
    pub description: String,
    pub suggestion: String,
    pub expected_improvement: f64, // Percentage
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub metrics: PerformanceMetrics,
    pub convergence_analysis: ConvergenceAnalysis,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub recommendations: Vec<OptimizationRecommendation>,
}

impl PerformanceReport {
    /// Print a formatted performance report
    pub fn print_summary(&self) {
        println!("=== Performance Analysis Report ===");
        println!(
            "Total computation time: {:.3}s",
            self.metrics.total_time.as_secs_f64()
        );
        println!(
            "Function evaluations: {}",
            self.metrics.function_evaluations
        );

        if let Some(rate) = self.metrics.algorithmmetrics.get("evaluations_per_second") {
            println!("Evaluation rate: {rate:.1} evals/sec");
        }

        println!(
            "Peak memory usage: {:.1} MB",
            self.metrics.memory_stats.peak_memory as f64 / (1024.0 * 1024.0)
        );

        if !self.bottlenecks.is_empty() {
            println!("\n=== Performance Bottlenecks ===");
            for bottleneck in &self.bottlenecks {
                println!("- {:?}: {}", bottleneck.category, bottleneck.description);
            }
        }

        if !self.recommendations.is_empty() {
            println!("\n=== Optimization Recommendations ===");
            for rec in &self.recommendations {
                println!(
                    "- {}: {} (Expected improvement: {:.1}%)",
                    rec.category, rec.suggestion, rec.expected_improvement
                );
            }
        }
    }
}

/// Macro for easy profiling of code blocks
#[macro_export]
macro_rules! profile_block {
    ($profiler:expr, $phase:expr, $code:block) => {{
        $profiler.start_phase($phase);
        let result = $code;
        $profiler.end_phase($phase);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new();

        // Simulate some work
        profiler.start_phase("initialization");
        thread::sleep(Duration::from_millis(10));
        profiler.end_phase("initialization");

        profiler.record_function_evaluation();
        profiler.record_convergence(1e-3);
        profiler.record_step_size(0.01);

        let metrics = profiler.finalize();

        assert!(metrics.total_time > Duration::ZERO);
        assert_eq!(metrics.function_evaluations, 1);
        assert_eq!(metrics.convergence_history.len(), 1);
        assert_eq!(metrics.step_size_history.len(), 1);
    }

    #[test]
    fn test_performance_analysis() {
        let mut metrics = PerformanceMetrics::default();

        // Add some test data
        metrics.convergence_history = vec![1e-1, 1e-2, 1e-3, 1e-4];
        metrics.step_size_history = vec![0.1, 0.05, 0.02, 0.01];

        let analysis = PerformanceAnalyzer::analyze_convergence(&metrics);
        assert!(analysis.convergence_rate.is_some());

        let report = PerformanceAnalyzer::generate_report(&metrics);
        assert!(!report.recommendations.is_empty() || report.bottlenecks.is_empty());
    }
}
