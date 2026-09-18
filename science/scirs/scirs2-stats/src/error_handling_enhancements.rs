//! Advanced Error Handling Enhancements
//!
//! Advanced error handling specifically designed for Advanced mode operations,
//! providing intelligent diagnostics, performance-aware suggestions, and
//! adaptive recovery strategies.

use crate::error::StatsError;
use std::time::{Duration, Instant};

/// Advanced-specific error context with performance tracking
#[derive(Debug, Clone)]
pub struct AdvancedErrorContext {
    pub operation_start: Instant,
    pub datasize: usize,
    pub memory_usage_mb: f64,
    pub simd_enabled: bool,
    pub parallel_enabled: bool,
    pub suggested_optimization: Option<OptimizationSuggestion>,
}

/// Performance-aware optimization suggestions
#[derive(Debug, Clone)]
pub enum OptimizationSuggestion {
    EnableSIMD {
        reason: String,
        expected_speedup: f64,
    },
    EnableParallel {
        reason: String,
        mindatasize: usize,
    },
    ReduceMemoryUsage {
        current_mb: f64,
        suggested_mb: f64,
        strategy: String,
    },
    ChunkProcessing {
        chunksize: usize,
        reason: String,
    },
    AlgorithmChange {
        current: String,
        suggested: String,
        reason: String,
    },
}

/// Enhanced error messages for Advanced mode
pub struct AdvancedErrorMessages;

impl AdvancedErrorMessages {
    /// Memory exhaustion with intelligent suggestions
    pub fn memory_exhaustion(required_mb: f64, available_mb: f64, datasize: usize) -> StatsError {
        let suggestion = if datasize > 10_000_000 {
            "Consider using chunked processing or streaming algorithms for large datasets."
        } else if required_mb > available_mb * 0.8 {
            "Use memory-efficient algorithms or increase available memory."
        } else {
            "Enable garbage collection or reduce concurrent operations."
        };

        StatsError::computation(format!(
            "Memory exhaustion: operation requires {:.1}MB but only {:.1}MB available. \
             Data size: {} elements. Suggestion: {}",
            required_mb, available_mb, datasize, suggestion
        ))
    }

    /// Performance degradation warnings
    pub fn performance_degradation(
        operation: &str,
        expected_duration: Duration,
        actual_duration: Duration,
        context: &AdvancedErrorContext,
    ) -> StatsError {
        let slowdown_factor = actual_duration.as_secs_f64() / expected_duration.as_secs_f64();

        let suggestion = match slowdown_factor {
            x if x > 10.0 => {
                if !context.simd_enabled && context.datasize > 1000 {
                    "Enable SIMD operations for significant performance improvement."
                } else if !context.parallel_enabled && context.datasize > 10_000 {
                    "Enable parallel processing for better performance on large datasets."
                } else {
                    "Check for memory pressure or system resource contention."
                }
            }
            x if x > 3.0 => "Consider optimizing data layout or using more efficient algorithms.",
            _ => "Performance is within acceptable range but could be optimized.",
        };

        StatsError::computation(format!(
            "Performance degradation in {}: expected {:.3}s, actual {:.3}s ({}x slower). \
             Data size: {} elements, SIMD: {}, Parallel: {}. Suggestion: {}",
            operation,
            expected_duration.as_secs_f64(),
            actual_duration.as_secs_f64(),
            slowdown_factor,
            context.datasize,
            context.simd_enabled,
            context.parallel_enabled,
            suggestion
        ))
    }

    /// Numerical precision warnings
    pub fn precision_warning(
        operation: &str,
        precision_loss: f64,
        data_characteristics: &str,
    ) -> StatsError {
        let suggestion = match precision_loss {
            x if x > 1e-6 => {
                "Use higher precision arithmetic or reorder operations to minimize error accumulation."
            }
            x if x > 1e-12 => {
                "Consider using numerically stable algorithms or regularization."
            }
            _ => "Precision _loss is minimal but monitor for accumulation in iterative algorithms."
        };

        StatsError::computation(format!(
            "Precision _loss detected in {}: estimated error {:.2e}. Data: {}. Suggestion: {}",
            operation, precision_loss, data_characteristics, suggestion
        ))
    }

    /// SIMD/parallel optimization recommendations
    pub fn optimization_opportunity(
        operation: &str,
        datasize: usize,
        current_performance: Duration,
        suggestion: OptimizationSuggestion,
    ) -> StatsError {
        let message = match suggestion {
            OptimizationSuggestion::EnableSIMD {
                reason,
                expected_speedup,
            } => {
                format!(
                    "SIMD optimization available for {}: {} Expected speedup: {:.1}x",
                    operation, reason, expected_speedup
                )
            }
            OptimizationSuggestion::EnableParallel {
                reason,
                mindatasize,
            } => {
                format!(
                    "Parallel processing recommended for {}: {} Minimum data size: {}",
                    operation, reason, mindatasize
                )
            }
            OptimizationSuggestion::ReduceMemoryUsage {
                current_mb,
                suggested_mb,
                strategy,
            } => {
                format!(
                    "Memory optimization for {}: reduce from {:.1}MB to {:.1}MB using {}",
                    operation, current_mb, suggested_mb, strategy
                )
            }
            OptimizationSuggestion::ChunkProcessing { chunksize, reason } => {
                format!(
                    "Chunked processing recommended for {}: use chunks of {} elements. {}",
                    operation, chunksize, reason
                )
            }
            OptimizationSuggestion::AlgorithmChange {
                current,
                suggested,
                reason,
            } => {
                format!(
                    "Algorithm optimization for {}: change from {} to {}. {}",
                    operation, current, suggested, reason
                )
            }
        };

        StatsError::computation(format!(
            "Optimization opportunity: {} Data , size: {} elements, Current time: {:.3}s",
            message,
            datasize,
            current_performance.as_secs_f64()
        ))
    }

    /// Resource contention warnings
    pub fn resource_contention(
        operation: &str,
        resource_type: &str,
        utilization: f64,
        impact: &str,
    ) -> StatsError {
        let suggestion = match resource_type {
            "cpu" if utilization > 0.9 => {
                "High CPU utilization detected. Consider reducing parallel thread count or scheduling operations."
            }
            "memory" if utilization > 0.8 => {
                "High memory utilization detected. Use streaming algorithms or reduce batch sizes."
            }
            "cache" => {
                "Cache pressure detected. Optimize data access patterns or reduce working set size."
            }
            _ => "Resource contention detected. Monitor system resources and adjust accordingly."
        };

        StatsError::computation(format!(
            "Resource contention in {}: {} utilization at {:.1}%. Impact: {}. Suggestion: {}",
            operation,
            resource_type,
            utilization * 100.0,
            impact,
            suggestion
        ))
    }
}

/// Adaptive error recovery system for Advanced mode
pub struct AdvancedErrorRecovery;

impl AdvancedErrorRecovery {
    /// Attempt automatic error recovery with performance optimization
    pub fn attempt_recovery(
        error: &StatsError,
        context: &AdvancedErrorContext,
        operation: &str,
    ) -> Option<RecoveryStrategy> {
        match error {
            StatsError::ComputationError(msg) if msg.contains("memory") => {
                Some(RecoveryStrategy::ReduceMemoryFootprint {
                    chunksize: context.datasize / 4,
                    streaming: true,
                })
            }
            StatsError::ComputationError(msg) if msg.contains("performance") => {
                if !context.simd_enabled && context.datasize > 1000 {
                    Some(RecoveryStrategy::EnableOptimizations {
                        simd: true,
                        parallel: context.datasize > 10_000,
                    })
                } else {
                    Some(RecoveryStrategy::AlgorithmFallback {
                        from: "optimized",
                        to: "stable",
                    })
                }
            }
            StatsError::ComputationError(msg) if msg.contains("precision") => {
                Some(RecoveryStrategy::IncreasePrecision {
                    use_f64: true,
                    use_stable_algorithms: true,
                })
            }
            _ => None,
        }
    }

    /// Generate context-aware recovery suggestions
    pub fn generate_suggestions(
        _error: &StatsError,
        context: &AdvancedErrorContext,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Data size-based suggestions
        if context.datasize > 1_000_000 {
            suggestions.push("Consider using chunked processing for large datasets".to_string());
        }

        // Performance-based suggestions
        if !context.simd_enabled && context.datasize > 1000 {
            suggestions.push("Enable SIMD operations for better performance".to_string());
        }

        if !context.parallel_enabled && context.datasize > 10_000 {
            suggestions.push("Enable parallel processing for large datasets".to_string());
        }

        // Memory-based suggestions
        if context.memory_usage_mb > 1000.0 {
            suggestions.push("Reduce memory usage with streaming algorithms".to_string());
        }

        // General suggestions
        match _error {
            StatsError::ComputationError(_) => {
                suggestions.push("Check input data for edge cases or numerical issues".to_string());
            }
            StatsError::InvalidArgument(_) => {
                suggestions.push(
                    "Validate input parameters before calling statistical functions".to_string(),
                );
            }
            StatsError::DomainError(_) => {
                suggestions.push(
                    "Ensure input values are within valid domain for the operation".to_string(),
                );
            }
            _ => {}
        }

        suggestions
    }
}

/// Recovery strategies for different error scenarios
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    ReduceMemoryFootprint {
        chunksize: usize,
        streaming: bool,
    },
    EnableOptimizations {
        simd: bool,
        parallel: bool,
    },
    AlgorithmFallback {
        from: &'static str,
        to: &'static str,
    },
    IncreasePrecision {
        use_f64: bool,
        use_stable_algorithms: bool,
    },
}

/// Context builder for Advanced error handling
pub struct AdvancedContextBuilder {
    datasize: usize,
    operation_start: Instant,
    memory_usage_mb: f64,
    simd_enabled: bool,
    parallel_enabled: bool,
}

impl AdvancedContextBuilder {
    pub fn new(datasize: usize) -> Self {
        Self {
            datasize,
            operation_start: Instant::now(),
            memory_usage_mb: 0.0,
            simd_enabled: false,
            parallel_enabled: false,
        }
    }

    pub fn memory_usage(mut self, mb: f64) -> Self {
        self.memory_usage_mb = mb;
        self
    }

    pub fn simd_enabled(mut self, enabled: bool) -> Self {
        self.simd_enabled = enabled;
        self
    }

    pub fn parallel_enabled(mut self, enabled: bool) -> Self {
        self.parallel_enabled = enabled;
        self
    }

    pub fn build(self) -> AdvancedErrorContext {
        AdvancedErrorContext {
            operation_start: self.operation_start,
            datasize: self.datasize,
            memory_usage_mb: self.memory_usage_mb,
            simd_enabled: self.simd_enabled,
            parallel_enabled: self.parallel_enabled,
            suggested_optimization: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_context_builder() {
        let context = AdvancedContextBuilder::new(10000)
            .memory_usage(256.0)
            .simd_enabled(true)
            .parallel_enabled(false)
            .build();

        assert_eq!(context.datasize, 10000);
        assert_eq!(context.memory_usage_mb, 256.0);
        assert!(context.simd_enabled);
        assert!(!context.parallel_enabled);
    }

    #[test]
    fn test_recovery_suggestions() {
        let context = AdvancedContextBuilder::new(50000)
            .memory_usage(500.0)
            .simd_enabled(false)
            .parallel_enabled(false)
            .build();

        let error = StatsError::computation("test error");
        let suggestions = AdvancedErrorRecovery::generate_suggestions(&error, &context);

        assert!(suggestions.contains(&"Enable SIMD operations for better performance".to_string()));
        assert!(suggestions.contains(&"Enable parallel processing for large datasets".to_string()));
    }
}
