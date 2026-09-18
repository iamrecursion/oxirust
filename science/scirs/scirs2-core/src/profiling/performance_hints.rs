//! # Function-Level Performance Hinting System
//!
//! This module provides a comprehensive performance hinting system that allows functions
//! to declare their performance characteristics and optimization preferences.

use crate::error::{CoreError, CoreResult, ErrorContext};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Performance characteristics of a function
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceHints {
    /// Expected computational complexity (e.g., O(n), O(n²), etc.)
    pub complexity: ComplexityClass,
    /// Whether the function benefits from SIMD optimization
    pub simd_friendly: bool,
    /// Whether the function can be parallelized
    pub parallelizable: bool,
    /// Whether the function benefits from GPU acceleration
    pub gpu_friendly: bool,
    /// Expected memory usage pattern
    pub memory_pattern: MemoryPattern,
    /// Cache behavior characteristics
    pub cache_behavior: CacheBehavior,
    /// I/O characteristics
    pub io_pattern: IoPattern,
    /// Preferred optimization level
    pub optimization_level: OptimizationLevel,
    /// Function-specific optimization hints
    pub custom_hints: HashMap<String, String>,
    /// Expected execution time range
    pub expected_duration: Option<DurationRange>,
    /// Memory requirements
    pub memory_requirements: Option<MemoryRequirements>,
}

/// Computational complexity classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplexityClass {
    /// O(1) - Constant time
    Constant,
    /// O(log n) - Logarithmic time
    Logarithmic,
    /// O(n) - Linear time
    Linear,
    /// O(n log n) - Linearithmic time
    Linearithmic,
    /// O(n²) - Quadratic time
    Quadratic,
    /// O(n³) - Cubic time
    Cubic,
    /// O(2^n) - Exponential time
    Exponential,
    /// O(n!) - Factorial time
    Factorial,
    /// Custom complexity description
    Custom(String),
}

/// Memory access pattern classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryPattern {
    /// Sequential memory access
    Sequential,
    /// Random memory access
    Random,
    /// Strided memory access
    Strided { stride: usize },
    /// Block-based memory access
    Blocked { block_size: usize },
    /// Cache-oblivious access pattern
    CacheOblivious,
    /// Mixed access pattern
    Mixed,
}

/// Cache behavior characteristics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheBehavior {
    /// Cache-friendly access pattern
    CacheFriendly,
    /// Cache-unfriendly access pattern
    CacheUnfriendly,
    /// Temporal locality (reuses data soon)
    TemporalLocality,
    /// Spatial locality (accesses nearby data)
    SpatialLocality,
    /// Mixed cache behavior
    Mixed,
    /// Unknown cache behavior
    Unknown,
}

/// I/O operation characteristics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoPattern {
    /// No I/O operations
    None,
    /// Read-only operations
    ReadOnly,
    /// Write-only operations
    WriteOnly,
    /// Read-write operations
    ReadWrite,
    /// Network I/O
    Network,
    /// Disk I/O
    Disk,
    /// Memory-mapped I/O
    MemoryMapped,
}

/// Optimization level preferences
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization (debug builds)
    None,
    /// Basic optimization
    Basic,
    /// Aggressive optimization
    Aggressive,
    /// Profile-guided optimization
    ProfileGuided,
    /// Custom optimization settings
    Custom(String),
}

/// Expected duration range
#[derive(Debug, Clone, PartialEq)]
pub struct DurationRange {
    pub min: Duration,
    pub max: Duration,
    pub typical: Duration,
}

/// Memory requirements specification
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRequirements {
    /// Minimum memory required
    pub min_memory: usize,
    /// Maximum memory that could be used
    pub max_memory: Option<usize>,
    /// Typical memory usage
    pub typical_memory: usize,
    /// Whether memory usage scales with input size
    pub scales_with_input: bool,
}

impl Default for PerformanceHints {
    fn default() -> Self {
        Self {
            complexity: ComplexityClass::Linear,
            simd_friendly: false,
            parallelizable: false,
            gpu_friendly: false,
            memory_pattern: MemoryPattern::Sequential,
            cache_behavior: CacheBehavior::Unknown,
            io_pattern: IoPattern::None,
            optimization_level: OptimizationLevel::Basic,
            custom_hints: HashMap::new(),
            expected_duration: None,
            memory_requirements: None,
        }
    }
}

impl PerformanceHints {
    /// Create a new set of performance hints
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the computational complexity
    pub fn with_complexity(mut self, complexity: ComplexityClass) -> Self {
        self.complexity = complexity;
        self
    }

    /// Mark as SIMD-friendly
    pub fn simd_friendly(mut self) -> Self {
        self.simd_friendly = true;
        self
    }

    /// Mark as parallelizable
    pub fn parallelizable(mut self) -> Self {
        self.parallelizable = true;
        self
    }

    /// Mark as GPU-friendly
    pub fn gpu_friendly(mut self) -> Self {
        self.gpu_friendly = true;
        self
    }

    /// Set memory access pattern
    pub fn with_memory_pattern(mut self, pattern: MemoryPattern) -> Self {
        self.memory_pattern = pattern;
        self
    }

    /// Set cache behavior
    pub fn with_cache_behavior(mut self, behavior: CacheBehavior) -> Self {
        self.cache_behavior = behavior;
        self
    }

    /// Set I/O pattern
    pub fn with_io_pattern(mut self, pattern: IoPattern) -> Self {
        self.io_pattern = pattern;
        self
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Add a custom hint
    pub fn with_custom_hint<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.custom_hints.insert(key.into(), value.into());
        self
    }

    /// Set expected duration range
    pub fn with_expected_duration(mut self, range: DurationRange) -> Self {
        self.expected_duration = Some(range);
        self
    }

    /// Set memory requirements
    pub fn with_memory_requirements(mut self, requirements: MemoryRequirements) -> Self {
        self.memory_requirements = Some(requirements);
        self
    }

    /// Get a specific custom hint
    pub fn get_custom_hint(&self, key: &str) -> Option<&String> {
        self.custom_hints.get(key)
    }

    /// Check if the function should use SIMD optimization
    pub fn should_use_simd(&self) -> bool {
        self.simd_friendly
            && matches!(
                self.optimization_level,
                OptimizationLevel::Aggressive | OptimizationLevel::ProfileGuided
            )
    }

    /// Check if the function should be parallelized
    pub fn should_parallelize(&self) -> bool {
        self.parallelizable && !matches!(self.optimization_level, OptimizationLevel::None)
    }

    /// Check if the function should use GPU acceleration
    pub fn should_use_gpu(&self) -> bool {
        self.gpu_friendly
            && matches!(
                self.optimization_level,
                OptimizationLevel::Aggressive | OptimizationLevel::ProfileGuided
            )
    }

    /// Estimate if the operation is suitable for chunking
    pub fn should_chunk(&self, inputsize: usize) -> bool {
        match self.complexity {
            ComplexityClass::Quadratic
            | ComplexityClass::Cubic
            | ComplexityClass::Exponential
            | ComplexityClass::Factorial => true,
            ComplexityClass::Linear | ComplexityClass::Linearithmic => inputsize > 10000,
            ComplexityClass::Constant | ComplexityClass::Logarithmic => false,
            ComplexityClass::Custom(_) => false,
        }
    }
}

/// Performance hint registry for functions
#[derive(Debug)]
pub struct PerformanceHintRegistry {
    hints: RwLock<HashMap<String, PerformanceHints>>,
    execution_stats: RwLock<HashMap<String, ExecutionStats>>,
}

/// Execution statistics for a function
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub total_calls: u64,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub last_updated: Instant,
}

impl Default for ExecutionStats {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            total_calls: 0,
            total_duration: Duration::ZERO,
            average_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            last_updated: now,
        }
    }
}

impl ExecutionStats {
    /// Update statistics with a new execution time
    pub fn update(&mut self, duration: Duration) {
        self.total_calls += 1;
        self.total_duration += duration;
        self.average_duration = self.total_duration / self.total_calls as u32;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.last_updated = Instant::now();
    }

    /// Check if the actual performance matches the hints
    pub fn matches_expected(&self, expected: &DurationRange) -> bool {
        self.average_duration >= expected.min && self.average_duration <= expected.max
    }
}

impl PerformanceHintRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            hints: RwLock::new(HashMap::new()),
            execution_stats: RwLock::new(HashMap::new()),
        }
    }

    /// Register performance hints for a function
    pub fn register(&self, functionname: &str, hints: PerformanceHints) -> CoreResult<()> {
        let mut hint_map = self.hints.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;
        hint_map.insert(functionname.to_string(), hints);
        Ok(())
    }

    /// Get performance hints for a function
    pub fn get_hint(&self, functionname: &str) -> CoreResult<Option<PerformanceHints>> {
        let hint_map = self.hints.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire read lock"))
        })?;
        Ok(hint_map.get(functionname).cloned())
    }

    /// Record execution statistics
    pub fn record_execution(&self, functionname: &str, duration: Duration) -> CoreResult<()> {
        let mut stats_map = self.execution_stats.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;

        let stats = stats_map.entry(functionname.to_string()).or_default();
        stats.update(std::time::Duration::from_secs(1));
        Ok(())
    }

    /// Get execution statistics for a function
    pub fn get_stats(&self, functionname: &str) -> CoreResult<Option<ExecutionStats>> {
        let stats_map = self.execution_stats.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire read lock"))
        })?;
        Ok(stats_map.get(functionname).cloned())
    }

    /// Get optimization recommendations based on hints and statistics
    pub fn get_optimization_recommendations(
        &self,
        function_name: &str,
    ) -> CoreResult<Vec<OptimizationRecommendation>> {
        let hints = self.get_hint(function_name)?;
        let stats = self.get_stats(function_name)?;

        let mut recommendations = Vec::new();

        if let Some(hints) = hints {
            // SIMD recommendations
            if hints.simd_friendly && !hints.should_use_simd() {
                recommendations.push(OptimizationRecommendation::EnableSIMD);
            }

            // Parallelization recommendations
            if hints.parallelizable && !hints.should_parallelize() {
                recommendations.push(OptimizationRecommendation::EnableParallelization);
            }

            // GPU recommendations
            if hints.gpu_friendly && !hints.should_use_gpu() {
                recommendations.push(OptimizationRecommendation::EnableGPU);
            }

            // Memory optimization recommendations
            match hints.memory_pattern {
                MemoryPattern::Random => {
                    recommendations.push(OptimizationRecommendation::OptimizeMemoryLayout);
                }
                MemoryPattern::Strided { .. } => {
                    recommendations.push(OptimizationRecommendation::UseVectorization);
                }
                _ => {}
            }

            // Duration-based recommendations
            if let (Some(expected), Some(stats)) =
                (hints.expected_duration.as_ref(), stats.as_ref())
            {
                if !stats.matches_expected(expected) && stats.average_duration > expected.max {
                    recommendations.push(OptimizationRecommendation::ProfileAndOptimize);
                }
            }
        }

        Ok(recommendations)
    }

    /// Clear all recorded statistics
    pub fn clear_stats(&self) -> CoreResult<()> {
        let mut stats_map = self.execution_stats.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;
        stats_map.clear();
        Ok(())
    }
}

impl Default for PerformanceHintRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization recommendations based on performance analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationRecommendation {
    /// Enable SIMD optimization
    EnableSIMD,
    /// Enable parallelization
    EnableParallelization,
    /// Enable GPU acceleration
    EnableGPU,
    /// Optimize memory layout
    OptimizeMemoryLayout,
    /// Use vectorization
    UseVectorization,
    /// Profile the function for bottlenecks
    ProfileAndOptimize,
    /// Use chunking for large inputs
    UseChunking,
    /// Cache intermediate results
    CacheResults,
    /// Custom recommendation
    Custom(String),
}

impl std::fmt::Display for OptimizationRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationRecommendation::EnableSIMD => write!(f, "Enable SIMD optimization"),
            OptimizationRecommendation::EnableParallelization => {
                write!(f, "Enable parallelization")
            }
            OptimizationRecommendation::EnableGPU => write!(f, "Enable GPU acceleration"),
            OptimizationRecommendation::OptimizeMemoryLayout => write!(f, "Optimize memory layout"),
            OptimizationRecommendation::UseVectorization => write!(f, "Use vectorization"),
            OptimizationRecommendation::ProfileAndOptimize => write!(f, "Profile and optimize"),
            OptimizationRecommendation::UseChunking => write!(f, "Use chunking for large inputs"),
            OptimizationRecommendation::CacheResults => write!(f, "Cache intermediate results"),
            OptimizationRecommendation::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

/// Global performance hint registry
static GLOBAL_REGISTRY: Lazy<PerformanceHintRegistry> = Lazy::new(PerformanceHintRegistry::new);

/// Get the global performance hint registry
#[allow(dead_code)]
pub fn global_registry() -> &'static PerformanceHintRegistry {
    &GLOBAL_REGISTRY
}

/// Macro to register performance hints for a function
#[macro_export]
macro_rules! register_performance_hints {
    ($function_name:expr, $hints:expr) => {
        $crate::profiling::performance_hints::global_registry()
            .register($function_name, $hints)
            .unwrap_or_else(|e| eprintln!("Failed to register performance hints: {:?}", e));
    };
}

/// Macro to create and register performance hints in one step
#[macro_export]
macro_rules! performance_hints {
    ($function_name:expr, {
        $(complexity: $complexity:expr,)?
        $(simdfriendly: $simd:expr,)?
        $(parallelizable: $parallel:expr,)?
        $(gpufriendly: $gpu:expr,)?
        $(memorypattern: $memory:expr,)?
        $(cachebehavior: $cache:expr,)?
        $(iopattern: $io:expr,)?
        $(optimizationlevel: $opt:expr,)?
        $(expectedduration: $duration:expr,)?
        $(memoryrequirements: $mem:expr,)?
        $(customhints: {$($key:expr => $value:expr),*$(,)?})?
    }) => {
        {
            let mut hints = $crate::profiling::performance_hints::PerformanceHints::new();

            $(hints = hints.with_complexity($complexity);)?
            $(if $simd { hints = hints.simd_friendly(); })?
            $(if $parallel { hints = hints.parallelizable(); })?
            $(if $gpu { hints = hints.gpu_friendly(); })?
            $(hints = hints.with_memory_pattern($memory);)?
            $(hints = hints.with_cache_behavior($cache);)?
            $(hints = hints.with_io_pattern($io);)?
            $(hints = hints.with_optimization_level($opt_level);)?
            $(hints = hints.with_expected_duration($std::time::Duration::from_secs(1));)?
            $(hints = hints.with_memory_requirements($mem_req);)?
            $($(hints = hints.with_custom_hint($key, $value);)*)?

            $crate::profiling::performance_hints::global_registry()
                .register($function_name, hints)
                .unwrap_or_else(|e| eprintln!("Failed to register performance hints: {:?}", e));
        }
    };
}

/// Function decorator for automatic performance tracking
pub struct PerformanceTracker {
    function_name: String,
    start_time: Instant,
}

impl PerformanceTracker {
    /// Start tracking performance for a function
    pub fn new(functionname: &str) -> Self {
        Self {
            function_name: functionname.to_string(),
            start_time: Instant::now(),
        }
    }

    /// Finish tracking and record the execution time
    pub fn finish(self) {
        let elapsed = self.start_time.elapsed();
        let _ = global_registry().record_execution(&self.function_name, elapsed);
    }
}

/// Macro to automatically track function performance
#[macro_export]
macro_rules! track_performance {
    ($function_name:expr, $code:block) => {{
        let tracker =
            $crate::profiling::performance_hints::PerformanceTracker::start($function_name);
        let result = $code;
        tracker.finish();
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_hints_creation() {
        let hints = PerformanceHints::new()
            .with_complexity(ComplexityClass::Quadratic)
            .simd_friendly()
            .parallelizable()
            .with_memory_pattern(MemoryPattern::Sequential)
            .with_cache_behavior(CacheBehavior::CacheFriendly);

        assert_eq!(hints.complexity, ComplexityClass::Quadratic);
        assert!(hints.simd_friendly);
        assert!(hints.parallelizable);
        assert_eq!(hints.memory_pattern, MemoryPattern::Sequential);
        assert_eq!(hints.cache_behavior, CacheBehavior::CacheFriendly);
    }

    #[test]
    fn test_registry_operations() {
        let registry = PerformanceHintRegistry::new();

        let hints = PerformanceHints::new()
            .with_complexity(ComplexityClass::Linear)
            .simd_friendly();

        // Register hints
        assert!(registry.register("test_function", hints.clone()).is_ok());

        // Retrieve hints
        let retrieved = registry
            .get_hint("test_function")
            .expect("Operation failed");
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.expect("Operation failed").complexity,
            ComplexityClass::Linear
        );

        // Record execution
        assert!(registry
            .record_execution("test_function", Duration::from_millis(100))
            .is_ok());

        // Get stats
        let stats = registry
            .get_stats("test_function")
            .expect("Operation failed");
        assert!(stats.is_some());
        assert_eq!(stats.expect("Operation failed").total_calls, 1);
    }

    #[test]
    fn test_optimization_recommendations() {
        let registry = PerformanceHintRegistry::new();

        let hints = PerformanceHints::new()
            .with_complexity(ComplexityClass::Quadratic)
            .simd_friendly()
            .parallelizable()
            .gpu_friendly()
            .with_memory_pattern(MemoryPattern::Random);

        registry
            .register("test_function", hints)
            .expect("Operation failed");

        let recommendations = registry
            .get_optimization_recommendations("test_function")
            .expect("Operation failed");
        assert!(!recommendations.is_empty());

        // Should recommend enabling optimizations since hints indicate suitability
        assert!(recommendations.contains(&OptimizationRecommendation::OptimizeMemoryLayout));
    }

    #[test]
    fn test_performance_tracker() {
        let tracker = PerformanceTracker::new("test_tracker");
        thread::sleep(Duration::from_millis(10));
        tracker.finish();

        let stats = global_registry()
            .get_stats("test_tracker")
            .expect("Operation failed");
        assert!(stats.is_some());
        let stats = stats.expect("Operation failed");
        assert_eq!(stats.total_calls, 1);
        assert!(stats.average_duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_execution_stats_update() {
        let mut stats = ExecutionStats::default();

        stats.update(Duration::from_millis(100));
        assert_eq!(stats.total_calls, 1);
        assert_eq!(stats.average_duration, Duration::from_millis(100));
        assert_eq!(stats.min_duration, Duration::from_millis(100));
        assert_eq!(stats.max_duration, Duration::from_millis(100));

        stats.update(Duration::from_millis(200));
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.average_duration, Duration::from_millis(150));
        assert_eq!(stats.min_duration, Duration::from_millis(100));
        assert_eq!(stats.max_duration, Duration::from_millis(200));
    }

    #[test]
    fn test_should_use_chunking() {
        let hints = PerformanceHints::new().with_complexity(ComplexityClass::Quadratic);

        assert!(hints.should_chunk(10000));

        let linear_hints = PerformanceHints::new().with_complexity(ComplexityClass::Linear);

        assert!(linear_hints.should_chunk(20000));
        assert!(!linear_hints.should_chunk(1000));
    }
}
