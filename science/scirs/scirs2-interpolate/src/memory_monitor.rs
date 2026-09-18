//! Memory leak detection and monitoring for continuous use
//!
//! This module provides utilities for tracking memory usage patterns, detecting
//! potential leaks, and monitoring memory-related performance issues during
//! long-running interpolation operations.
//!
//! # Overview
//!
//! The memory monitoring system tracks:
//! - Memory allocations and deallocations per interpolator
//! - Cache memory usage and growth patterns  
//! - Peak memory usage across operations
//! - Memory leaks through reference counting
//! - Memory pressure and allocation patterns
//!
//! # Usage
//!
//! ```rust
//! use scirs2_interpolate::memory_monitor::{MemoryMonitor, start_monitoring};
//!
//! // Start global memory monitoring
//! start_monitoring();
//!
//! // Create a monitored interpolator
//! let mut monitor = MemoryMonitor::new("rbf_interpolator");
//!
//! // Track memory during operations
//! monitor.track_allocation(1024, "distance_matrix");
//! // ... perform interpolation operations ...
//! monitor.track_deallocation(1024, "distance_matrix");
//!
//! // Check for memory leaks
//! let report = monitor.generate_report();
//! if report.has_potential_leaks() {
//!     println!("Warning: Potential memory leaks detected");
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Global memory monitoring registry
static GLOBAL_MONITOR: OnceLock<Arc<Mutex<GlobalMemoryMonitor>>> = OnceLock::new();

/// Global memory monitoring system
#[derive(Debug)]
struct GlobalMemoryMonitor {
    /// Active memory monitors by name
    monitors: HashMap<String, Arc<Mutex<MemoryMonitor>>>,

    /// Global memory statistics
    global_stats: GlobalMemoryStats,

    /// Whether monitoring is enabled
    enabled: bool,

    /// Maximum number of monitors to track
    max_monitors: usize,
}

/// Global memory statistics across all interpolators
#[derive(Debug, Clone)]
pub struct GlobalMemoryStats {
    /// Total memory allocated across all interpolators
    pub total_allocated_bytes: usize,

    /// Peak memory usage across all interpolators
    pub peak_total_bytes: usize,

    /// Number of active interpolators being monitored
    pub active_interpolators: usize,

    /// Total number of allocations tracked
    pub total_allocations: u64,

    /// Total number of deallocations tracked
    pub total_deallocations: u64,

    /// Start time of monitoring
    pub monitoring_start: Instant,
}

impl Default for GlobalMemoryStats {
    fn default() -> Self {
        Self {
            total_allocated_bytes: 0,
            peak_total_bytes: 0,
            active_interpolators: 0,
            total_allocations: 0,
            total_deallocations: 0,
            monitoring_start: Instant::now(),
        }
    }
}

/// Individual memory monitor for a specific interpolator
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Name/identifier for this monitor
    name: String,

    /// Current memory allocations by category
    allocations: HashMap<String, usize>,

    /// Memory allocation history
    allocation_history: VecDeque<AllocationEvent>,

    /// Peak memory usage for this interpolator
    peak_memory_bytes: usize,

    /// Current total memory usage
    current_memory_bytes: usize,

    /// Statistics for leak detection
    leak_stats: LeakDetectionStats,

    /// Performance metrics
    perf_metrics: MemoryPerformanceMetrics,

    /// Whether this monitor is active
    active: bool,

    /// Creation timestamp
    created_at: Instant,
}

/// Memory allocation/deallocation event
#[derive(Debug, Clone)]
struct AllocationEvent {
    /// Type of event (allocation or deallocation)
    event_type: EventType,

    /// Size in bytes
    sizebytes: usize,

    /// Category of memory (e.g., "distance_matrix", "cache", "coefficients")
    #[allow(dead_code)]
    category: String,

    /// Timestamp of event
    #[allow(dead_code)]
    timestamp: Instant,
}

/// Type of memory event
#[derive(Debug, Clone, Copy, PartialEq)]
enum EventType {
    Allocation,
    Deallocation,
}

/// Statistics for leak detection
#[derive(Debug, Clone)]
struct LeakDetectionStats {
    /// Total number of allocations
    total_allocations: u64,

    /// Total number of deallocations
    total_deallocations: u64,

    /// Number of unmatched allocations (potential leaks)
    #[allow(dead_code)]
    unmatched_allocations: u64,

    /// Memory that has been allocated but not freed for a long time
    long_lived_allocations: HashMap<String, (usize, Instant)>,

    /// Threshold for considering allocations as potential leaks (in seconds)
    leak_detection_threshold: Duration,
}

impl Default for LeakDetectionStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            unmatched_allocations: 0,
            long_lived_allocations: HashMap::new(),
            leak_detection_threshold: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Memory performance metrics
#[derive(Debug, Clone)]
struct MemoryPerformanceMetrics {
    /// Average allocation size
    avg_allocation_size: f64,

    /// Average time between allocations
    #[allow(dead_code)]
    avg_allocation_interval: Duration,

    /// Memory fragmentation estimate (0.0 to 1.0)
    #[allow(dead_code)]
    fragmentation_estimate: f64,

    /// Cache hit ratio for memory reuse
    cache_hit_ratio: f64,

    /// Last update timestamp
    last_update: Instant,
}

impl Default for MemoryPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_allocation_size: 0.0,
            avg_allocation_interval: Duration::from_millis(0),
            fragmentation_estimate: 0.0,
            cache_hit_ratio: 0.0,
            last_update: Instant::now(),
        }
    }
}

/// Memory monitoring report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Monitor name
    pub monitorname: String,

    /// Current memory usage by category
    pub current_allocations: HashMap<String, usize>,

    /// Peak memory usage
    pub peak_memory_bytes: usize,

    /// Total memory allocated over lifetime
    pub total_allocated_bytes: usize,

    /// Memory leak indicators
    pub leak_indicators: LeakIndicators,

    /// Performance metrics
    pub performance_summary: PerformanceSummary,

    /// Recommendations for memory optimization
    pub recommendations: Vec<String>,

    /// Report generation timestamp
    pub generated_at: Instant,
}

/// Memory leak indicators
#[derive(Debug, Clone)]
pub struct LeakIndicators {
    /// Potential memory leaks detected
    pub has_potential_leaks: bool,

    /// Number of unmatched allocations
    pub unmatched_allocations: u64,

    /// Memory that has been held for a long time
    pub long_lived_memory_bytes: usize,

    /// Categories with suspicious allocation patterns
    pub suspicious_categories: Vec<String>,

    /// Leak severity (0.0 = no leaks, 1.0 = severe leaks)
    pub leak_severity: f64,
}

/// Performance summary for memory usage
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Memory efficiency (lower is better)
    pub memory_efficiency_score: f64,

    /// Allocation pattern efficiency
    pub allocation_pattern_score: f64,

    /// Cache utilization score
    pub cache_utilization_score: f64,

    /// Overall memory performance grade
    pub overall_grade: PerformanceGrade,
}

/// Performance grade classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerformanceGrade {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let monitor = Self {
            name: name.clone(),
            allocations: HashMap::new(),
            allocation_history: VecDeque::new(),
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            leak_stats: LeakDetectionStats::default(),
            perf_metrics: MemoryPerformanceMetrics::default(),
            active: true,
            created_at: Instant::now(),
        };

        // Register with global monitor
        register_monitor(&name, monitor.clone());
        monitor
    }

    /// Track a memory allocation
    pub fn track_allocation(&mut self, sizebytes: usize, category: impl Into<String>) {
        if !self.active {
            return;
        }

        let category = category.into();
        let now = Instant::now();

        // Update current allocations
        *self.allocations.entry(category.clone()).or_insert(0) += sizebytes;
        self.current_memory_bytes += sizebytes;

        // Update peak usage
        if self.current_memory_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = self.current_memory_bytes;
        }

        // Record allocation event
        let event = AllocationEvent {
            event_type: EventType::Allocation,
            sizebytes,
            category: category.clone(),
            timestamp: now,
        };

        self.allocation_history.push_back(event);

        // Limit history size to prevent memory growth
        if self.allocation_history.len() > 10000 {
            self.allocation_history.pop_front();
        }

        // Update leak detection stats
        self.leak_stats.total_allocations += 1;
        self.leak_stats.long_lived_allocations.insert(
            format!("{}_{}", category, self.leak_stats.total_allocations),
            (sizebytes, now),
        );

        // Update performance metrics
        self.update_performance_metrics();

        // Update global stats
        update_global_stats(sizebytes, true);
    }

    /// Track a memory deallocation
    pub fn track_deallocation(&mut self, sizebytes: usize, category: impl Into<String>) {
        if !self.active {
            return;
        }

        let category = category.into();
        let now = Instant::now();

        // Update current allocations
        if let Some(current) = self.allocations.get_mut(&category) {
            *current = current.saturating_sub(sizebytes);
            if *current == 0 {
                self.allocations.remove(&category);
            }
        }

        self.current_memory_bytes = self.current_memory_bytes.saturating_sub(sizebytes);

        // Record deallocation event
        let event = AllocationEvent {
            event_type: EventType::Deallocation,
            sizebytes,
            category: category.clone(),
            timestamp: now,
        };

        self.allocation_history.push_back(event);

        // Update leak detection stats
        self.leak_stats.total_deallocations += 1;

        // Remove from long-lived allocations (simplified - would need better matching in production)
        self.leak_stats
            .long_lived_allocations
            .retain(|k, _| !k.starts_with(&category));

        // Update performance metrics
        self.update_performance_metrics();

        // Update global stats
        update_global_stats(sizebytes, false);
    }

    /// Generate a comprehensive memory report
    pub fn generate_report(&self) -> MemoryReport {
        let leak_indicators = self.analyze_leaks();
        let performance_summary = self.analyze_performance();
        let recommendations = self.generate_recommendations(&leak_indicators, &performance_summary);

        MemoryReport {
            monitorname: self.name.clone(),
            current_allocations: self.allocations.clone(),
            peak_memory_bytes: self.peak_memory_bytes,
            total_allocated_bytes: self.calculate_total_allocated(),
            leak_indicators,
            performance_summary,
            recommendations,
            generated_at: Instant::now(),
        }
    }

    /// Analyze potential memory leaks
    fn analyze_leaks(&self) -> LeakIndicators {
        let unmatched = self
            .leak_stats
            .total_allocations
            .saturating_sub(self.leak_stats.total_deallocations);

        // Calculate long-lived memory
        let now = Instant::now();
        let long_lived_memory: usize = self
            .leak_stats
            .long_lived_allocations
            .values()
            .filter(|(_, timestamp)| {
                now.duration_since(*timestamp) > self.leak_stats.leak_detection_threshold
            })
            .map(|(size, _)| *size)
            .sum();

        // Identify suspicious categories (categories with consistently growing memory)
        let suspicious_categories: Vec<String> = self.allocations
            .iter()
            .filter(|(_, &size)| size > 1024 * 1024) // More than 1MB
            .map(|(cat, _)| cat.clone())
            .collect();

        let has_potential_leaks =
            unmatched > 0 || long_lived_memory > 0 || !suspicious_categories.is_empty();

        // Calculate leak severity
        let leak_severity = if has_potential_leaks {
            let severity_factors = [
                (unmatched as f64) / (self.leak_stats.total_allocations as f64).max(1.0),
                (long_lived_memory as f64) / (self.peak_memory_bytes as f64).max(1.0),
                (suspicious_categories.len() as f64) / 10.0, // Normalize by 10 categories
            ];
            severity_factors.iter().sum::<f64>() / severity_factors.len() as f64
        } else {
            0.0
        };

        LeakIndicators {
            has_potential_leaks,
            unmatched_allocations: unmatched,
            long_lived_memory_bytes: long_lived_memory,
            suspicious_categories,
            leak_severity: leak_severity.min(1.0),
        }
    }

    /// Analyze memory performance
    fn analyze_performance(&self) -> PerformanceSummary {
        // Calculate memory efficiency (lower peak/current ratio is better)
        let memory_efficiency_score = if self.peak_memory_bytes > 0 {
            1.0 - (self.current_memory_bytes as f64 / self.peak_memory_bytes as f64)
        } else {
            1.0
        };

        // Calculate allocation pattern efficiency
        let allocation_pattern_score = if self.leak_stats.total_allocations > 0 {
            let deallocation_ratio = self.leak_stats.total_deallocations as f64
                / self.leak_stats.total_allocations as f64;
            deallocation_ratio.min(1.0)
        } else {
            1.0
        };

        // Use cached cache utilization score
        let cache_utilization_score = self.perf_metrics.cache_hit_ratio;

        // Calculate overall grade
        let overall_score =
            (memory_efficiency_score + allocation_pattern_score + cache_utilization_score) / 3.0;
        let overall_grade = match overall_score {
            s if s >= 0.9 => PerformanceGrade::Excellent,
            s if s >= 0.7 => PerformanceGrade::Good,
            s if s >= 0.5 => PerformanceGrade::Fair,
            s if s >= 0.3 => PerformanceGrade::Poor,
            _ => PerformanceGrade::Critical,
        };

        PerformanceSummary {
            memory_efficiency_score,
            allocation_pattern_score,
            cache_utilization_score,
            overall_grade,
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        leak_indicators: &LeakIndicators,
        performance: &PerformanceSummary,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if leak_indicators.has_potential_leaks {
            recommendations
                .push("Consider implementing explicit memory cleanup in destructor".to_string());

            if leak_indicators.unmatched_allocations > 0 {
                recommendations.push(format!(
                    "Found {} unmatched allocations - check for missing deallocations",
                    leak_indicators.unmatched_allocations
                ));
            }

            if leak_indicators.long_lived_memory_bytes > 1024 * 1024 {
                recommendations.push(format!(
                    "Large amount of long-lived memory ({} MB) - consider periodic cleanup",
                    leak_indicators.long_lived_memory_bytes / (1024 * 1024)
                ));
            }
        }

        if matches!(
            performance.overall_grade,
            PerformanceGrade::Fair | PerformanceGrade::Poor | PerformanceGrade::Critical
        ) {
            recommendations.push("Memory performance can be improved".to_string());

            if performance.memory_efficiency_score < 0.5 {
                recommendations.push(
                    "High peak memory usage - consider processing data in chunks".to_string(),
                );
            }

            if performance.cache_utilization_score < 0.3 {
                recommendations.push(
                    "Low cache utilization - enable caching for repeated operations".to_string(),
                );
            }
        }

        #[cfg(target_pointer_width = "32")]
        let high_memory_threshold = 256 * 1024 * 1024; // 256MB for 32-bit
        #[cfg(target_pointer_width = "64")]
        let high_memory_threshold = 1024 * 1024 * 1024; // 1GB for 64-bit

        if self.peak_memory_bytes > high_memory_threshold {
            recommendations.push(
                "Very high memory usage - consider using memory-efficient algorithms".to_string(),
            );
        }

        recommendations
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self) {
        let now = Instant::now();

        // Update average allocation size
        if self.leak_stats.total_allocations > 0 {
            let total_size: usize = self
                .allocation_history
                .iter()
                .filter(|e| e.event_type == EventType::Allocation)
                .map(|e| e.sizebytes)
                .sum();
            self.perf_metrics.avg_allocation_size =
                total_size as f64 / self.leak_stats.total_allocations as f64;
        }

        // Simple cache hit ratio simulation (would need actual cache statistics in practice)
        self.perf_metrics.cache_hit_ratio = 0.7; // Placeholder

        self.perf_metrics.last_update = now;
    }

    /// Calculate total memory allocated over lifetime
    fn calculate_total_allocated(&self) -> usize {
        self.allocation_history
            .iter()
            .filter(|e| e.event_type == EventType::Allocation)
            .map(|e| e.sizebytes)
            .sum()
    }

    /// Disable this monitor
    pub fn disable(&mut self) {
        self.active = false;
    }

    /// Check if monitor is active
    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl Clone for MemoryMonitor {
    fn clone(&self) -> Self {
        Self {
            name: format!("{}_clone", self.name),
            allocations: self.allocations.clone(),
            allocation_history: self.allocation_history.clone(),
            peak_memory_bytes: self.peak_memory_bytes,
            current_memory_bytes: self.current_memory_bytes,
            leak_stats: self.leak_stats.clone(),
            perf_metrics: self.perf_metrics.clone(),
            active: self.active,
            created_at: self.created_at,
        }
    }
}

impl MemoryReport {
    /// Check if the report indicates potential memory leaks
    pub fn has_potential_leaks(&self) -> bool {
        self.leak_indicators.has_potential_leaks
    }

    /// Get memory efficiency rating
    pub fn memory_efficiency_rating(&self) -> PerformanceGrade {
        self.performance_summary.overall_grade
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Memory Report for '{}': Current: {} KB, Peak: {} KB, Grade: {:?}, Leaks: {}",
            self.monitorname,
            self.current_allocations.values().sum::<usize>() / 1024,
            self.peak_memory_bytes / 1024,
            self.performance_summary.overall_grade,
            if self.has_potential_leaks() {
                "Detected"
            } else {
                "None"
            }
        )
    }
}

/// Global memory monitoring functions
/// Start global memory monitoring
#[allow(dead_code)]
pub fn start_monitoring() {
    let _ = GLOBAL_MONITOR.set(Arc::new(Mutex::new(GlobalMemoryMonitor {
        monitors: HashMap::new(),
        global_stats: GlobalMemoryStats::default(),
        enabled: true,
        max_monitors: 100,
    })));
}

/// Stop global memory monitoring
#[allow(dead_code)]
pub fn stop_monitoring() {
    if let Some(monitor) = GLOBAL_MONITOR.get() {
        if let Ok(mut global) = monitor.lock() {
            global.enabled = false;
            global.monitors.clear();
        }
    }
}

/// Register a memory monitor with the global system
#[allow(dead_code)]
fn register_monitor(name: &str, monitor: MemoryMonitor) {
    if let Some(global_monitor) = GLOBAL_MONITOR.get() {
        if let Ok(mut global) = global_monitor.lock() {
            if global.enabled && global.monitors.len() < global.max_monitors {
                global
                    .monitors
                    .insert(name.to_string(), Arc::new(Mutex::new(monitor)));
                global.global_stats.active_interpolators = global.monitors.len();
            }
        }
    }
}

/// Update global memory statistics
#[allow(dead_code)]
fn update_global_stats(sizebytes: usize, isallocation: bool) {
    if let Some(global_monitor) = GLOBAL_MONITOR.get() {
        if let Ok(mut global) = global_monitor.lock() {
            if isallocation {
                global.global_stats.total_allocated_bytes += sizebytes;
                global.global_stats.total_allocations += 1;

                if global.global_stats.total_allocated_bytes > global.global_stats.peak_total_bytes
                {
                    global.global_stats.peak_total_bytes =
                        global.global_stats.total_allocated_bytes;
                }
            } else {
                global.global_stats.total_allocated_bytes = global
                    .global_stats
                    .total_allocated_bytes
                    .saturating_sub(sizebytes);
                global.global_stats.total_deallocations += 1;
            }
        }
    }
}

/// Get global memory statistics
#[allow(dead_code)]
pub fn get_global_stats() -> Option<GlobalMemoryStats> {
    GLOBAL_MONITOR
        .get()
        .and_then(|monitor| monitor.lock().ok())
        .map(|global| global.global_stats.clone())
}

/// Get report for a specific monitor
#[allow(dead_code)]
pub fn get_monitor_report(name: &str) -> Option<MemoryReport> {
    GLOBAL_MONITOR
        .get()
        .and_then(|global_monitor| {
            global_monitor
                .lock()
                .ok()
                .and_then(|global| global.monitors.get(name).cloned())
        })
        .and_then(|monitor| monitor.lock().ok().map(|m| m.generate_report()))
}

/// Get reports for all active monitors
#[allow(dead_code)]
pub fn get_all_reports() -> Vec<MemoryReport> {
    if let Some(global_monitor) = GLOBAL_MONITOR.get() {
        if let Ok(global) = global_monitor.lock() {
            return global
                .monitors
                .values()
                .filter_map(|monitor| monitor.lock().ok())
                .map(|m| m.generate_report())
                .collect();
        }
    }
    Vec::new()
}

/// Enhanced stress testing memory profiler
#[derive(Debug)]
pub struct StressMemoryProfiler {
    /// Base monitor for standard tracking
    base_monitor: MemoryMonitor,

    /// Stress test specific metrics
    stress_metrics: StressMemoryMetrics,

    /// Memory usage history during stress tests
    stress_history: VecDeque<MemorySnapshot>,

    /// System memory pressure indicators
    pressure_indicators: MemoryPressureIndicators,

    /// Configuration for stress profiling
    stress_config: StressProfilingConfig,
}

/// Stress-specific memory metrics
#[derive(Debug, Clone)]
pub struct StressMemoryMetrics {
    /// Maximum memory growth rate during stress (bytes/second)
    pub max_growth_rate: f64,

    /// Memory allocation spikes during stress
    pub allocation_spikes: Vec<AllocationSpike>,

    /// Memory fragmentation under stress
    pub stress_fragmentation: f64,

    /// Concurrent access memory overhead
    pub concurrent_overhead: f64,

    /// Large dataset memory efficiency
    pub large_dataset_efficiency: f64,

    /// Memory recovery time after stress
    pub recovery_time_seconds: f64,
}

/// Memory allocation spike during stress testing
#[derive(Debug, Clone)]
pub struct AllocationSpike {
    /// Time when spike occurred
    pub timestamp: Instant,

    /// Size of the spike in bytes
    pub spike_size: usize,

    /// Duration of the spike
    pub duration: Duration,

    /// Stress condition that caused the spike
    pub stresscondition: String,
}

/// Memory snapshot during stress testing
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp of snapshot
    pub timestamp: Instant,

    /// Total memory usage at this point
    pub total_memory: usize,

    /// Memory usage by category
    pub category_breakdown: HashMap<String, usize>,

    /// System memory pressure level (0.0 to 1.0)
    pub system_pressure: f64,

    /// Active stress conditions
    pub active_stressconditions: Vec<String>,
}

/// System memory pressure indicators
#[derive(Debug, Clone)]
pub struct MemoryPressureIndicators {
    /// System memory utilization percentage
    pub system_memory_utilization: f64,

    /// Available memory in bytes
    pub available_memory: usize,

    /// Memory allocation failure rate
    pub allocation_failure_rate: f64,

    /// Garbage collection frequency (if applicable)
    pub gc_frequency: f64,

    /// Swap usage percentage
    pub swap_utilization: f64,
}

impl Default for MemoryPressureIndicators {
    fn default() -> Self {
        #[cfg(target_pointer_width = "32")]
        let available_memory = 512 * 1024 * 1024; // 512MB default for 32-bit
        #[cfg(target_pointer_width = "64")]
        let available_memory = 8usize * 1024 * 1024 * 1024; // 8GB default for 64-bit

        Self {
            system_memory_utilization: 0.0,
            available_memory,
            allocation_failure_rate: 0.0,
            gc_frequency: 0.0,
            swap_utilization: 0.0,
        }
    }
}

/// Configuration for stress profiling
#[derive(Debug, Clone)]
pub struct StressProfilingConfig {
    /// Sampling interval for memory snapshots during stress
    pub snapshot_interval: Duration,

    /// Maximum number of snapshots to retain
    pub max_snapshots: usize,

    /// Threshold for detecting allocation spikes (bytes)
    pub spike_threshold: usize,

    /// Enable system memory pressure monitoring
    pub monitor_system_pressure: bool,

    /// Enable detailed category tracking under stress
    pub detailed_category_tracking: bool,
}

impl Default for StressProfilingConfig {
    fn default() -> Self {
        Self {
            snapshot_interval: Duration::from_millis(100), // 10 samples per second
            max_snapshots: 10000,                          // ~17 minutes at 100ms intervals
            spike_threshold: 10 * 1024 * 1024,             // 10MB
            monitor_system_pressure: true,
            detailed_category_tracking: true,
        }
    }
}

impl StressMemoryProfiler {
    /// Create a new stress memory profiler
    pub fn new(name: impl Into<String>, config: Option<StressProfilingConfig>) -> Self {
        Self {
            base_monitor: MemoryMonitor::new(name),
            stress_metrics: StressMemoryMetrics {
                max_growth_rate: 0.0,
                allocation_spikes: Vec::new(),
                stress_fragmentation: 0.0,
                concurrent_overhead: 0.0,
                large_dataset_efficiency: 1.0,
                recovery_time_seconds: 0.0,
            },
            stress_history: VecDeque::new(),
            pressure_indicators: MemoryPressureIndicators::default(),
            stress_config: config.unwrap_or_default(),
        }
    }

    /// Start profiling under specific stress condition
    pub fn start_stress_profiling(&mut self, stresscondition: &str) {
        println!("Starting stress memory profiling for: {}", stresscondition);

        // Take initial snapshot
        self.take_memory_snapshot(vec![stresscondition.to_string()]);

        // Update system pressure indicators
        self.update_system_pressure();
    }

    /// Track memory allocation during stress test
    pub fn track_stress_allocation(
        &mut self,
        sizebytes: usize,
        category: impl Into<String>,
        stresscondition: &str,
    ) {
        let category = category.into();

        // Track with base monitor
        self.base_monitor.track_allocation(sizebytes, &category);

        // Check for allocation spike
        if sizebytes >= self.stress_config.spike_threshold {
            self.stress_metrics.allocation_spikes.push(AllocationSpike {
                timestamp: Instant::now(),
                spike_size: sizebytes,
                duration: Duration::from_millis(0), // Would measure actual duration
                stresscondition: stresscondition.to_string(),
            });
        }

        // Take periodic snapshots
        if self.should_take_snapshot() {
            self.take_memory_snapshot(vec![stresscondition.to_string()]);
        }

        // Update growth rate
        self.update_growth_rate();
    }

    /// Track memory deallocation during stress test
    pub fn track_stress_deallocation(&mut self, sizebytes: usize, category: impl Into<String>) {
        self.base_monitor.track_deallocation(sizebytes, category);

        // Update stress metrics
        self.update_growth_rate();
    }

    /// Take a memory snapshot for stress analysis
    fn take_memory_snapshot(&mut self, active_stressconditions: Vec<String>) {
        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            total_memory: self.base_monitor.current_memory_bytes,
            category_breakdown: self.base_monitor.allocations.clone(),
            system_pressure: self.calculate_system_pressure(),
            active_stressconditions,
        };

        self.stress_history.push_back(snapshot);

        // Limit history size
        if self.stress_history.len() > self.stress_config.max_snapshots {
            self.stress_history.pop_front();
        }
    }

    /// Check if should take snapshot based on timing
    fn should_take_snapshot(&self) -> bool {
        if let Some(last_snapshot) = self.stress_history.back() {
            last_snapshot.timestamp.elapsed() >= self.stress_config.snapshot_interval
        } else {
            true // Always take first snapshot
        }
    }

    /// Update memory growth rate during stress
    fn update_growth_rate(&mut self) {
        if self.stress_history.len() >= 2 {
            let recent_snapshots: Vec<_> = self.stress_history.iter().rev().take(10).collect();

            if recent_snapshots.len() >= 2 {
                let latest = recent_snapshots[0];
                let previous = recent_snapshots[recent_snapshots.len() - 1];

                let memory_delta = latest.total_memory as i64 - previous.total_memory as i64;
                let time_delta = latest
                    .timestamp
                    .duration_since(previous.timestamp)
                    .as_secs_f64();

                if time_delta > 0.0 {
                    let growth_rate = memory_delta as f64 / time_delta;
                    self.stress_metrics.max_growth_rate =
                        self.stress_metrics.max_growth_rate.max(growth_rate);
                }
            }
        }
    }

    /// Update system memory pressure indicators
    fn update_system_pressure(&mut self) {
        // In a real implementation, this would query the operating system
        // For now, simulate pressure based on our current usage

        #[cfg(target_pointer_width = "32")]
        let total_system_memory: u64 = 1024 * 1024 * 1024; // 1GB assumed for 32-bit
        #[cfg(target_pointer_width = "64")]
        let total_system_memory: u64 = 16u64 * 1024 * 1024 * 1024; // 16GB assumed for 64-bit

        let our_usage = self.base_monitor.current_memory_bytes;

        self.pressure_indicators.system_memory_utilization =
            (our_usage as f64 / total_system_memory as f64 * 100.0).min(100.0);

        self.pressure_indicators.available_memory =
            (total_system_memory as usize).saturating_sub(our_usage);

        // Simulate other metrics
        self.pressure_indicators.allocation_failure_rate =
            if self.pressure_indicators.system_memory_utilization > 90.0 {
                0.1
            } else {
                0.0
            };
    }

    /// Calculate current system pressure level
    fn calculate_system_pressure(&self) -> f64 {
        let pressure_factors = [
            self.pressure_indicators.system_memory_utilization / 100.0,
            self.pressure_indicators.allocation_failure_rate,
            self.pressure_indicators.swap_utilization / 100.0,
        ];

        pressure_factors.iter().sum::<f64>() / pressure_factors.len() as f64
    }

    /// Analyze memory efficiency under large dataset stress
    pub fn analyze_large_dataset_efficiency(
        &mut self,
        dataset_size: usize,
        expected_memory: usize,
    ) {
        let actual_memory = self.base_monitor.current_memory_bytes;

        self.stress_metrics.large_dataset_efficiency =
            expected_memory as f64 / actual_memory.max(1) as f64;

        println!(
            "Large dataset efficiency for {} elements: {:.2} (expected: {}MB, actual: {}MB)",
            dataset_size,
            self.stress_metrics.large_dataset_efficiency,
            expected_memory / (1024 * 1024),
            actual_memory / (1024 * 1024)
        );
    }

    /// Analyze concurrent access memory overhead
    pub fn analyze_concurrent_overhead(
        &mut self,
        baseline_memory: usize,
        concurrent_threads: usize,
    ) {
        let current_memory = self.base_monitor.current_memory_bytes;
        let overhead = current_memory.saturating_sub(baseline_memory);

        self.stress_metrics.concurrent_overhead = overhead as f64 / concurrent_threads as f64;

        println!(
            "Concurrent access overhead: {:.1}KB per thread ({} threads)",
            self.stress_metrics.concurrent_overhead / 1024.0,
            concurrent_threads
        );
    }

    /// Measure memory recovery time after stress
    pub fn measure_recovery_time(&mut self, stress_endtime: Instant) {
        let _recovery_start_memory = self.base_monitor.current_memory_bytes;

        // Monitor memory for recovery (simplified - would need async monitoring in practice)
        let recovery_time = Instant::now().duration_since(stress_endtime);
        self.stress_metrics.recovery_time_seconds = recovery_time.as_secs_f64();

        println!(
            "Memory recovery _time: {:.2}s",
            self.stress_metrics.recovery_time_seconds
        );
    }

    /// Generate comprehensive stress memory report
    pub fn generate_stress_report(&self) -> StressMemoryReport {
        let base_report = self.base_monitor.generate_report();

        let memory_pressure_analysis = self.analyze_memory_pressure();
        let allocation_pattern_analysis = self.analyze_allocation_patterns();
        let stress_performance_analysis = self.analyze_stress_performance();

        StressMemoryReport {
            base_report,
            stress_metrics: self.stress_metrics.clone(),
            memory_pressure_analysis,
            allocation_pattern_analysis,
            stress_performance_analysis,
            system_pressure: self.pressure_indicators.clone(),
            snapshot_count: self.stress_history.len(),
            stress_recommendations: self.generate_stress_recommendations(),
        }
    }

    /// Analyze memory pressure patterns
    fn analyze_memory_pressure(&self) -> MemoryPressureAnalysis {
        let max_pressure = self
            .stress_history
            .iter()
            .map(|s| s.system_pressure)
            .fold(0.0, f64::max);

        let avg_pressure = if !self.stress_history.is_empty() {
            self.stress_history
                .iter()
                .map(|s| s.system_pressure)
                .sum::<f64>()
                / self.stress_history.len() as f64
        } else {
            0.0
        };

        let pressure_spikes = self
            .stress_history
            .iter()
            .filter(|s| s.system_pressure > 0.8)
            .count();

        MemoryPressureAnalysis {
            max_pressure,
            avg_pressure,
            pressure_spikes,
            critical_periods: pressure_spikes, // Simplified
        }
    }

    /// Analyze allocation patterns under stress
    fn analyze_allocation_patterns(&self) -> AllocationPatternAnalysis {
        let spike_count = self.stress_metrics.allocation_spikes.len();
        let total_spike_memory: usize = self
            .stress_metrics
            .allocation_spikes
            .iter()
            .map(|s| s.spike_size)
            .sum();

        let pattern_regularity = if spike_count > 1 {
            // Calculate variance in spike timing
            let intervals: Vec<_> = self
                .stress_metrics
                .allocation_spikes
                .windows(2)
                .map(|pair| {
                    pair[1]
                        .timestamp
                        .duration_since(pair[0].timestamp)
                        .as_secs_f64()
                })
                .collect();

            if !intervals.is_empty() {
                let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
                let variance = intervals
                    .iter()
                    .map(|&x| (x - mean_interval).powi(2))
                    .sum::<f64>()
                    / intervals.len() as f64;
                1.0 / (1.0 + variance) // Higher variance = lower regularity
            } else {
                1.0
            }
        } else {
            1.0
        };

        AllocationPatternAnalysis {
            spike_count,
            total_spike_memory,
            pattern_regularity,
            fragmentation_level: self.stress_metrics.stress_fragmentation,
        }
    }

    /// Analyze stress performance
    fn analyze_stress_performance(&self) -> StressPerformanceAnalysis {
        StressPerformanceAnalysis {
            max_growth_rate: self.stress_metrics.max_growth_rate,
            concurrent_overhead: self.stress_metrics.concurrent_overhead,
            large_dataset_efficiency: self.stress_metrics.large_dataset_efficiency,
            recovery_time: self.stress_metrics.recovery_time_seconds,
            overall_stress_grade: self.calculate_stress_grade(),
        }
    }

    /// Calculate overall stress performance grade
    fn calculate_stress_grade(&self) -> StressPerformanceGrade {
        let factors = [
            if self.stress_metrics.max_growth_rate < 1024.0 * 1024.0 {
                1.0
            } else {
                0.0
            }, // < 1MB/s growth
            if self.stress_metrics.concurrent_overhead < 1024.0 * 1024.0 {
                1.0
            } else {
                0.0
            }, // < 1MB overhead per thread
            self.stress_metrics.large_dataset_efficiency.min(1.0), // Efficiency ratio
            if self.stress_metrics.recovery_time_seconds < 10.0 {
                1.0
            } else {
                0.0
            }, // < 10s recovery
        ];

        let score = factors.iter().sum::<f64>() / factors.len() as f64;

        match score {
            s if s >= 0.9 => StressPerformanceGrade::Excellent,
            s if s >= 0.7 => StressPerformanceGrade::Good,
            s if s >= 0.5 => StressPerformanceGrade::Fair,
            s if s >= 0.3 => StressPerformanceGrade::Poor,
            _ => StressPerformanceGrade::Critical,
        }
    }

    /// Generate stress-specific recommendations
    fn generate_stress_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.stress_metrics.max_growth_rate > 10.0 * 1024.0 * 1024.0 {
            // > 10MB/s
            recommendations
                .push("High memory growth rate detected - consider batch processing".to_string());
        }

        if self.stress_metrics.allocation_spikes.len() > 10 {
            recommendations
                .push("Frequent allocation spikes - implement memory pre-allocation".to_string());
        }

        if self.stress_metrics.concurrent_overhead > 5.0 * 1024.0 * 1024.0 {
            // > 5MB per thread
            recommendations
                .push("High concurrent overhead - review thread-local memory usage".to_string());
        }

        if self.stress_metrics.large_dataset_efficiency < 0.7 {
            recommendations
                .push("Poor large dataset efficiency - optimize memory layout".to_string());
        }

        if self.stress_metrics.recovery_time_seconds > 30.0 {
            recommendations.push("Slow memory recovery - implement explicit cleanup".to_string());
        }

        recommendations
    }
}

/// Comprehensive stress memory report
#[derive(Debug, Clone)]
pub struct StressMemoryReport {
    /// Base memory report
    pub base_report: MemoryReport,

    /// Stress-specific metrics
    pub stress_metrics: StressMemoryMetrics,

    /// Memory pressure analysis
    pub memory_pressure_analysis: MemoryPressureAnalysis,

    /// Allocation pattern analysis
    pub allocation_pattern_analysis: AllocationPatternAnalysis,

    /// Stress performance analysis
    pub stress_performance_analysis: StressPerformanceAnalysis,

    /// System pressure indicators
    pub system_pressure: MemoryPressureIndicators,

    /// Number of snapshots taken
    pub snapshot_count: usize,

    /// Stress-specific recommendations
    pub stress_recommendations: Vec<String>,
}

/// Memory pressure analysis results
#[derive(Debug, Clone)]
pub struct MemoryPressureAnalysis {
    /// Maximum pressure level reached (0.0 to 1.0)
    pub max_pressure: f64,

    /// Average pressure level
    pub avg_pressure: f64,

    /// Number of pressure spikes
    pub pressure_spikes: usize,

    /// Number of critical pressure periods
    pub critical_periods: usize,
}

/// Allocation pattern analysis results
#[derive(Debug, Clone)]
pub struct AllocationPatternAnalysis {
    /// Number of allocation spikes
    pub spike_count: usize,

    /// Total memory in spikes
    pub total_spike_memory: usize,

    /// Pattern regularity (0.0 to 1.0)
    pub pattern_regularity: f64,

    /// Memory fragmentation level
    pub fragmentation_level: f64,
}

/// Stress performance analysis
#[derive(Debug, Clone)]
pub struct StressPerformanceAnalysis {
    /// Maximum memory growth rate (bytes/second)
    pub max_growth_rate: f64,

    /// Concurrent access overhead per thread
    pub concurrent_overhead: f64,

    /// Large dataset memory efficiency
    pub large_dataset_efficiency: f64,

    /// Recovery time after stress
    pub recovery_time: f64,

    /// Overall stress performance grade
    pub overall_stress_grade: StressPerformanceGrade,
}

/// Stress performance grades
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StressPerformanceGrade {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Create a stress memory profiler for testing
#[allow(dead_code)]
pub fn create_stress_profiler(name: impl Into<String>) -> StressMemoryProfiler {
    StressMemoryProfiler::new(name, None)
}

/// Create a stress memory profiler with custom configuration
#[allow(dead_code)]
pub fn create_stress_profiler_with_config(
    name: impl Into<String>,
    config: StressProfilingConfig,
) -> StressMemoryProfiler {
    StressMemoryProfiler::new(name, Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_monitor_basic() {
        let mut monitor = MemoryMonitor::new("test");

        // Track some allocations
        monitor.track_allocation(1024, "matrix");
        monitor.track_allocation(512, "cache");

        assert_eq!(monitor.current_memory_bytes, 1536);
        assert_eq!(monitor.peak_memory_bytes, 1536);

        // Track deallocations - deallocate all to avoid false leak detection
        monitor.track_deallocation(512, "cache");
        assert_eq!(monitor.current_memory_bytes, 1024);

        monitor.track_deallocation(1024, "matrix");
        assert_eq!(monitor.current_memory_bytes, 0);

        let report = monitor.generate_report();
        // After deallocating all memory, should have no leaks
        assert!(!report.has_potential_leaks());
    }

    #[test]
    fn test_leak_detection() {
        let mut monitor = MemoryMonitor::new("leak_test");

        // Allocate without deallocating (potential leak)
        monitor.track_allocation(2048, "leaked_memory");

        let report = monitor.generate_report();
        assert!(report.leak_indicators.unmatched_allocations > 0);
    }

    #[test]
    fn test_global_monitoring() {
        start_monitoring();

        let _monitor1 = MemoryMonitor::new("global_test_1");
        let _monitor2 = MemoryMonitor::new("global_test_2");

        let stats = get_global_stats().expect("Operation failed");
        // Test may run in parallel with other tests, so check >= 2 instead of == 2
        assert!(
            stats.active_interpolators >= 2,
            "Expected at least 2 active interpolators, got {}",
            stats.active_interpolators
        );

        stop_monitoring();
    }

    #[test]
    fn test_stress_profiler_basic() {
        let mut profiler = create_stress_profiler("stress_test");

        profiler.start_stress_profiling("large_dataset");
        profiler.track_stress_allocation(10 * 1024 * 1024, "large_matrix", "large_dataset");
        profiler.track_stress_allocation(5 * 1024 * 1024, "cache", "large_dataset");

        let report = profiler.generate_stress_report();
        assert!(report.stress_metrics.allocation_spikes.len() > 0);
        assert!(report.snapshot_count > 0);
    }

    #[test]
    fn test_stress_allocation_spike_detection() {
        let mut profiler = create_stress_profiler("spike_test");

        // Trigger allocation spike (default threshold is 10MB)
        profiler.track_stress_allocation(15 * 1024 * 1024, "spike", "stress_test");

        assert_eq!(profiler.stress_metrics.allocation_spikes.len(), 1);
        assert_eq!(
            profiler.stress_metrics.allocation_spikes[0].spike_size,
            15 * 1024 * 1024
        );
    }
}
