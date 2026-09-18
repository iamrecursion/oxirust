//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::memory_efficient::adaptive_feedback::SharedPredictor;
use crate::memory_efficient::chunked::ChunkingStrategy;
use crate::memory_efficient::platform_memory::PlatformMemoryInfo;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::functions::SharedMemoryMonitor;

/// Memory limits configuration for OOM prevention.
///
/// This struct defines the memory constraints that the adaptive chunking
/// system will respect to prevent out-of-memory errors.
#[derive(Debug, Clone)]
pub struct MemoryLimits {
    /// Maximum memory that can be used for chunk processing (in bytes)
    pub max_memory_usage: usize,
    /// Memory threshold at which to start reducing chunk sizes (0.0 to 1.0)
    pub pressure_threshold: f64,
    /// Critical memory threshold - emergency chunk size reduction (0.0 to 1.0)
    pub critical_threshold: f64,
    /// Minimum memory to reserve for system operations (in bytes)
    pub reserved_memory: usize,
    /// Whether to enable automatic memory pressure monitoring
    pub auto_monitor: bool,
}
impl MemoryLimits {
    /// Create memory limits with auto-detection based on system capabilities.
    pub fn auto_detect() -> Self {
        let (max_memory, available) = PlatformMemoryInfo::detect()
            .map(|info| (info.total_memory, info.available_memory))
            .unwrap_or((
                (4u64 * 1024 * 1024 * 1024) as usize,
                (2u64 * 1024 * 1024 * 1024) as usize,
            ));
        let max_usage = available / 2;
        let reserved = max_memory / 10;
        Self {
            max_memory_usage: max_usage,
            pressure_threshold: 0.75,
            critical_threshold: 0.90,
            reserved_memory: reserved,
            auto_monitor: true,
        }
    }
    /// Create memory limits with explicit values.
    pub const fn with_limits(max_memory: usize, reserved: usize) -> Self {
        Self {
            max_memory_usage: max_memory,
            pressure_threshold: 0.75,
            critical_threshold: 0.90,
            reserved_memory: reserved,
            auto_monitor: true,
        }
    }
    /// Create conservative memory limits for low-memory environments.
    pub fn conservative() -> Self {
        let mut limits = Self::auto_detect();
        limits.max_memory_usage /= 2;
        limits.pressure_threshold = 0.60;
        limits.critical_threshold = 0.80;
        limits
    }
    /// Create aggressive memory limits for high-performance scenarios.
    pub fn aggressive() -> Self {
        let mut limits = Self::auto_detect();
        limits.max_memory_usage = limits.max_memory_usage.saturating_mul(3).saturating_div(2);
        limits.pressure_threshold = 0.85;
        limits.critical_threshold = 0.95;
        limits
    }
}
/// Real-time memory pressure monitor for adaptive chunking.
///
/// This monitor tracks system memory usage and provides feedback
/// for dynamic chunk size adjustment.
#[derive(Debug)]
pub struct MemoryPressureMonitor {
    /// Memory limits configuration
    pub(super) limits: MemoryLimits,
    /// Current memory pressure level (atomic for thread safety)
    pub(super) current_level: AtomicUsize,
    /// Whether monitoring is active
    pub(super) active: AtomicBool,
    /// Last check timestamp
    pub(super) last_check: std::sync::Mutex<Instant>,
    /// Minimum interval between checks
    pub(super) check_interval: Duration,
    /// Historical pressure levels for trend analysis
    pub(super) pressure_history: std::sync::Mutex<Vec<(Instant, MemoryPressureLevel)>>,
    /// Maximum history entries to keep
    pub(super) max_history: usize,
}
impl MemoryPressureMonitor {
    /// Create a new memory pressure monitor with the given limits.
    pub fn new(limits: MemoryLimits) -> Self {
        Self {
            limits,
            current_level: AtomicUsize::new(0),
            active: AtomicBool::new(true),
            last_check: std::sync::Mutex::new(Instant::now()),
            check_interval: Duration::from_millis(100),
            pressure_history: std::sync::Mutex::new(Vec::with_capacity(100)),
            max_history: 100,
        }
    }
    /// Create a monitor with default limits.
    pub fn with_defaults() -> Self {
        Self::new(MemoryLimits::default())
    }
    /// Check current memory pressure and update state.
    pub fn check_pressure(&self) -> MemoryPressureLevel {
        {
            let mut last = self.last_check.lock().unwrap_or_else(|e| e.into_inner());
            if last.elapsed() < self.check_interval {
                return self.get_current_level();
            }
            *last = Instant::now();
        }
        if !self.active.load(Ordering::Relaxed) {
            return MemoryPressureLevel::Normal;
        }
        let pressure_ratio = PlatformMemoryInfo::detect()
            .map(|info| {
                if info.total_memory == 0 {
                    0.0
                } else {
                    1.0 - (info.available_memory as f64 / info.total_memory as f64)
                }
            })
            .unwrap_or(0.5);
        let level = if pressure_ratio >= self.limits.critical_threshold {
            MemoryPressureLevel::Critical
        } else if pressure_ratio >= self.limits.pressure_threshold {
            MemoryPressureLevel::High
        } else if pressure_ratio >= self.limits.pressure_threshold * 0.8 {
            MemoryPressureLevel::Elevated
        } else {
            MemoryPressureLevel::Normal
        };
        self.current_level.store(level as usize, Ordering::Relaxed);
        if let Ok(mut history) = self.pressure_history.lock() {
            history.push((Instant::now(), level));
            while history.len() > self.max_history {
                history.remove(0);
            }
        }
        level
    }
    /// Get the current pressure level without checking.
    pub fn get_current_level(&self) -> MemoryPressureLevel {
        match self.current_level.load(Ordering::Relaxed) {
            0 => MemoryPressureLevel::Normal,
            1 => MemoryPressureLevel::Elevated,
            2 => MemoryPressureLevel::High,
            _ => MemoryPressureLevel::Critical,
        }
    }
    /// Calculate the recommended chunk size based on current pressure.
    pub fn recommended_chunk_size(&self, base_size: usize) -> usize {
        let level = self.check_pressure();
        let factor = level.reduction_factor();
        let reduced = (base_size as f64 * factor) as usize;
        reduced.max(1024)
    }
    /// Check if processing should pause due to critical pressure.
    pub fn should_pause(&self) -> bool {
        self.check_pressure() == MemoryPressureLevel::Critical
    }
    /// Get the trend of memory pressure (increasing, stable, decreasing).
    pub fn get_trend(&self) -> MemoryTrend {
        let history = match self.pressure_history.lock() {
            Ok(h) => h,
            Err(e) => e.into_inner(),
        };
        if history.len() < 3 {
            return MemoryTrend::Stable;
        }
        let recent: Vec<_> = history.iter().rev().take(3).collect();
        let older: Vec<_> = history.iter().rev().skip(3).take(3).collect();
        if older.is_empty() {
            return MemoryTrend::Stable;
        }
        let recent_avg: f64 = recent
            .iter()
            .map(|(_, level)| *level as usize as f64)
            .sum::<f64>()
            / recent.len() as f64;
        let older_avg: f64 = older
            .iter()
            .map(|(_, level)| *level as usize as f64)
            .sum::<f64>()
            / older.len() as f64;
        let diff = recent_avg - older_avg;
        if diff > 0.5 {
            MemoryTrend::Increasing
        } else if diff < -0.5 {
            MemoryTrend::Decreasing
        } else {
            MemoryTrend::Stable
        }
    }
    /// Pause or resume monitoring.
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
    /// Get the configured memory limits.
    pub fn limits(&self) -> &MemoryLimits {
        &self.limits
    }
}
/// Statistics from chunk processing for adaptive feedback.
#[derive(Debug, Clone, Default)]
pub struct ChunkProcessingStats {
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Total processing time
    pub total_time: Duration,
    /// Average chunk processing time
    pub avg_chunk_time: Duration,
    /// Peak memory usage observed (estimated)
    pub peak_memory_estimate: usize,
    /// Number of chunk size adjustments made
    pub adjustments_made: usize,
    /// Final chunk size used
    pub final_chunk_size: usize,
    /// Memory pressure events encountered
    pub pressure_events: usize,
}
/// Memory pressure level for adaptive decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressureLevel {
    /// Normal operation - no pressure
    Normal,
    /// Elevated pressure - consider reducing chunk sizes
    Elevated,
    /// High pressure - actively reduce chunk sizes
    High,
    /// Critical pressure - emergency measures needed
    Critical,
}
impl MemoryPressureLevel {
    /// Get the chunk size reduction factor for this pressure level.
    pub const fn reduction_factor(&self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::Elevated => 0.75,
            Self::High => 0.5,
            Self::Critical => 0.25,
        }
    }
}
/// Memory pressure trend for predictive adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTrend {
    /// Memory pressure is increasing
    Increasing,
    /// Memory pressure is stable
    Stable,
    /// Memory pressure is decreasing
    Decreasing,
}
/// Builder for creating adaptive chunking parameters with a fluent API.
#[derive(Debug, Clone)]
pub struct AdaptiveChunkingBuilder {
    params: AdaptiveChunkingParams,
}
impl AdaptiveChunkingBuilder {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            params: AdaptiveChunkingParams::default(),
        }
    }
    /// Set the target memory usage per chunk.
    pub const fn with_target_memory(mut self, bytes: usize) -> Self {
        self.params.target_memory_usage = bytes;
        self
    }
    /// Set the maximum chunk size.
    pub const fn with_max_chunksize(mut self, size: usize) -> Self {
        self.params.max_chunksize = size;
        self
    }
    /// Set the minimum chunk size.
    pub const fn with_min_chunksize(mut self, size: usize) -> Self {
        self.params.min_chunksize = size;
        self
    }
    /// Set the target chunk processing duration.
    pub fn with_target_duration(mut self, duration: Duration) -> Self {
        self.params.target_chunk_duration = Some(duration);
        self
    }
    /// Enable consideration of data distribution.
    pub const fn consider_distribution(mut self, enable: bool) -> Self {
        self.params.consider_distribution = enable;
        self
    }
    /// Enable optimization for parallel processing.
    pub const fn optimize_for_parallel(mut self, enable: bool) -> Self {
        self.params.optimize_for_parallel = enable;
        self
    }
    /// Set the number of worker threads to optimize for.
    pub fn with_numworkers(mut self, workers: usize) -> Self {
        self.params.numworkers = Some(workers);
        self
    }
    /// Set memory limits for OOM prevention.
    pub fn with_memory_limits(mut self, limits: MemoryLimits) -> Self {
        self.params.memory_limits = Some(limits);
        self
    }
    /// Enable or disable OOM prevention.
    pub const fn with_oom_prevention(mut self, enable: bool) -> Self {
        self.params.enable_oom_prevention = enable;
        self
    }
    /// Enable or disable dynamic chunk size adjustment.
    pub const fn with_dynamic_adjustment(mut self, enable: bool) -> Self {
        self.params.enable_dynamic_adjustment = enable;
        self
    }
    /// Set a shared memory pressure monitor.
    pub fn with_memory_monitor(mut self, monitor: SharedMemoryMonitor) -> Self {
        self.params.memory_monitor = Some(monitor);
        self
    }
    /// Configure for memory-constrained environments.
    pub fn memory_constrained(mut self) -> Self {
        self.params.memory_limits = Some(MemoryLimits::conservative());
        self.params.enable_oom_prevention = true;
        self.params.enable_dynamic_adjustment = true;
        self.params.target_memory_usage /= 2;
        self
    }
    /// Configure for high-performance environments.
    pub fn high_performance(mut self) -> Self {
        self.params.memory_limits = Some(MemoryLimits::aggressive());
        self.params.enable_oom_prevention = true;
        self.params.enable_dynamic_adjustment = true;
        self.params.target_memory_usage = self.params.target_memory_usage.saturating_mul(2);
        self
    }
    /// Configure for workload type with appropriate memory settings.
    pub fn for_workload(mut self, workload: WorkloadType) -> Self {
        let workload_params = AdaptiveChunkingParams::for_workload(workload);
        self.params.target_memory_usage = workload_params.target_memory_usage;
        self.params.min_chunksize = workload_params.min_chunksize;
        self.params.target_chunk_duration = workload_params.target_chunk_duration;
        self.params.optimize_for_parallel = workload_params.optimize_for_parallel;
        self.params.consider_distribution = workload_params.consider_distribution;
        self.params.memory_limits = Some(match workload {
            WorkloadType::MemoryIntensive => MemoryLimits::conservative(),
            WorkloadType::ComputeIntensive | WorkloadType::IoIntensive => {
                MemoryLimits::aggressive()
            }
            WorkloadType::Balanced => MemoryLimits::auto_detect(),
        });
        self
    }
    /// Build the parameters.
    pub fn build(self) -> AdaptiveChunkingParams {
        self.params
    }
}
/// Workload types for optimized chunking strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// Memory-intensive workloads that need smaller chunks
    MemoryIntensive,
    /// Compute-intensive workloads that can benefit from larger chunks
    ComputeIntensive,
    /// I/O-intensive workloads that need optimized for throughput
    IoIntensive,
    /// Balanced workloads with mixed requirements
    Balanced,
}
/// Result of adaptive chunking analysis.
#[derive(Debug, Clone)]
pub struct AdaptiveChunkingResult {
    /// Recommended chunking strategy
    pub strategy: ChunkingStrategy,
    /// Estimated memory usage per chunk (in bytes)
    pub estimated_memory_per_chunk: usize,
    /// Factors that influenced the chunking decision
    pub decision_factors: Vec<String>,
}
/// Dynamic chunk size adjuster that adapts during processing.
#[derive(Debug)]
pub struct DynamicChunkAdjuster {
    /// Current chunk size
    pub(super) current_size: AtomicUsize,
    /// Initial chunk size
    pub(super) initial_size: usize,
    /// Minimum chunk size
    pub(super) min_size: usize,
    /// Maximum chunk size
    pub(super) max_size: usize,
    /// Memory monitor (optional)
    pub(super) monitor: Option<SharedMemoryMonitor>,
    /// Processing times for recent chunks
    pub(super) chunk_times: std::sync::Mutex<Vec<Duration>>,
    /// Target processing time per chunk
    pub(super) target_time: Duration,
    /// Number of adjustments made
    pub(super) adjustments: AtomicUsize,
    /// Whether dynamic adjustment is enabled
    pub(super) enabled: AtomicBool,
}
impl DynamicChunkAdjuster {
    /// Create a new dynamic chunk adjuster.
    pub fn new(initial_size: usize, min_size: usize, max_size: usize) -> Self {
        Self {
            current_size: AtomicUsize::new(initial_size),
            initial_size,
            min_size,
            max_size,
            monitor: None,
            chunk_times: std::sync::Mutex::new(Vec::with_capacity(20)),
            target_time: Duration::from_millis(100),
            adjustments: AtomicUsize::new(0),
            enabled: AtomicBool::new(true),
        }
    }
    /// Create with memory monitoring.
    pub fn with_monitor(mut self, monitor: SharedMemoryMonitor) -> Self {
        self.monitor = Some(monitor);
        self
    }
    /// Set the target processing time per chunk.
    pub fn with_target_time(mut self, target: Duration) -> Self {
        self.target_time = target;
        self
    }
    /// Get the current recommended chunk size.
    pub fn get_chunk_size(&self) -> usize {
        if !self.enabled.load(Ordering::Relaxed) {
            return self.initial_size;
        }
        let mut size = self.current_size.load(Ordering::Relaxed);
        if let Some(ref monitor) = self.monitor {
            size = monitor.recommended_chunk_size(size);
        }
        size.clamp(self.min_size, self.max_size)
    }
    /// Record the processing time for a chunk and adjust if needed.
    pub fn record_chunk_time(&self, time: Duration) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        if let Ok(mut times) = self.chunk_times.lock() {
            times.push(time);
            while times.len() > 20 {
                times.remove(0);
            }
            if times.len() >= 5 {
                self.adjust_based_on_times(&times);
            }
        }
    }
    /// Adjust chunk size based on recorded times.
    fn adjust_based_on_times(&self, times: &[Duration]) {
        let avg_time: Duration = times.iter().sum::<Duration>() / times.len() as u32;
        let current = self.current_size.load(Ordering::Relaxed);
        let new_size = if avg_time > self.target_time.mul_f64(1.5) {
            (current as f64 * 0.75) as usize
        } else if avg_time < self.target_time.mul_f64(0.5) {
            (current as f64 * 1.25) as usize
        } else {
            return;
        };
        let clamped = new_size.clamp(self.min_size, self.max_size);
        if clamped != current {
            self.current_size.store(clamped, Ordering::Relaxed);
            self.adjustments.fetch_add(1, Ordering::Relaxed);
        }
    }
    /// Force an immediate chunk size reduction (for emergency situations).
    pub fn emergency_reduce(&self) {
        let current = self.current_size.load(Ordering::Relaxed);
        let reduced = (current / 2).max(self.min_size);
        self.current_size.store(reduced, Ordering::Relaxed);
        self.adjustments.fetch_add(1, Ordering::Relaxed);
    }
    /// Reset to initial settings.
    pub fn reset(&self) {
        self.current_size
            .store(self.initial_size, Ordering::Relaxed);
        if let Ok(mut times) = self.chunk_times.lock() {
            times.clear();
        }
    }
    /// Get the number of adjustments made.
    pub fn adjustment_count(&self) -> usize {
        self.adjustments.load(Ordering::Relaxed)
    }
    /// Enable or disable dynamic adjustment.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
}
/// Parameters for configuring adaptive chunking behavior.
#[derive(Clone)]
pub struct AdaptiveChunkingParams {
    /// Target memory usage per chunk (in bytes)
    pub target_memory_usage: usize,
    /// Maximum chunk size (in elements)
    pub max_chunksize: usize,
    /// Minimum chunk size (in elements)
    pub min_chunksize: usize,
    /// Target processing time per chunk (for time-based adaptation)
    pub target_chunk_duration: Option<Duration>,
    /// Whether to consider data distribution (can be expensive to calculate)
    pub consider_distribution: bool,
    /// Whether to adjust for parallel processing
    pub optimize_for_parallel: bool,
    /// Number of worker threads to optimize for (when parallel is enabled)
    pub numworkers: Option<usize>,
    /// Optional chunk size predictor for adaptive feedback
    pub predictor: Option<SharedPredictor>,
    /// Memory limits configuration for OOM prevention
    pub memory_limits: Option<MemoryLimits>,
    /// Whether to enable OOM prevention mechanisms
    pub enable_oom_prevention: bool,
    /// Whether to enable dynamic chunk size adjustment during processing
    pub enable_dynamic_adjustment: bool,
    /// Shared memory pressure monitor (optional, for multi-threaded use)
    pub memory_monitor: Option<SharedMemoryMonitor>,
}
impl AdaptiveChunkingParams {
    /// Beta 2: Detect available system memory using cross-platform detection
    pub fn detect_available_memory() -> Option<usize> {
        use crate::memory_efficient::platform_memory::PlatformMemoryInfo;
        PlatformMemoryInfo::detect().map(|info| info.available_memory)
    }
    /// Beta 2: Create optimized parameters for specific workload types
    pub fn for_workload(workload: WorkloadType) -> Self {
        let mut params = Self::default();
        match workload {
            WorkloadType::MemoryIntensive => {
                params.target_memory_usage /= 2;
                params.consider_distribution = false;
            }
            WorkloadType::ComputeIntensive => {
                params.target_chunk_duration = Some(Duration::from_millis(500));
                params.optimize_for_parallel = true;
            }
            WorkloadType::IoIntensive => {
                params.target_memory_usage *= 2;
                params.min_chunksize = 64 * 1024;
            }
            WorkloadType::Balanced => {}
        }
        params
    }
}
