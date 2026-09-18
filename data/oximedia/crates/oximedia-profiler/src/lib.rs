//! Performance profiling and optimization tools for OxiMedia.
//!
//! This crate provides comprehensive profiling capabilities including:
//! - CPU profiling with sampling and instrumentation
//! - Memory allocation tracking and leak detection
//! - GPU profiling and timeline analysis
//! - Frame timing and budget analysis
//! - Bottleneck detection and classification
//! - Resource tracking (files, network, etc.)
//! - Cache analysis and miss profiling
//! - Thread utilization and contention detection
//! - Flame graph generation
//! - Automated benchmarking and regression detection
//! - Optimization suggestions
//!
//! # Examples
//!
//! ```
//! use oximedia_profiler::{Profiler, ProfilingMode};
//!
//! let mut profiler = Profiler::new(ProfilingMode::Sampling);
//! profiler.start();
//!
//! // Your code here
//!
//! profiler.stop();
//! let report = profiler.generate_report();
//! println!("{}", report);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod allocation_tracker;
pub mod benchmark;
pub mod bottleneck;
pub mod bottleneck_analysis;
pub mod cache;
pub mod call_graph;
pub mod chrome_trace;
pub mod chrome_tracing;
pub mod codec_profiler;
pub mod cpu;
pub mod distributed_profile;
pub mod event_ring_buffer;
pub mod event_trace;
pub mod flame;
pub mod flame_graph;
pub mod flamegraph;
pub mod frame;
pub mod frame_budget;
pub mod frame_profiler;
pub mod gpu;
pub mod hierarchical_span;
pub mod hotspot;
pub mod latency_profiler;
pub mod mem_profile;
pub mod mem_stage_profiler;
pub mod memory;
pub mod memory_profiler;
pub mod network_profiler;
pub mod optimize;
pub mod perf_script;
pub mod pipeline_bottleneck;
pub mod pipeline_profiler;
pub mod power_energy;
pub mod profile_compare;
pub mod regression;
pub mod report;
pub mod report_format;
pub mod resource;
pub mod sampling_profiler;
pub mod session;
pub mod span;
pub mod span_bottleneck;
pub mod thread;
pub mod throughput_profiler;
pub mod tls_counters;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Profiler error types.
#[derive(Error, Debug)]
pub enum ProfilerError {
    /// Profiler is already running.
    #[error("Profiler is already running")]
    AlreadyRunning,

    /// Profiler is not running.
    #[error("Profiler is not running")]
    NotRunning,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

/// Result type for profiler operations.
pub type Result<T> = std::result::Result<T, ProfilerError>;

/// Profiling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfilingMode {
    /// Low overhead statistical sampling.
    Sampling,

    /// Detailed instrumentation (higher overhead).
    Instrumentation,

    /// Event-based profiling.
    EventBased,

    /// Continuous light profiling.
    Continuous,
}

/// Profiler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Profiling mode.
    pub mode: ProfilingMode,

    /// Sampling rate in Hz (for sampling mode).
    pub sample_rate: u32,

    /// Enable CPU profiling.
    pub cpu_profiling: bool,

    /// Enable memory profiling.
    pub memory_profiling: bool,

    /// Enable GPU profiling.
    pub gpu_profiling: bool,

    /// Enable frame timing.
    pub frame_timing: bool,

    /// Enable resource tracking.
    pub resource_tracking: bool,

    /// Enable cache analysis.
    pub cache_analysis: bool,

    /// Enable thread analysis.
    pub thread_analysis: bool,

    /// Maximum overhead percentage (0.0-100.0).
    pub max_overhead: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            mode: ProfilingMode::Sampling,
            sample_rate: 100,
            cpu_profiling: true,
            memory_profiling: true,
            gpu_profiling: false,
            frame_timing: false,
            resource_tracking: true,
            cache_analysis: false,
            thread_analysis: true,
            max_overhead: 1.0,
        }
    }
}

/// Main profiler structure.
pub struct Profiler {
    config: ProfilerConfig,
    running: bool,
    start_time: Option<Instant>,
    cpu_profiler: Option<cpu::profile::CpuProfiler>,
    memory_tracker: Option<memory::track::MemoryTracker>,
    metrics: HashMap<String, ProfileMetric>,
}

/// Profile metric value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfileMetric {
    /// Duration metric.
    Duration(Duration),

    /// Count metric.
    Count(u64),

    /// Percentage metric.
    Percentage(f64),

    /// Bytes metric.
    Bytes(u64),

    /// Custom metric.
    Custom(String),
}

impl Profiler {
    /// Create a new profiler with the given mode.
    pub fn new(mode: ProfilingMode) -> Self {
        let config = ProfilerConfig {
            mode,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new profiler with custom configuration.
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            config,
            running: false,
            start_time: None,
            cpu_profiler: None,
            memory_tracker: None,
            metrics: HashMap::new(),
        }
    }

    /// Start profiling.
    pub fn start(&mut self) -> Result<()> {
        if self.running {
            return Err(ProfilerError::AlreadyRunning);
        }

        self.start_time = Some(Instant::now());
        self.running = true;

        if self.config.cpu_profiling {
            let mut cpu_profiler = cpu::profile::CpuProfiler::new(self.config.sample_rate);
            cpu_profiler.start()?;
            self.cpu_profiler = Some(cpu_profiler);
        }

        if self.config.memory_profiling {
            let mut memory_tracker = memory::track::MemoryTracker::new();
            memory_tracker.start();
            self.memory_tracker = Some(memory_tracker);
        }

        Ok(())
    }

    /// Stop profiling.
    pub fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Err(ProfilerError::NotRunning);
        }

        if let Some(ref mut cpu_profiler) = self.cpu_profiler {
            cpu_profiler.stop()?;
        }

        if let Some(ref mut memory_tracker) = self.memory_tracker {
            memory_tracker.stop();
        }

        self.running = false;
        Ok(())
    }

    /// Check if profiler is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get elapsed time since profiling started.
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Record a custom metric.
    pub fn record_metric(&mut self, name: String, metric: ProfileMetric) {
        self.metrics.insert(name, metric);
    }

    /// Get a recorded metric.
    pub fn get_metric(&self, name: &str) -> Option<&ProfileMetric> {
        self.metrics.get(name)
    }

    /// Generate a profiling report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== OxiMedia Profiler Report ===\n\n");

        if let Some(elapsed) = self.elapsed() {
            report.push_str(&format!("Total Duration: {:?}\n", elapsed));
        }

        report.push_str(&format!("Mode: {:?}\n", self.config.mode));
        report.push_str(&format!("Running: {}\n\n", self.running));

        if !self.metrics.is_empty() {
            report.push_str("Custom Metrics:\n");
            for (name, metric) in &self.metrics {
                report.push_str(&format!("  {}: {:?}\n", name, metric));
            }
            report.push('\n');
        }

        if let Some(ref cpu_profiler) = self.cpu_profiler {
            report.push_str("CPU Profiling:\n");
            report.push_str(&cpu_profiler.summary());
            report.push('\n');
        }

        if let Some(ref memory_tracker) = self.memory_tracker {
            report.push_str("Memory Tracking:\n");
            report.push_str(&memory_tracker.summary());
            report.push('\n');
        }

        report
    }

    /// Get the profiler configuration.
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new(ProfilingMode::Sampling);
        assert!(!profiler.is_running());
        assert_eq!(profiler.config().mode, ProfilingMode::Sampling);
    }

    #[test]
    fn test_profiler_start_stop() {
        let mut profiler = Profiler::new(ProfilingMode::Sampling);
        assert!(profiler.start().is_ok());
        assert!(profiler.is_running());
        assert!(profiler.start().is_err()); // Already running
        assert!(profiler.stop().is_ok());
        assert!(!profiler.is_running());
        assert!(profiler.stop().is_err()); // Not running
    }

    #[test]
    fn test_profiler_metrics() {
        let mut profiler = Profiler::new(ProfilingMode::Sampling);
        profiler.record_metric("test".to_string(), ProfileMetric::Count(42));
        assert!(profiler.get_metric("test").is_some());
        assert!(profiler.get_metric("nonexistent").is_none());
    }

    #[test]
    fn test_profiler_elapsed() {
        let mut profiler = Profiler::new(ProfilingMode::Sampling);
        assert!(profiler.elapsed().is_none());
        profiler.start().expect("should succeed in test");
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = profiler.elapsed().expect("should succeed in test");
        assert!(elapsed >= Duration::from_millis(10));
        profiler.stop().expect("should succeed in test");
    }

    #[test]
    fn test_profiler_config() {
        let config = ProfilerConfig {
            mode: ProfilingMode::Instrumentation,
            sample_rate: 200,
            cpu_profiling: true,
            memory_profiling: false,
            gpu_profiling: true,
            frame_timing: true,
            resource_tracking: false,
            cache_analysis: true,
            thread_analysis: false,
            max_overhead: 2.0,
        };

        let profiler = Profiler::with_config(config.clone());
        assert_eq!(profiler.config().mode, ProfilingMode::Instrumentation);
        assert_eq!(profiler.config().sample_rate, 200);
        assert!(profiler.config().cpu_profiling);
        assert!(!profiler.config().memory_profiling);
    }

    #[test]
    fn test_profiler_report() {
        let mut profiler = Profiler::new(ProfilingMode::Sampling);
        profiler.start().expect("should succeed in test");
        std::thread::sleep(Duration::from_millis(10));
        profiler.stop().expect("should succeed in test");
        let report = profiler.generate_report();
        assert!(report.contains("OxiMedia Profiler Report"));
        assert!(report.contains("Mode: Sampling"));
    }

    #[test]
    fn test_profile_metric_types() {
        let duration = ProfileMetric::Duration(Duration::from_secs(1));
        let count = ProfileMetric::Count(100);
        let percentage = ProfileMetric::Percentage(75.5);
        let bytes = ProfileMetric::Bytes(1024);
        let custom = ProfileMetric::Custom("test".to_string());

        assert!(matches!(duration, ProfileMetric::Duration(_)));
        assert!(matches!(count, ProfileMetric::Count(_)));
        assert!(matches!(percentage, ProfileMetric::Percentage(_)));
        assert!(matches!(bytes, ProfileMetric::Bytes(_)));
        assert!(matches!(custom, ProfileMetric::Custom(_)));
    }
}
