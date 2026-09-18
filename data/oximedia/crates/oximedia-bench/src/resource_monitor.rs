#![allow(dead_code)]
//! System resource monitoring during benchmarks.
//!
//! Samples CPU usage, memory consumption, and other metrics at regular
//! intervals while a benchmark is running. This data helps explain
//! performance anomalies (e.g. memory pressure, thermal throttling).

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single resource snapshot taken at a point in time.
#[derive(Debug, Clone)]
pub struct ResourceSample {
    /// Offset from the start of monitoring.
    pub offset: Duration,
    /// Estimated CPU usage fraction (0..1).
    pub cpu_usage: f64,
    /// Resident memory in bytes.
    pub memory_bytes: u64,
    /// Optional disk I/O bytes read since last sample.
    pub disk_read_bytes: u64,
    /// Optional disk I/O bytes written since last sample.
    pub disk_write_bytes: u64,
}

/// Aggregated statistics over all collected samples.
#[derive(Debug, Clone)]
pub struct ResourceStats {
    /// Number of samples collected.
    pub sample_count: usize,
    /// Mean CPU usage.
    pub mean_cpu: f64,
    /// Peak CPU usage.
    pub peak_cpu: f64,
    /// Mean memory usage in bytes.
    pub mean_memory_bytes: u64,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Total disk read bytes.
    pub total_disk_read: u64,
    /// Total disk write bytes.
    pub total_disk_write: u64,
    /// Total monitoring duration.
    pub duration: Duration,
}

/// Configuration for the resource monitor.
#[derive(Debug, Clone)]
pub struct ResourceMonitorConfig {
    /// Interval between samples.
    pub sample_interval: Duration,
    /// Maximum number of samples to keep (ring buffer).
    pub max_samples: usize,
    /// Whether to track disk I/O.
    pub track_disk_io: bool,
}

impl Default for ResourceMonitorConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_millis(100),
            max_samples: 10_000,
            track_disk_io: false,
        }
    }
}

impl ResourceMonitorConfig {
    /// Create a new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sample interval.
    pub fn with_sample_interval(mut self, d: Duration) -> Self {
        self.sample_interval = d;
        self
    }

    /// Set the maximum number of samples.
    pub fn with_max_samples(mut self, n: usize) -> Self {
        self.max_samples = n;
        self
    }

    /// Enable or disable disk I/O tracking.
    pub fn with_disk_io(mut self, enabled: bool) -> Self {
        self.track_disk_io = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

/// Collects resource samples during a benchmark.
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// Configuration.
    config: ResourceMonitorConfig,
    /// Collected samples.
    samples: Vec<ResourceSample>,
    /// Start instant.
    start: Option<Instant>,
}

impl ResourceMonitor {
    /// Create a new monitor.
    pub fn new(config: ResourceMonitorConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            start: None,
        }
    }

    /// Start monitoring (records the start time).
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
        self.samples.clear();
    }

    /// Take a snapshot of current system resources.
    ///
    /// The `cpu_usage` and `memory_bytes` values are caller-provided because
    /// reading them portably requires platform-specific code. The monitor
    /// simply records them with a timestamp.
    pub fn sample(&mut self, cpu_usage: f64, memory_bytes: u64) {
        self.sample_with_io(cpu_usage, memory_bytes, 0, 0);
    }

    /// Take a snapshot including disk I/O counters.
    pub fn sample_with_io(
        &mut self,
        cpu_usage: f64,
        memory_bytes: u64,
        disk_read_bytes: u64,
        disk_write_bytes: u64,
    ) {
        let offset = self.start.map(|s| s.elapsed()).unwrap_or(Duration::ZERO);
        let sample = ResourceSample {
            offset,
            cpu_usage: cpu_usage.clamp(0.0, 1.0),
            memory_bytes,
            disk_read_bytes,
            disk_write_bytes,
        };
        if self.samples.len() >= self.config.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    /// Stop monitoring and compute aggregated statistics.
    #[allow(clippy::cast_precision_loss)]
    pub fn stop(&self) -> ResourceStats {
        if self.samples.is_empty() {
            return ResourceStats {
                sample_count: 0,
                mean_cpu: 0.0,
                peak_cpu: 0.0,
                mean_memory_bytes: 0,
                peak_memory_bytes: 0,
                total_disk_read: 0,
                total_disk_write: 0,
                duration: Duration::ZERO,
            };
        }

        let n = self.samples.len();
        let sum_cpu: f64 = self.samples.iter().map(|s| s.cpu_usage).sum();
        let peak_cpu = self
            .samples
            .iter()
            .map(|s| s.cpu_usage)
            .fold(0.0_f64, f64::max);

        let sum_mem: u64 = self.samples.iter().map(|s| s.memory_bytes).sum();
        let peak_mem = self
            .samples
            .iter()
            .map(|s| s.memory_bytes)
            .max()
            .unwrap_or(0);

        let total_disk_read: u64 = self.samples.iter().map(|s| s.disk_read_bytes).sum();
        let total_disk_write: u64 = self.samples.iter().map(|s| s.disk_write_bytes).sum();

        let duration = self
            .samples
            .last()
            .map(|s| s.offset)
            .unwrap_or(Duration::ZERO);

        ResourceStats {
            sample_count: n,
            mean_cpu: sum_cpu / n as f64,
            peak_cpu,
            mean_memory_bytes: sum_mem / n as u64,
            peak_memory_bytes: peak_mem,
            total_disk_read,
            total_disk_write,
            duration,
        }
    }

    /// Get all collected samples.
    pub fn samples(&self) -> &[ResourceSample] {
        &self.samples
    }

    /// Number of collected samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Whether the monitor is currently running.
    pub fn is_running(&self) -> bool {
        self.start.is_some()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ResourceMonitorConfig::default();
        assert_eq!(cfg.sample_interval, Duration::from_millis(100));
        assert_eq!(cfg.max_samples, 10_000);
        assert!(!cfg.track_disk_io);
    }

    #[test]
    fn test_config_builder() {
        let cfg = ResourceMonitorConfig::new()
            .with_sample_interval(Duration::from_millis(50))
            .with_max_samples(500)
            .with_disk_io(true);
        assert_eq!(cfg.sample_interval, Duration::from_millis(50));
        assert_eq!(cfg.max_samples, 500);
        assert!(cfg.track_disk_io);
    }

    #[test]
    fn test_empty_monitor() {
        let mon = ResourceMonitor::new(ResourceMonitorConfig::default());
        let stats = mon.stop();
        assert_eq!(stats.sample_count, 0);
        assert!((stats.mean_cpu - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_basic_sampling() {
        let mut mon = ResourceMonitor::new(ResourceMonitorConfig::default());
        mon.start();
        mon.sample(0.5, 1_000_000);
        mon.sample(0.7, 2_000_000);
        assert_eq!(mon.sample_count(), 2);
    }

    #[test]
    fn test_stop_computes_stats() {
        let mut mon = ResourceMonitor::new(ResourceMonitorConfig::default());
        mon.start();
        mon.sample(0.4, 100);
        mon.sample(0.6, 200);
        let stats = mon.stop();
        assert_eq!(stats.sample_count, 2);
        assert!((stats.mean_cpu - 0.5).abs() < f64::EPSILON);
        assert!((stats.peak_cpu - 0.6).abs() < f64::EPSILON);
        assert_eq!(stats.peak_memory_bytes, 200);
    }

    #[test]
    fn test_cpu_clamped() {
        let mut mon = ResourceMonitor::new(ResourceMonitorConfig::default());
        mon.start();
        mon.sample(1.5, 0);
        assert!((mon.samples()[0].cpu_usage - 1.0).abs() < f64::EPSILON);
        mon.sample(-0.5, 0);
        assert!((mon.samples()[1].cpu_usage - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_samples_ring() {
        let cfg = ResourceMonitorConfig::new().with_max_samples(3);
        let mut mon = ResourceMonitor::new(cfg);
        mon.start();
        for i in 0..5 {
            mon.sample(0.1 * i as f64, 0);
        }
        assert_eq!(mon.sample_count(), 3);
    }

    #[test]
    fn test_disk_io_tracking() {
        let mut mon = ResourceMonitor::new(ResourceMonitorConfig::default());
        mon.start();
        mon.sample_with_io(0.5, 0, 1024, 512);
        mon.sample_with_io(0.5, 0, 2048, 1024);
        let stats = mon.stop();
        assert_eq!(stats.total_disk_read, 3072);
        assert_eq!(stats.total_disk_write, 1536);
    }

    #[test]
    fn test_is_running() {
        let mut mon = ResourceMonitor::new(ResourceMonitorConfig::default());
        assert!(!mon.is_running());
        mon.start();
        assert!(mon.is_running());
    }

    #[test]
    fn test_mean_memory() {
        let mut mon = ResourceMonitor::new(ResourceMonitorConfig::default());
        mon.start();
        mon.sample(0.0, 100);
        mon.sample(0.0, 300);
        let stats = mon.stop();
        assert_eq!(stats.mean_memory_bytes, 200);
    }

    #[test]
    fn test_resource_sample_clone() {
        let s = ResourceSample {
            offset: Duration::from_secs(1),
            cpu_usage: 0.5,
            memory_bytes: 1024,
            disk_read_bytes: 0,
            disk_write_bytes: 0,
        };
        let c = s.clone();
        assert_eq!(c.offset, Duration::from_secs(1));
        assert!((c.cpu_usage - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_stats_fields() {
        let stats = ResourceStats {
            sample_count: 10,
            mean_cpu: 0.45,
            peak_cpu: 0.9,
            mean_memory_bytes: 2048,
            peak_memory_bytes: 4096,
            total_disk_read: 100,
            total_disk_write: 200,
            duration: Duration::from_secs(5),
        };
        assert_eq!(stats.sample_count, 10);
        assert!((stats.peak_cpu - 0.9).abs() < f64::EPSILON);
        assert_eq!(stats.peak_memory_bytes, 4096);
    }
}
