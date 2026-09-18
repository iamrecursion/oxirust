//! Performance monitoring.

use std::time::Duration;

/// Performance monitor.
pub struct PerformanceMonitor {
    metrics: PerformanceMetrics,
}

/// Performance metrics.
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Current FPS
    pub fps: f64,
    /// CPU usage (%)
    pub cpu_usage: f64,
    /// GPU usage (%)
    pub gpu_usage: f64,
    /// Memory usage (MB)
    pub memory_usage: u64,
    /// Encoding latency
    pub encoding_latency: Duration,
    /// Total latency
    pub total_latency: Duration,
}

impl PerformanceMonitor {
    /// Create a new performance monitor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: PerformanceMetrics::default(),
        }
    }

    /// Update metrics.
    pub fn update(&mut self) {
        // In a real implementation, this would query system metrics
    }

    /// Get current metrics.
    #[must_use]
    pub fn metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert_eq!(monitor.metrics().fps, 0.0);
    }
}
