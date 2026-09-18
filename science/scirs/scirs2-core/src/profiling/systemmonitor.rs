//! System monitoring functionality

use std::collections::HashMap;
use std::time::Instant;

/// System monitor for tracking resources
pub struct SystemMonitor {
    start_time: Instant,
    metrics: HashMap<String, f64>,
}

impl SystemMonitor {
    /// Create new system monitor
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            metrics: HashMap::new(),
        }
    }

    /// Record a metric
    pub fn record(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }

    /// Get all metrics
    pub fn metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}
