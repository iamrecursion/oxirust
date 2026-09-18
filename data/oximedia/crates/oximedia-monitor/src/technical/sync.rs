//! Sync pulse monitoring.

use serde::{Deserialize, Serialize};

/// Sync monitoring metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncMetrics {
    /// Sync present.
    pub sync_present: bool,

    /// Sync errors detected.
    pub sync_errors: u64,

    /// Last sync timestamp.
    pub last_sync_time: u64,
}

/// Sync monitor.
pub struct SyncMonitor {
    metrics: SyncMetrics,
}

impl SyncMonitor {
    /// Create a new sync monitor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: SyncMetrics::default(),
        }
    }

    /// Update sync monitoring.
    pub fn update(&mut self) {
        self.metrics.sync_present = true;
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &SyncMetrics {
        &self.metrics
    }

    /// Reset monitor.
    pub fn reset(&mut self) {
        self.metrics = SyncMetrics::default();
    }
}

impl Default for SyncMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_monitor() {
        let mut monitor = SyncMonitor::new();
        monitor.update();
        assert!(monitor.metrics().sync_present);
    }
}
