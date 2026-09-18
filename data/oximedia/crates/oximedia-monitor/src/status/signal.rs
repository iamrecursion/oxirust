//! Signal status monitoring.

use super::MonitoringStatus;

/// Signal monitor.
pub struct SignalMonitor {
    status: MonitoringStatus,
}

impl SignalMonitor {
    /// Create a new signal monitor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            status: MonitoringStatus::default(),
        }
    }

    /// Get current status.
    #[must_use]
    pub const fn status(&self) -> &MonitoringStatus {
        &self.status
    }
}

impl Default for SignalMonitor {
    fn default() -> Self {
        Self::new()
    }
}
