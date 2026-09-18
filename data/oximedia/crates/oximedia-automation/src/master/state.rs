//! System state management.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// System status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SystemStatus {
    /// System is stopped
    #[default]
    Stopped,
    /// System is starting
    Starting,
    /// System is running normally
    Running,
    /// System is in failover mode
    Failover,
    /// System is stopping
    Stopping,
    /// System has encountered an error
    Error,
}

/// Overall system state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// Current system status
    pub status: SystemStatus,
    /// System uptime
    pub uptime: Duration,
    /// Last state change time
    pub last_state_change: SystemTime,
    /// Number of active channels
    pub active_channels: usize,
    /// Number of failovers triggered
    pub failover_count: u64,
    /// Number of EAS alerts processed
    pub eas_alert_count: u64,
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            status: SystemStatus::Stopped,
            uptime: Duration::ZERO,
            last_state_change: SystemTime::now(),
            active_channels: 0,
            failover_count: 0,
            eas_alert_count: 0,
        }
    }
}

impl SystemState {
    /// Update the system status.
    pub fn set_status(&mut self, status: SystemStatus) {
        self.status = status;
        self.last_state_change = SystemTime::now();
    }

    /// Increment failover count.
    pub fn increment_failover(&mut self) {
        self.failover_count += 1;
    }

    /// Increment EAS alert count.
    pub fn increment_eas_alert(&mut self) {
        self.eas_alert_count += 1;
    }

    /// Update uptime.
    pub fn update_uptime(&mut self, uptime: Duration) {
        self.uptime = uptime;
    }

    /// Set active channel count.
    pub fn set_active_channels(&mut self, count: usize) {
        self.active_channels = count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_state_default() {
        let state = SystemState::default();
        assert_eq!(state.status, SystemStatus::Stopped);
        assert_eq!(state.active_channels, 0);
        assert_eq!(state.failover_count, 0);
    }

    #[test]
    fn test_set_status() {
        let mut state = SystemState::default();
        state.set_status(SystemStatus::Running);
        assert_eq!(state.status, SystemStatus::Running);
    }

    #[test]
    fn test_increment_counts() {
        let mut state = SystemState::default();
        state.increment_failover();
        state.increment_eas_alert();
        assert_eq!(state.failover_count, 1);
        assert_eq!(state.eas_alert_count, 1);
    }
}
