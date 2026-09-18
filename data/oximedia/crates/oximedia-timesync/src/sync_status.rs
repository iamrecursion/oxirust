//! Sync-source enumeration, health reporting, and status monitoring.

#![allow(dead_code)]

use std::collections::HashMap;

/// A time-synchronization source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyncSource {
    /// IEEE 1588 Precision Time Protocol.
    Ptp,
    /// Network Time Protocol.
    Ntp,
    /// GPS disciplined oscillator.
    Gps,
    /// Local free-running crystal oscillator (holdover).
    FreeRunning,
    /// Audio Engineering Society AES67.
    Aes67,
    /// External word-clock or video sync (genlock).
    Genlock,
}

impl SyncSource {
    /// Lower value = higher priority (PTP is best, free-running is worst).
    #[must_use]
    pub fn priority(&self) -> u8 {
        match self {
            Self::Gps => 0,
            Self::Ptp => 1,
            Self::Aes67 => 2,
            Self::Genlock => 3,
            Self::Ntp => 4,
            Self::FreeRunning => 255,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ptp => "PTP",
            Self::Ntp => "NTP",
            Self::Gps => "GPS",
            Self::FreeRunning => "Free-Running",
            Self::Aes67 => "AES67",
            Self::Genlock => "Genlock",
        }
    }
}

/// Health state of a synchronization source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncHealth {
    /// Locked and within specification.
    Ok,
    /// Tracking but error is elevated.
    Degraded,
    /// Source lost or error exceeds threshold.
    Lost,
    /// Not yet attempted.
    Unknown,
}

impl SyncHealth {
    /// `true` when the source is healthy.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        *self == Self::Ok
    }

    /// `true` when the source is usable even if not perfect.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Ok | Self::Degraded)
    }
}

/// The complete synchronization status of a node.
#[derive(Debug, Clone)]
pub struct SyncStatus {
    /// Active source driving the local clock.
    pub active_source: Option<SyncSource>,
    /// Current offset from reference in nanoseconds.
    pub offset_ns: i64,
    /// Estimated maximum error in nanoseconds.
    pub max_error_ns: u64,
    /// Health per source.
    pub health: HashMap<SyncSource, SyncHealth>,
}

impl SyncStatus {
    /// Create a blank (not yet synchronized) status.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_source: None,
            offset_ns: 0,
            max_error_ns: u64::MAX,
            health: HashMap::new(),
        }
    }

    /// `true` when the node is currently synchronized to a usable source.
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        if let Some(src) = self.active_source {
            self.health.get(&src).map_or(false, SyncHealth::is_usable)
        } else {
            false
        }
    }

    /// Absolute offset in microseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn offset_us(&self) -> f64 {
        self.offset_ns.unsigned_abs() as f64 / 1_000.0
    }
}

impl Default for SyncStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// Monitors multiple synchronization sources and maintains current status.
#[derive(Debug)]
pub struct SyncStatusMonitor {
    status: SyncStatus,
}

impl SyncStatusMonitor {
    /// Create a new monitor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            status: SyncStatus::new(),
        }
    }

    /// Update the health and offset of a given source.
    pub fn update(&mut self, source: SyncSource, health: SyncHealth, offset_ns: i64) {
        self.status.health.insert(source, health);

        // Promote this source to active if it is healthier than the current active.
        let should_activate = match self.status.active_source {
            None => health.is_usable(),
            Some(current) => health.is_usable() && source.priority() < current.priority(),
        };

        if should_activate {
            self.status.active_source = Some(source);
            self.status.offset_ns = offset_ns;
        } else if self.status.active_source == Some(source) {
            self.status.offset_ns = offset_ns;
            // If the active source just became unusable, find the next best.
            if !health.is_usable() {
                self.status.active_source = self.best_source();
            }
        }
    }

    /// Return the best currently usable source (by priority), if any.
    #[must_use]
    pub fn best_source(&self) -> Option<SyncSource> {
        self.status
            .health
            .iter()
            .filter(|(_, h)| h.is_usable())
            .min_by_key(|(s, _)| s.priority())
            .map(|(s, _)| *s)
    }

    /// Return a reference to the current status.
    #[must_use]
    pub fn status(&self) -> &SyncStatus {
        &self.status
    }
}

impl Default for SyncStatusMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptp_higher_priority_than_ntp() {
        assert!(SyncSource::Ptp.priority() < SyncSource::Ntp.priority());
    }

    #[test]
    fn test_gps_highest_priority() {
        assert_eq!(SyncSource::Gps.priority(), 0);
    }

    #[test]
    fn test_free_running_lowest_priority() {
        assert_eq!(SyncSource::FreeRunning.priority(), 255);
    }

    #[test]
    fn test_source_name_ptp() {
        assert_eq!(SyncSource::Ptp.name(), "PTP");
    }

    #[test]
    fn test_sync_health_ok_is_ok() {
        assert!(SyncHealth::Ok.is_ok());
    }

    #[test]
    fn test_sync_health_degraded_not_ok() {
        assert!(!SyncHealth::Degraded.is_ok());
    }

    #[test]
    fn test_sync_health_degraded_is_usable() {
        assert!(SyncHealth::Degraded.is_usable());
    }

    #[test]
    fn test_sync_health_lost_not_usable() {
        assert!(!SyncHealth::Lost.is_usable());
    }

    #[test]
    fn test_sync_status_not_synchronized_initially() {
        let s = SyncStatus::new();
        assert!(!s.is_synchronized());
    }

    #[test]
    fn test_sync_status_offset_us() {
        let mut s = SyncStatus::new();
        s.offset_ns = 5_000;
        assert!((s.offset_us() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_monitor_update_activates_source() {
        let mut m = SyncStatusMonitor::new();
        m.update(SyncSource::Ntp, SyncHealth::Ok, 1000);
        assert!(m.status().is_synchronized());
        assert_eq!(m.status().active_source, Some(SyncSource::Ntp));
    }

    #[test]
    fn test_monitor_best_source_higher_priority_wins() {
        let mut m = SyncStatusMonitor::new();
        m.update(SyncSource::Ntp, SyncHealth::Ok, 2000);
        m.update(SyncSource::Ptp, SyncHealth::Ok, 100);
        assert_eq!(m.best_source(), Some(SyncSource::Ptp));
    }

    #[test]
    fn test_monitor_lost_source_not_usable() {
        let mut m = SyncStatusMonitor::new();
        m.update(SyncSource::Ptp, SyncHealth::Ok, 500);
        m.update(SyncSource::Ptp, SyncHealth::Lost, 0);
        // PTP is now lost; monitor should have no active source (no other source).
        assert!(!m.status().is_synchronized());
    }

    #[test]
    fn test_monitor_fallback_on_active_source_loss() {
        let mut m = SyncStatusMonitor::new();
        m.update(SyncSource::Ntp, SyncHealth::Ok, 3000);
        m.update(SyncSource::Ptp, SyncHealth::Ok, 200);
        m.update(SyncSource::Ptp, SyncHealth::Lost, 0);
        // Should fall back to NTP.
        assert_eq!(m.best_source(), Some(SyncSource::Ntp));
    }

    #[test]
    fn test_sync_status_default() {
        let s = SyncStatus::default();
        assert!(s.active_source.is_none());
    }
}
