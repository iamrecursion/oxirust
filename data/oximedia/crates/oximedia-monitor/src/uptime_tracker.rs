//! Service uptime tracking and availability percentage calculation.
//!
//! Records intervals of service health state and computes the
//! availability percentage (`uptime / total_observed_time * 100`).

#![allow(dead_code)]

/// Operational state of a monitored service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServiceStatus {
    /// Service is responding normally.
    Up,
    /// Service is not responding or returning errors.
    Down,
    /// Service is responding but with degraded performance.
    Degraded,
    /// No health check has been performed yet.
    Unknown,
}

impl ServiceStatus {
    /// Returns `true` when the status counts as "up" for SLA purposes.
    #[must_use]
    pub fn is_available(self) -> bool {
        matches!(self, Self::Up | Self::Degraded)
    }

    /// Human-readable label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Up => "UP",
            Self::Down => "DOWN",
            Self::Degraded => "DEGRADED",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// A timestamped record of a status transition.
#[derive(Debug, Clone)]
pub struct UptimeRecord {
    /// The status that became active at this point.
    pub status: ServiceStatus,
    /// Monotonic start time in milliseconds.
    pub start_ms: u64,
    /// Monotonic end time in milliseconds (`None` = still active).
    pub end_ms: Option<u64>,
}

impl UptimeRecord {
    /// Create an open-ended record.
    #[must_use]
    pub fn new(status: ServiceStatus, start_ms: u64) -> Self {
        Self {
            status,
            start_ms,
            end_ms: None,
        }
    }

    /// Close the record by setting its end time.
    pub fn close(&mut self, end_ms: u64) {
        self.end_ms = Some(end_ms);
    }

    /// Duration of this record in milliseconds, using `now_ms` for open records.
    #[must_use]
    pub fn duration_ms(&self, now_ms: u64) -> u64 {
        let end = self.end_ms.unwrap_or(now_ms);
        end.saturating_sub(self.start_ms)
    }

    /// Returns `true` when the record is still open.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.end_ms.is_none()
    }
}

/// Tracks a service's uptime over time and computes availability.
#[derive(Debug)]
pub struct UptimeTracker {
    /// Name of the monitored service.
    pub service_name: String,
    /// Ordered list of status records (oldest first).
    records: Vec<UptimeRecord>,
    /// Monotonic time of the first recorded event.
    observation_start_ms: Option<u64>,
}

impl UptimeTracker {
    /// Create a new tracker for a named service.
    #[must_use]
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            records: Vec::new(),
            observation_start_ms: None,
        }
    }

    /// Record a status transition at `timestamp_ms`.
    ///
    /// If the status is identical to the current status the call is a no-op.
    pub fn record_status(&mut self, status: ServiceStatus, timestamp_ms: u64) {
        if self.observation_start_ms.is_none() {
            self.observation_start_ms = Some(timestamp_ms);
        }

        // Close the last open record if there is one.
        if let Some(last) = self.records.last_mut() {
            if last.status == status {
                return; // No change.
            }
            if last.is_open() {
                last.close(timestamp_ms);
            }
        }

        self.records.push(UptimeRecord::new(status, timestamp_ms));
    }

    /// Availability percentage over the entire observation window.
    ///
    /// Returns `0.0` if no events have been recorded.
    /// Uses `now_ms` to compute the duration of any open record.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn availability_pct(&self, now_ms: u64) -> f64 {
        let total = self.total_observation_ms(now_ms);
        if total == 0 {
            return 0.0;
        }
        let up_ms = self.uptime_ms(now_ms);
        (up_ms as f64 / total as f64) * 100.0
    }

    /// Total milliseconds the service was considered available.
    #[must_use]
    pub fn uptime_ms(&self, now_ms: u64) -> u64 {
        self.records
            .iter()
            .filter(|r| r.status.is_available())
            .map(|r| r.duration_ms(now_ms))
            .sum()
    }

    /// Total milliseconds the service was considered unavailable.
    #[must_use]
    pub fn downtime_ms(&self, now_ms: u64) -> u64 {
        self.records
            .iter()
            .filter(|r| !r.status.is_available() && r.status != ServiceStatus::Unknown)
            .map(|r| r.duration_ms(now_ms))
            .sum()
    }

    /// Total observation window length in milliseconds.
    #[must_use]
    pub fn total_observation_ms(&self, now_ms: u64) -> u64 {
        match self.observation_start_ms {
            None => 0,
            Some(start) => now_ms.saturating_sub(start),
        }
    }

    /// Current status (most recent record), or `Unknown`.
    #[must_use]
    pub fn current_status(&self) -> ServiceStatus {
        self.records
            .last()
            .map_or(ServiceStatus::Unknown, |r| r.status)
    }

    /// Number of status transitions recorded.
    #[must_use]
    pub fn transition_count(&self) -> usize {
        self.records.len()
    }

    /// Count of discrete down events.
    #[must_use]
    pub fn outage_count(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.status == ServiceStatus::Down)
            .count()
    }

    /// All records in chronological order.
    #[must_use]
    pub fn records(&self) -> &[UptimeRecord] {
        &self.records
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_status_is_available() {
        assert!(ServiceStatus::Up.is_available());
        assert!(ServiceStatus::Degraded.is_available());
        assert!(!ServiceStatus::Down.is_available());
        assert!(!ServiceStatus::Unknown.is_available());
    }

    #[test]
    fn test_service_status_as_str() {
        assert_eq!(ServiceStatus::Up.as_str(), "UP");
        assert_eq!(ServiceStatus::Down.as_str(), "DOWN");
        assert_eq!(ServiceStatus::Degraded.as_str(), "DEGRADED");
        assert_eq!(ServiceStatus::Unknown.as_str(), "UNKNOWN");
    }

    #[test]
    fn test_uptime_record_duration_closed() {
        let mut r = UptimeRecord::new(ServiceStatus::Up, 1000);
        r.close(2000);
        assert_eq!(r.duration_ms(9999), 1000);
        assert!(!r.is_open());
    }

    #[test]
    fn test_uptime_record_duration_open() {
        let r = UptimeRecord::new(ServiceStatus::Up, 1000);
        assert_eq!(r.duration_ms(3000), 2000);
        assert!(r.is_open());
    }

    #[test]
    fn test_no_events_availability_zero() {
        let tracker = UptimeTracker::new("svc");
        assert_eq!(tracker.availability_pct(5000), 0.0);
    }

    #[test]
    fn test_full_uptime_100_pct() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Up, 0);
        // All time = up time → 100 %
        assert!((t.availability_pct(1000) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_half_uptime_50_pct() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Up, 0);
        t.record_status(ServiceStatus::Down, 500);
        // 500ms up, 500ms down out of 1000ms total
        let pct = t.availability_pct(1000);
        assert!((pct - 50.0).abs() < 1e-6, "pct={pct}");
    }

    #[test]
    fn test_duplicate_status_no_op() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Up, 0);
        t.record_status(ServiceStatus::Up, 500); // duplicate, should not add a record
        assert_eq!(t.transition_count(), 1);
    }

    #[test]
    fn test_current_status() {
        let mut t = UptimeTracker::new("svc");
        assert_eq!(t.current_status(), ServiceStatus::Unknown);
        t.record_status(ServiceStatus::Down, 0);
        assert_eq!(t.current_status(), ServiceStatus::Down);
    }

    #[test]
    fn test_outage_count() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Up, 0);
        t.record_status(ServiceStatus::Down, 100);
        t.record_status(ServiceStatus::Up, 200);
        t.record_status(ServiceStatus::Down, 300);
        assert_eq!(t.outage_count(), 2);
    }

    #[test]
    fn test_downtime_ms() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Down, 0);
        t.record_status(ServiceStatus::Up, 400);
        assert_eq!(t.downtime_ms(1000), 400);
    }

    #[test]
    fn test_total_observation_window() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Up, 100);
        assert_eq!(t.total_observation_ms(600), 500);
    }

    #[test]
    fn test_degraded_counts_as_available() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Degraded, 0);
        // All time degraded → still 100 % available
        assert!((t.availability_pct(1000) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_records_slice_length() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Up, 0);
        t.record_status(ServiceStatus::Down, 50);
        assert_eq!(t.records().len(), 2);
    }

    #[test]
    fn test_uptime_ms_accuracy() {
        let mut t = UptimeTracker::new("svc");
        t.record_status(ServiceStatus::Up, 0);
        t.record_status(ServiceStatus::Down, 300);
        assert_eq!(t.uptime_ms(1000), 300);
    }
}
