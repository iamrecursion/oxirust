//! Delivery log for tracking asset delivery history.
//!
//! Records each delivery attempt with its outcome and provides
//! summary queries over the history.

#![allow(dead_code)]

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Outcome of a delivery attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryStatus {
    /// Delivery completed successfully.
    Success,
    /// Delivery failed with an error message.
    Failed(String),
    /// Delivery was cancelled before it completed.
    Cancelled,
    /// Delivery is still in progress.
    InProgress,
}

impl DeliveryStatus {
    /// Returns `true` when the delivery succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` when the delivery is terminal (not in progress).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::InProgress)
    }

    /// Short label for display.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Success => "Success",
            Self::Failed(_) => "Failed",
            Self::Cancelled => "Cancelled",
            Self::InProgress => "In Progress",
        }
    }
}

/// A single delivery record.
#[derive(Debug, Clone)]
pub struct DeliveryRecord {
    /// Unique record id.
    pub id: u64,
    /// Asset that was delivered.
    pub asset_id: u64,
    /// Destination identifier (path, URL, service name, etc.).
    pub destination: String,
    /// Outcome of the delivery.
    pub status: DeliveryStatus,
    /// Unix timestamp (seconds) when the delivery was recorded.
    pub timestamp_secs: u64,
    /// Optional bytes transferred.
    pub bytes_transferred: Option<u64>,
}

impl DeliveryRecord {
    /// Create a new delivery record with the current system time.
    #[must_use]
    pub fn new(
        id: u64,
        asset_id: u64,
        destination: impl Into<String>,
        status: DeliveryStatus,
    ) -> Self {
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self {
            id,
            asset_id,
            destination: destination.into(),
            status,
            timestamp_secs,
            bytes_transferred: None,
        }
    }

    /// Create a record with an explicit timestamp (useful for tests).
    #[must_use]
    pub fn with_timestamp(
        id: u64,
        asset_id: u64,
        destination: impl Into<String>,
        status: DeliveryStatus,
        timestamp_secs: u64,
    ) -> Self {
        Self {
            id,
            asset_id,
            destination: destination.into(),
            status,
            timestamp_secs,
            bytes_transferred: None,
        }
    }

    /// Approximate number of days ago this delivery was recorded.
    ///
    /// Uses the current system time and the stored Unix timestamp.
    #[must_use]
    pub fn days_ago(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        now.saturating_sub(self.timestamp_secs) / 86_400
    }

    /// Returns `true` if the delivery was successful.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }
}

/// An append-only log of delivery records.
#[derive(Debug, Default)]
pub struct DeliveryLog {
    records: Vec<DeliveryRecord>,
    next_id: u64,
}

impl DeliveryLog {
    /// Create a new, empty delivery log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new delivery attempt, returning the assigned record id.
    pub fn record(
        &mut self,
        asset_id: u64,
        destination: impl Into<String>,
        status: DeliveryStatus,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.records
            .push(DeliveryRecord::new(id, asset_id, destination, status));
        id
    }

    /// All records in insertion order.
    #[must_use]
    pub fn all(&self) -> &[DeliveryRecord] {
        &self.records
    }

    /// Records for which the delivery was successful.
    #[must_use]
    pub fn successful_deliveries(&self) -> Vec<&DeliveryRecord> {
        self.records.iter().filter(|r| r.is_success()).collect()
    }

    /// Records recorded within the last `days` days.
    #[must_use]
    pub fn recent(&self, days: u64) -> Vec<&DeliveryRecord> {
        self.records
            .iter()
            .filter(|r| r.days_ago() <= days)
            .collect()
    }

    /// Records for a specific asset id.
    #[must_use]
    pub fn for_asset(&self, asset_id: u64) -> Vec<&DeliveryRecord> {
        self.records
            .iter()
            .filter(|r| r.asset_id == asset_id)
            .collect()
    }

    /// Total number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if the log has no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delivery_status_is_success() {
        assert!(DeliveryStatus::Success.is_success());
        assert!(!DeliveryStatus::Failed("err".to_string()).is_success());
        assert!(!DeliveryStatus::Cancelled.is_success());
        assert!(!DeliveryStatus::InProgress.is_success());
    }

    #[test]
    fn test_delivery_status_is_terminal() {
        assert!(DeliveryStatus::Success.is_terminal());
        assert!(DeliveryStatus::Failed("x".to_string()).is_terminal());
        assert!(DeliveryStatus::Cancelled.is_terminal());
        assert!(!DeliveryStatus::InProgress.is_terminal());
    }

    #[test]
    fn test_delivery_status_label() {
        assert_eq!(DeliveryStatus::Success.label(), "Success");
        assert_eq!(DeliveryStatus::Failed("oops".to_string()).label(), "Failed");
        assert_eq!(DeliveryStatus::Cancelled.label(), "Cancelled");
        assert_eq!(DeliveryStatus::InProgress.label(), "In Progress");
    }

    #[test]
    fn test_delivery_record_new() {
        let rec = DeliveryRecord::new(1, 100, "s3://bucket/file.mp4", DeliveryStatus::Success);
        assert_eq!(rec.id, 1);
        assert_eq!(rec.asset_id, 100);
        assert_eq!(rec.destination, "s3://bucket/file.mp4");
        assert!(rec.is_success());
    }

    #[test]
    fn test_delivery_record_days_ago_recent() {
        // A record created now should report 0 days ago
        let rec = DeliveryRecord::new(0, 1, "dest", DeliveryStatus::Success);
        assert_eq!(rec.days_ago(), 0);
    }

    #[test]
    fn test_delivery_record_days_ago_old() {
        // Timestamp 10 days in the past
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("should succeed in test")
            .as_secs();
        let ten_days_ago = now.saturating_sub(10 * 86_400);
        let rec =
            DeliveryRecord::with_timestamp(0, 1, "dest", DeliveryStatus::Success, ten_days_ago);
        assert_eq!(rec.days_ago(), 10);
    }

    #[test]
    fn test_delivery_log_new_is_empty() {
        let log = DeliveryLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_delivery_log_record_and_len() {
        let mut log = DeliveryLog::new();
        let id1 = log.record(1, "dest1", DeliveryStatus::Success);
        let id2 = log.record(2, "dest2", DeliveryStatus::Cancelled);
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_delivery_log_successful_deliveries() {
        let mut log = DeliveryLog::new();
        log.record(1, "a", DeliveryStatus::Success);
        log.record(2, "b", DeliveryStatus::Failed("err".to_string()));
        log.record(3, "c", DeliveryStatus::Success);
        let ok = log.successful_deliveries();
        assert_eq!(ok.len(), 2);
        assert!(ok.iter().all(|r| r.is_success()));
    }

    #[test]
    fn test_delivery_log_recent_all_current() {
        let mut log = DeliveryLog::new();
        log.record(1, "a", DeliveryStatus::Success);
        log.record(2, "b", DeliveryStatus::Success);
        // All records are just created — 0 days ago
        let recent = log.recent(0);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_delivery_log_recent_excludes_old() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("should succeed in test")
            .as_secs();
        let old_ts = now.saturating_sub(20 * 86_400);
        let mut log = DeliveryLog::new();
        log.records.push(DeliveryRecord::with_timestamp(
            0,
            1,
            "old",
            DeliveryStatus::Success,
            old_ts,
        ));
        log.records
            .push(DeliveryRecord::new(1, 2, "new", DeliveryStatus::Success));
        let recent = log.recent(7);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].destination, "new");
    }

    #[test]
    fn test_delivery_log_for_asset() {
        let mut log = DeliveryLog::new();
        log.record(10, "x", DeliveryStatus::Success);
        log.record(20, "y", DeliveryStatus::Success);
        log.record(10, "z", DeliveryStatus::Cancelled);
        let asset10 = log.for_asset(10);
        assert_eq!(asset10.len(), 2);
        assert!(asset10.iter().all(|r| r.asset_id == 10));
    }

    #[test]
    fn test_delivery_log_all_returns_all() {
        let mut log = DeliveryLog::new();
        for i in 0..5 {
            log.record(i, "d", DeliveryStatus::Success);
        }
        assert_eq!(log.all().len(), 5);
    }
}
