//! Task deadline management.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Task deadline information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDeadline {
    /// Deadline timestamp.
    pub due_date: DateTime<Utc>,
    /// Warning threshold (hours before deadline).
    pub warning_threshold: i64,
    /// Whether to send reminders.
    pub send_reminders: bool,
}

impl TaskDeadline {
    /// Create a new deadline.
    #[must_use]
    pub fn new(due_date: DateTime<Utc>) -> Self {
        Self {
            due_date,
            warning_threshold: 24,
            send_reminders: true,
        }
    }

    /// Check if deadline is approaching.
    #[must_use]
    pub fn is_approaching(&self) -> bool {
        let now = Utc::now();
        let threshold = self.due_date - Duration::hours(self.warning_threshold);
        now >= threshold && now < self.due_date
    }

    /// Check if deadline is passed.
    #[must_use]
    pub fn is_passed(&self) -> bool {
        Utc::now() > self.due_date
    }

    /// Get time remaining.
    #[must_use]
    pub fn time_remaining(&self) -> Duration {
        self.due_date - Utc::now()
    }

    /// Get time remaining in hours.
    #[must_use]
    pub fn hours_remaining(&self) -> i64 {
        self.time_remaining().num_hours()
    }

    /// Get formatted deadline string.
    #[must_use]
    pub fn format_deadline(&self) -> String {
        self.due_date.format("%Y-%m-%d %H:%M:%S UTC").to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deadline_creation() {
        let due_date = Utc::now() + Duration::days(1);
        let deadline = TaskDeadline::new(due_date);
        assert_eq!(deadline.warning_threshold, 24);
        assert!(deadline.send_reminders);
    }

    #[test]
    fn test_deadline_is_passed() {
        let past_date = Utc::now() - Duration::days(1);
        let deadline = TaskDeadline::new(past_date);
        assert!(deadline.is_passed());

        let future_date = Utc::now() + Duration::days(1);
        let deadline = TaskDeadline::new(future_date);
        assert!(!deadline.is_passed());
    }

    #[test]
    fn test_deadline_is_approaching() {
        let soon = Utc::now() + Duration::hours(12);
        let deadline = TaskDeadline::new(soon);
        assert!(deadline.is_approaching());

        let far = Utc::now() + Duration::days(7);
        let deadline = TaskDeadline::new(far);
        assert!(!deadline.is_approaching());
    }

    #[test]
    fn test_hours_remaining() {
        let due_date = Utc::now() + Duration::hours(48);
        let deadline = TaskDeadline::new(due_date);
        let hours = deadline.hours_remaining();
        assert!(hours >= 47 && hours <= 48);
    }
}
