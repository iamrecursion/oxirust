//! Status summary generation.

use crate::status::ReviewStatus;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Status summary for a review session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusSummary {
    /// Current status.
    pub current_status: ReviewStatus,
    /// Status history.
    pub history: Vec<StatusChange>,
    /// Summary generated at.
    pub generated_at: DateTime<Utc>,
}

/// Status change record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusChange {
    /// Previous status.
    pub from_status: ReviewStatus,
    /// New status.
    pub to_status: ReviewStatus,
    /// User who changed the status.
    pub changed_by: String,
    /// Change timestamp.
    pub changed_at: DateTime<Utc>,
    /// Reason for change.
    pub reason: Option<String>,
}

impl StatusSummary {
    /// Create a new status summary.
    #[must_use]
    pub fn new(current_status: ReviewStatus) -> Self {
        Self {
            current_status,
            history: Vec::new(),
            generated_at: Utc::now(),
        }
    }

    /// Add a status change to history.
    pub fn add_change(&mut self, change: StatusChange) {
        self.current_status = change.to_status;
        self.history.push(change);
    }

    /// Get the latest status change.
    #[must_use]
    pub fn latest_change(&self) -> Option<&StatusChange> {
        self.history.last()
    }

    /// Get status changes by user.
    #[must_use]
    pub fn changes_by_user(&self, user_id: &str) -> Vec<&StatusChange> {
        self.history
            .iter()
            .filter(|c| c.changed_by == user_id)
            .collect()
    }

    /// Count status changes.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.history.len()
    }
}

impl StatusChange {
    /// Create a new status change.
    #[must_use]
    pub fn new(from_status: ReviewStatus, to_status: ReviewStatus, changed_by: String) -> Self {
        Self {
            from_status,
            to_status,
            changed_by,
            changed_at: Utc::now(),
            reason: None,
        }
    }

    /// Set the reason for the change.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_summary_creation() {
        let summary = StatusSummary::new(ReviewStatus::NotStarted);
        assert_eq!(summary.current_status, ReviewStatus::NotStarted);
        assert_eq!(summary.change_count(), 0);
    }

    #[test]
    fn test_status_summary_add_change() {
        let mut summary = StatusSummary::new(ReviewStatus::NotStarted);

        let change = StatusChange::new(
            ReviewStatus::NotStarted,
            ReviewStatus::InProgress,
            "user-1".to_string(),
        );

        summary.add_change(change);

        assert_eq!(summary.current_status, ReviewStatus::InProgress);
        assert_eq!(summary.change_count(), 1);
    }

    #[test]
    fn test_status_change_with_reason() {
        let change = StatusChange::new(
            ReviewStatus::InProgress,
            ReviewStatus::OnHold,
            "user-1".to_string(),
        )
        .with_reason("Waiting for client feedback");

        assert_eq!(
            change.reason,
            Some("Waiting for client feedback".to_string())
        );
    }

    #[test]
    fn test_changes_by_user() {
        let mut summary = StatusSummary::new(ReviewStatus::NotStarted);

        summary.add_change(StatusChange::new(
            ReviewStatus::NotStarted,
            ReviewStatus::InProgress,
            "user-1".to_string(),
        ));

        summary.add_change(StatusChange::new(
            ReviewStatus::InProgress,
            ReviewStatus::UnderReview,
            "user-2".to_string(),
        ));

        summary.add_change(StatusChange::new(
            ReviewStatus::UnderReview,
            ReviewStatus::Approved,
            "user-1".to_string(),
        ));

        let user1_changes = summary.changes_by_user("user-1");
        assert_eq!(user1_changes.len(), 2);
    }
}
