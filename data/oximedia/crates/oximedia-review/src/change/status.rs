//! Change request status tracking.

use serde::{Deserialize, Serialize};

/// Status of a change request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeStatus {
    /// Request is pending review.
    Pending,
    /// Request is in progress.
    InProgress,
    /// Request is blocked.
    Blocked,
    /// Request is completed.
    Completed,
    /// Request is rejected.
    Rejected,
    /// Request is deferred.
    Deferred,
}

impl ChangeStatus {
    /// Get all status values.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Pending,
            Self::InProgress,
            Self::Blocked,
            Self::Completed,
            Self::Rejected,
            Self::Deferred,
        ]
    }

    /// Check if status is final.
    #[must_use]
    pub fn is_final(self) -> bool {
        matches!(self, Self::Completed | Self::Rejected | Self::Deferred)
    }

    /// Check if status is active.
    #[must_use]
    pub fn is_active(self) -> bool {
        matches!(self, Self::InProgress | Self::Blocked)
    }

    /// Get status name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::InProgress => "In Progress",
            Self::Blocked => "Blocked",
            Self::Completed => "Completed",
            Self::Rejected => "Rejected",
            Self::Deferred => "Deferred",
        }
    }
}

impl std::fmt::Display for ChangeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_is_final() {
        assert!(!ChangeStatus::Pending.is_final());
        assert!(!ChangeStatus::InProgress.is_final());
        assert!(ChangeStatus::Completed.is_final());
        assert!(ChangeStatus::Rejected.is_final());
        assert!(ChangeStatus::Deferred.is_final());
    }

    #[test]
    fn test_status_is_active() {
        assert!(ChangeStatus::InProgress.is_active());
        assert!(ChangeStatus::Blocked.is_active());
        assert!(!ChangeStatus::Pending.is_active());
        assert!(!ChangeStatus::Completed.is_active());
    }

    #[test]
    fn test_status_name() {
        assert_eq!(ChangeStatus::Pending.name(), "Pending");
        assert_eq!(ChangeStatus::InProgress.name(), "In Progress");
        assert_eq!(ChangeStatus::Completed.name(), "Completed");
    }

    #[test]
    fn test_status_all() {
        let all = ChangeStatus::all();
        assert_eq!(all.len(), 6);
    }
}
