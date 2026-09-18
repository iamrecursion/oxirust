//! Change request types.

use crate::change::{ChangePriority, ChangeRequestId, ChangeStatus};
use crate::SessionId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Change request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRequest {
    /// Request ID.
    pub id: ChangeRequestId,
    /// Session ID.
    pub session_id: SessionId,
    /// Request title.
    pub title: String,
    /// Detailed description.
    pub description: String,
    /// Priority level.
    pub priority: ChangePriority,
    /// Current status.
    pub status: ChangeStatus,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
    /// Completion timestamp.
    pub completed_at: Option<DateTime<Utc>>,
    /// Assigned user ID.
    pub assigned_to: Option<String>,
}

impl ChangeRequest {
    /// Check if request is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            ChangeStatus::Completed | ChangeStatus::Rejected
        )
    }

    /// Check if request is pending.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.status == ChangeStatus::Pending
    }

    /// Mark as complete.
    pub fn complete(&mut self) {
        self.status = ChangeStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Assign to a user.
    pub fn assign(&mut self, user_id: String) {
        self.assigned_to = Some(user_id);
        self.updated_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request() -> ChangeRequest {
        ChangeRequest {
            id: ChangeRequestId::new(),
            session_id: SessionId::new(),
            title: "Test request".to_string(),
            description: "Test description".to_string(),
            priority: ChangePriority::Normal,
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            completed_at: None,
            assigned_to: None,
        }
    }

    #[test]
    fn test_change_request_is_pending() {
        let request = create_test_request();
        assert!(request.is_pending());
        assert!(!request.is_complete());
    }

    #[test]
    fn test_change_request_complete() {
        let mut request = create_test_request();
        request.complete();

        assert!(request.is_complete());
        assert!(request.completed_at.is_some());
    }

    #[test]
    fn test_change_request_assign() {
        let mut request = create_test_request();
        request.assign("user-123".to_string());

        assert_eq!(request.assigned_to, Some("user-123".to_string()));
    }
}
