//! Approval workflow engine.

use crate::{error::ReviewResult, SessionId, User};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod conditions;
pub mod decision;
pub mod stage;
pub mod workflow;

pub use decision::{ApprovalDecision, DecisionType};
pub use stage::{ApprovalStage, StageStatus};
pub use workflow::{ApprovalWorkflow, WorkflowStatus};

/// Approval ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ApprovalId(Uuid);

impl ApprovalId {
    /// Create a new approval ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ApprovalId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ApprovalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Approval ID.
    pub id: ApprovalId,
    /// Session ID.
    pub session_id: SessionId,
    /// Requester.
    pub requester: User,
    /// Approver.
    pub approver: User,
    /// Request message.
    pub message: Option<String>,
    /// Due date.
    pub due_date: Option<DateTime<Utc>>,
    /// Status.
    pub status: ApprovalStatus,
    /// Created timestamp.
    pub created_at: DateTime<Utc>,
    /// Responded timestamp.
    pub responded_at: Option<DateTime<Utc>>,
}

/// Status of an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalStatus {
    /// Pending approval.
    Pending,
    /// Approved.
    Approved,
    /// Rejected.
    Rejected,
    /// Conditionally approved.
    Conditional,
    /// Cancelled.
    Cancelled,
    /// Expired.
    Expired,
}

impl ApprovalStatus {
    /// Check if status is final (not pending).
    #[must_use]
    pub fn is_final(self) -> bool {
        !matches!(self, Self::Pending)
    }

    /// Check if status is positive (approved or conditional).
    #[must_use]
    pub fn is_positive(self) -> bool {
        matches!(self, Self::Approved | Self::Conditional)
    }
}

/// Create an approval request.
///
/// # Errors
///
/// Returns error if creation fails.
pub async fn create_approval_request(
    session_id: SessionId,
    requester: User,
    approver: User,
) -> ReviewResult<ApprovalRequest> {
    Ok(ApprovalRequest {
        id: ApprovalId::new(),
        session_id,
        requester,
        approver,
        message: None,
        due_date: None,
        status: ApprovalStatus::Pending,
        created_at: Utc::now(),
        responded_at: None,
    })
}

/// Submit approval decision.
///
/// # Errors
///
/// Returns error if submission fails.
pub async fn submit_approval(
    approval_id: ApprovalId,
    decision: DecisionType,
    comments: Option<String>,
) -> ReviewResult<()> {
    let _ = (approval_id, decision, comments);
    // In a real implementation, this would update the database
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UserRole;

    fn create_test_user(id: &str) -> User {
        User {
            id: id.to_string(),
            name: format!("User {}", id),
            email: format!("{}@example.com", id),
            role: UserRole::Approver,
        }
    }

    #[tokio::test]
    async fn test_create_approval_request() {
        let session_id = SessionId::new();
        let requester = create_test_user("requester");
        let approver = create_test_user("approver");

        let request = create_approval_request(session_id, requester, approver)
            .await
            .expect("should succeed in test");

        assert_eq!(request.status, ApprovalStatus::Pending);
        assert!(request.responded_at.is_none());
    }

    #[test]
    fn test_approval_status_is_final() {
        assert!(!ApprovalStatus::Pending.is_final());
        assert!(ApprovalStatus::Approved.is_final());
        assert!(ApprovalStatus::Rejected.is_final());
    }

    #[test]
    fn test_approval_status_is_positive() {
        assert!(ApprovalStatus::Approved.is_positive());
        assert!(ApprovalStatus::Conditional.is_positive());
        assert!(!ApprovalStatus::Rejected.is_positive());
    }
}
