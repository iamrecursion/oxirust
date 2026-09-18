//! Session closure.

use crate::{
    error::{ReviewError, ReviewResult},
    SessionId,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Reason for closing a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloseReason {
    /// Session completed successfully.
    Completed,
    /// Session cancelled by owner.
    Cancelled,
    /// Session expired (deadline passed).
    Expired,
    /// Session closed due to approval.
    Approved,
    /// Session closed due to rejection.
    Rejected,
    /// Custom reason.
    Custom(String),
}

/// Session closure details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionClosure {
    /// Session ID.
    pub session_id: SessionId,
    /// Reason for closure.
    pub reason: CloseReason,
    /// User who closed the session.
    pub closed_by: String,
    /// Closure timestamp.
    pub closed_at: DateTime<Utc>,
    /// Optional notes.
    pub notes: Option<String>,
}

/// Close a review session.
///
/// # Arguments
///
/// * `session_id` - ID of the session to close
/// * `reason` - Reason for closing
///
/// # Errors
///
/// Returns error if session cannot be closed.
pub async fn close_session(session_id: SessionId, reason: CloseReason) -> ReviewResult<()> {
    // In a real implementation, this would:
    // 1. Validate that session can be closed
    // 2. Update session status
    // 3. Archive session data
    // 4. Send notifications to participants
    // 5. Generate final report

    let _closure = SessionClosure {
        session_id,
        reason,
        closed_by: "system".to_string(),
        closed_at: Utc::now(),
        notes: None,
    };

    Ok(())
}

/// Close a session with detailed information.
///
/// # Arguments
///
/// * `session_id` - ID of the session
/// * `reason` - Reason for closing
/// * `closed_by` - User ID who is closing the session
/// * `notes` - Optional closure notes
///
/// # Errors
///
/// Returns error if session cannot be closed.
pub async fn close_session_detailed(
    session_id: SessionId,
    reason: CloseReason,
    closed_by: String,
    notes: Option<String>,
) -> ReviewResult<()> {
    let closure = SessionClosure {
        session_id,
        reason,
        closed_by,
        closed_at: Utc::now(),
        notes,
    };

    // In a real implementation, this would persist the closure details
    let _ = closure;

    Ok(())
}

/// Finalize a session after approval.
///
/// # Arguments
///
/// * `session_id` - ID of the session
///
/// # Errors
///
/// Returns error if finalization fails.
pub async fn finalize_approved_session(session_id: SessionId) -> ReviewResult<()> {
    close_session(session_id, CloseReason::Approved).await
}

/// Archive a closed session.
///
/// # Arguments
///
/// * `session_id` - ID of the session
///
/// # Errors
///
/// Returns error if archiving fails.
pub async fn archive_session(session_id: SessionId) -> ReviewResult<()> {
    // In a real implementation, this would:
    // 1. Move session data to archive storage
    // 2. Compress attachments
    // 3. Update indexes
    // 4. Generate archive manifest

    let _ = session_id;
    Ok(())
}

/// Reopen a closed session.
///
/// # Arguments
///
/// * `session_id` - ID of the session
///
/// # Errors
///
/// Returns error if session cannot be reopened.
pub async fn reopen_session(session_id: SessionId) -> ReviewResult<()> {
    // In a real implementation, this would:
    // 1. Validate that session can be reopened
    // 2. Restore session from archive if needed
    // 3. Update session status
    // 4. Send notifications

    let _ = session_id;
    Ok(())
}

/// Check if a session can be closed.
///
/// # Errors
///
/// Returns error if session cannot be closed.
pub fn validate_closure(
    has_pending_comments: bool,
    has_pending_approvals: bool,
) -> ReviewResult<()> {
    if has_pending_comments {
        return Err(ReviewError::InvalidStateTransition(
            "Cannot close session with unresolved comments".to_string(),
        ));
    }

    if has_pending_approvals {
        return Err(ReviewError::InvalidStateTransition(
            "Cannot close session with pending approvals".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_close_session() {
        let session_id = SessionId::new();
        let result = close_session(session_id, CloseReason::Completed).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_close_session_detailed() {
        let session_id = SessionId::new();
        let result = close_session_detailed(
            session_id,
            CloseReason::Approved,
            "user-123".to_string(),
            Some("All feedback addressed".to_string()),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_finalize_approved_session() {
        let session_id = SessionId::new();
        let result = finalize_approved_session(session_id).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_closure() {
        assert!(validate_closure(false, false).is_ok());
        assert!(validate_closure(true, false).is_err());
        assert!(validate_closure(false, true).is_err());
    }

    #[test]
    fn test_close_reason_equality() {
        assert_eq!(CloseReason::Completed, CloseReason::Completed);
        assert_ne!(CloseReason::Completed, CloseReason::Cancelled);
    }
}
