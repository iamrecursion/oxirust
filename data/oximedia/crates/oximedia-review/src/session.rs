//! Review session management.

use crate::{
    error::{ReviewError, ReviewResult},
    AnnotationType, CommentId, SessionConfig, SessionId, User, UserRole, VersionId,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::RwLock;

pub mod close;
pub mod create;
pub mod invite;

pub use close::CloseReason;
pub use create::create_session;
pub use invite::invite_participant;

/// Status of a review session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    /// Session is being created.
    Creating,
    /// Session is active and accepting reviews.
    Active,
    /// Session is paused.
    Paused,
    /// Session is completed successfully.
    Completed,
    /// Session is cancelled.
    Cancelled,
    /// Session is closed.
    Closed,
}

/// Review session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewSession {
    /// Session ID.
    pub id: SessionId,
    /// Session configuration.
    pub config: SessionConfig,
    /// Session status.
    pub status: SessionStatus,
    /// Session creator.
    pub creator: User,
    /// Participants in the session.
    pub participants: HashMap<String, User>,
    /// Current version being reviewed.
    pub current_version: Option<VersionId>,
    /// All versions in the session.
    pub versions: Vec<VersionId>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
    /// Closed timestamp.
    pub closed_at: Option<DateTime<Utc>>,
}

impl ReviewSession {
    /// Create a new review session.
    ///
    /// # Errors
    ///
    /// Returns error if session creation fails.
    pub async fn create(config: SessionConfig) -> ReviewResult<Self> {
        create::create_session(config).await
    }

    /// Invite a user to the session.
    ///
    /// # Errors
    ///
    /// Returns error if user cannot be invited.
    pub async fn invite_user(&self, email: &str) -> ReviewResult<()> {
        invite::invite_participant(self.id, email, UserRole::Reviewer).await
    }

    /// Close the session.
    ///
    /// # Errors
    ///
    /// Returns error if session cannot be closed.
    pub async fn close(&mut self, reason: CloseReason) -> ReviewResult<()> {
        close::close_session(self.id, reason).await?;
        self.status = SessionStatus::Closed;
        self.closed_at = Some(Utc::now());
        Ok(())
    }

    /// Add a comment to a specific frame.
    ///
    /// # Errors
    ///
    /// Returns error if comment cannot be added.
    pub async fn add_comment(
        &self,
        frame: i64,
        text: &str,
        annotation_type: AnnotationType,
    ) -> ReviewResult<CommentId> {
        if frame < 0 {
            return Err(ReviewError::InvalidFrame(frame));
        }

        if self.status == SessionStatus::Closed {
            return Err(ReviewError::SessionClosed);
        }

        crate::comment::add::add_comment(self.id, frame, text, annotation_type).await
    }

    /// Check if user has permission to perform an action.
    #[must_use]
    pub fn has_permission(&self, user_id: &str, required_role: UserRole) -> bool {
        self.participants
            .get(user_id)
            .is_some_and(|user| self.role_has_permission(user.role, required_role))
    }

    fn role_has_permission(&self, user_role: UserRole, required_role: UserRole) -> bool {
        match (user_role, required_role) {
            (UserRole::Owner, _) => true,
            (UserRole::Approver, UserRole::Approver | UserRole::Reviewer | UserRole::Observer) => {
                true
            }
            (UserRole::Reviewer, UserRole::Reviewer | UserRole::Observer) => true,
            (UserRole::Observer, UserRole::Observer) => true,
            _ => false,
        }
    }
}

/// Session manager for handling multiple sessions.
#[cfg(not(target_arch = "wasm32"))]
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<SessionId, ReviewSession>>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl SessionManager {
    /// Create a new session manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a session to the manager.
    pub async fn add_session(&self, session: ReviewSession) {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.id, session);
    }

    /// Get a session by ID.
    ///
    /// # Errors
    ///
    /// Returns error if session is not found.
    pub async fn get_session(&self, id: SessionId) -> ReviewResult<ReviewSession> {
        let sessions = self.sessions.read().await;
        sessions
            .get(&id)
            .cloned()
            .ok_or_else(|| ReviewError::SessionNotFound(id.to_string()))
    }

    /// Remove a session.
    pub async fn remove_session(&self, id: SessionId) -> ReviewResult<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(&id);
        Ok(())
    }

    /// List all sessions.
    pub async fn list_sessions(&self) -> Vec<ReviewSession> {
        let sessions = self.sessions.read().await;
        sessions.values().cloned().collect()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkflowType;

    #[tokio::test]
    async fn test_session_manager() {
        let manager = SessionManager::new();

        let config = SessionConfig::builder()
            .title("Test Session")
            .content_id("video-123")
            .workflow_type(WorkflowType::Simple)
            .build()
            .expect("valid config");

        let session = ReviewSession::create(config)
            .await
            .expect("should succeed in test");
        let session_id = session.id;

        manager.add_session(session).await;

        let retrieved = manager
            .get_session(session_id)
            .await
            .expect("should succeed in test");
        assert_eq!(retrieved.id, session_id);

        let sessions = manager.list_sessions().await;
        assert_eq!(sessions.len(), 1);
    }

    #[test]
    fn test_session_status_equality() {
        assert_eq!(SessionStatus::Active, SessionStatus::Active);
        assert_ne!(SessionStatus::Active, SessionStatus::Closed);
    }

    #[test]
    fn test_role_permissions() {
        let config = SessionConfig::builder()
            .title("Test")
            .content_id("test")
            .build()
            .expect("valid config");

        let session = ReviewSession {
            id: SessionId::new(),
            config,
            status: SessionStatus::Active,
            creator: User {
                id: "creator".to_string(),
                name: "Creator".to_string(),
                email: "creator@example.com".to_string(),
                role: UserRole::Owner,
            },
            participants: HashMap::new(),
            current_version: None,
            versions: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            closed_at: None,
        };

        assert!(session.role_has_permission(UserRole::Owner, UserRole::Reviewer));
        assert!(session.role_has_permission(UserRole::Approver, UserRole::Reviewer));
        assert!(!session.role_has_permission(UserRole::Observer, UserRole::Approver));
    }
}
