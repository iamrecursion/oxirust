//! Session creation.

use crate::{
    error::ReviewResult,
    session::{ReviewSession, SessionStatus},
    SessionConfig, SessionId, User, UserRole,
};
use chrono::Utc;
use std::collections::HashMap;

/// Create a new review session.
///
/// # Arguments
///
/// * `config` - Session configuration
///
/// # Errors
///
/// Returns error if session creation fails.
pub async fn create_session(config: SessionConfig) -> ReviewResult<ReviewSession> {
    let session_id = SessionId::new();

    // Create the session creator as a default user
    let creator = User {
        id: "system".to_string(),
        name: "System".to_string(),
        email: "system@oximedia.local".to_string(),
        role: UserRole::Owner,
    };

    let now = Utc::now();

    let session = ReviewSession {
        id: session_id,
        config,
        status: SessionStatus::Active,
        creator: creator.clone(),
        participants: {
            let mut map = HashMap::new();
            map.insert(creator.id.clone(), creator);
            map
        },
        current_version: None,
        versions: Vec::new(),
        created_at: now,
        updated_at: now,
        closed_at: None,
    };

    Ok(session)
}

/// Create a session with a specific creator.
///
/// # Arguments
///
/// * `config` - Session configuration
/// * `creator` - User who is creating the session
///
/// # Errors
///
/// Returns error if session creation fails.
pub async fn create_session_with_creator(
    config: SessionConfig,
    creator: User,
) -> ReviewResult<ReviewSession> {
    let session_id = SessionId::new();
    let now = Utc::now();

    let session = ReviewSession {
        id: session_id,
        config,
        status: SessionStatus::Active,
        creator: creator.clone(),
        participants: {
            let mut map = HashMap::new();
            map.insert(creator.id.clone(), creator);
            map
        },
        current_version: None,
        versions: Vec::new(),
        created_at: now,
        updated_at: now,
        closed_at: None,
    };

    Ok(session)
}

/// Validate session configuration.
///
/// # Errors
///
/// Returns error if configuration is invalid.
pub fn validate_config(config: &SessionConfig) -> ReviewResult<()> {
    if config.title.is_empty() {
        return Err(crate::error::ReviewError::InvalidConfig(
            "Title cannot be empty".to_string(),
        ));
    }

    if config.content_id.is_empty() {
        return Err(crate::error::ReviewError::InvalidConfig(
            "Content ID cannot be empty".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkflowType;

    #[tokio::test]
    async fn test_create_session() {
        let config = SessionConfig::builder()
            .title("Test Session")
            .content_id("video-123")
            .workflow_type(WorkflowType::Simple)
            .build()
            .expect("valid config");

        let session = create_session(config)
            .await
            .expect("should succeed in test");
        assert_eq!(session.status, SessionStatus::Active);
        assert_eq!(session.participants.len(), 1);
    }

    #[tokio::test]
    async fn test_create_session_with_creator() {
        let config = SessionConfig::builder()
            .title("Test Session")
            .content_id("video-123")
            .build()
            .expect("valid config");

        let creator = User {
            id: "user-1".to_string(),
            name: "John Doe".to_string(),
            email: "john@example.com".to_string(),
            role: UserRole::Owner,
        };

        let session = create_session_with_creator(config, creator)
            .await
            .expect("should succeed in test");
        assert_eq!(session.creator.id, "user-1");
    }

    #[test]
    fn test_validate_config() {
        let valid_config = SessionConfig::builder()
            .title("Test")
            .content_id("video-123")
            .build()
            .expect("valid config");

        assert!(validate_config(&valid_config).is_ok());

        // build() itself now validates: empty title yields an error
        let invalid_result = SessionConfig::builder()
            .title("")
            .content_id("video-123")
            .build();
        assert!(invalid_result.is_err());
    }
}
