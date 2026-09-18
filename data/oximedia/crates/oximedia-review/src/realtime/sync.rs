//! Real-time synchronization.

use crate::SessionId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Sync message for real-time updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMessage {
    /// Message ID.
    pub id: String,
    /// Session ID.
    pub session_id: SessionId,
    /// Message type.
    pub message_type: SyncMessageType,
    /// Message payload.
    pub payload: String,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

/// Type of sync message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncMessageType {
    /// Comment update.
    CommentUpdate,
    /// Drawing update.
    DrawingUpdate,
    /// Status update.
    StatusUpdate,
    /// User action.
    UserAction,
    /// Cursor update.
    CursorUpdate,
    /// Presence update.
    PresenceUpdate,
}

impl SyncMessage {
    /// Create a new sync message.
    #[must_use]
    pub fn new(session_id: SessionId, message_type: SyncMessageType, payload: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id,
            message_type,
            payload,
            timestamp: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_message_creation() {
        let session_id = SessionId::new();
        let message =
            SyncMessage::new(session_id, SyncMessageType::CommentUpdate, "{}".to_string());

        assert_eq!(message.session_id, session_id);
        assert_eq!(message.message_type, SyncMessageType::CommentUpdate);
    }

    #[test]
    fn test_sync_message_types() {
        assert_eq!(
            SyncMessageType::CommentUpdate,
            SyncMessageType::CommentUpdate
        );
        assert_ne!(
            SyncMessageType::CommentUpdate,
            SyncMessageType::DrawingUpdate
        );
    }
}
