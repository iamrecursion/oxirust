//! Frame-accurate commenting system.

use crate::{AnnotationType, CommentId, SessionId, User};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod add;
pub mod pagination;
pub mod reply;
pub mod resolve;
pub mod thread;

pub use add::add_comment;
pub use pagination::{cursor_page, paginate_comments, CommentPage, CommentSortOrder, PageRequest};
pub use reply::add_reply;
pub use resolve::{resolve_comment, unresolve_comment};
pub use thread::{create_thread, CommentThread};

/// Status of a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommentStatus {
    /// Comment is open.
    Open,
    /// Comment is resolved.
    Resolved,
    /// Comment is archived.
    Archived,
}

/// Priority level for a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommentPriority {
    /// Low priority.
    Low,
    /// Normal priority.
    Normal,
    /// High priority.
    High,
    /// Critical priority.
    Critical,
}

/// A frame-accurate comment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    /// Comment ID.
    pub id: CommentId,
    /// Session ID.
    pub session_id: SessionId,
    /// Frame number (0-indexed).
    pub frame: i64,
    /// Comment text.
    pub text: String,
    /// Type of annotation.
    pub annotation_type: AnnotationType,
    /// Comment author.
    pub author: User,
    /// Comment status.
    pub status: CommentStatus,
    /// Priority level.
    pub priority: CommentPriority,
    /// Parent comment ID (for replies).
    pub parent_id: Option<CommentId>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
    /// Resolution timestamp.
    pub resolved_at: Option<DateTime<Utc>>,
    /// User who resolved the comment.
    pub resolved_by: Option<String>,
}

impl Comment {
    /// Check if comment is resolved.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.status == CommentStatus::Resolved
    }

    /// Check if comment is a reply.
    #[must_use]
    pub fn is_reply(&self) -> bool {
        self.parent_id.is_some()
    }

    /// Check if comment is high priority.
    #[must_use]
    pub fn is_high_priority(&self) -> bool {
        matches!(
            self.priority,
            CommentPriority::High | CommentPriority::Critical
        )
    }
}

/// Comment with timecode information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimecodeComment {
    /// Base comment.
    pub comment: Comment,
    /// Timecode string (e.g., "01:23:45:12").
    pub timecode: String,
    /// Frame rate.
    pub frame_rate: f64,
}

impl TimecodeComment {
    /// Create a new timecode comment.
    #[must_use]
    pub fn new(comment: Comment, timecode: String, frame_rate: f64) -> Self {
        Self {
            comment,
            timecode,
            frame_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UserRole;

    fn create_test_comment() -> Comment {
        Comment {
            id: CommentId::new(),
            session_id: SessionId::new(),
            frame: 100,
            text: "Test comment".to_string(),
            annotation_type: AnnotationType::Issue,
            author: User {
                id: "user-1".to_string(),
                name: "Test User".to_string(),
                email: "test@example.com".to_string(),
                role: UserRole::Reviewer,
            },
            status: CommentStatus::Open,
            priority: CommentPriority::Normal,
            parent_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            resolved_at: None,
            resolved_by: None,
        }
    }

    #[test]
    fn test_comment_is_resolved() {
        let mut comment = create_test_comment();
        assert!(!comment.is_resolved());

        comment.status = CommentStatus::Resolved;
        assert!(comment.is_resolved());
    }

    #[test]
    fn test_comment_is_reply() {
        let mut comment = create_test_comment();
        assert!(!comment.is_reply());

        comment.parent_id = Some(CommentId::new());
        assert!(comment.is_reply());
    }

    #[test]
    fn test_comment_is_high_priority() {
        let mut comment = create_test_comment();
        assert!(!comment.is_high_priority());

        comment.priority = CommentPriority::High;
        assert!(comment.is_high_priority());

        comment.priority = CommentPriority::Critical;
        assert!(comment.is_high_priority());
    }

    #[test]
    fn test_comment_priority_ordering() {
        assert!(CommentPriority::Low < CommentPriority::Normal);
        assert!(CommentPriority::Normal < CommentPriority::High);
        assert!(CommentPriority::High < CommentPriority::Critical);
    }
}
