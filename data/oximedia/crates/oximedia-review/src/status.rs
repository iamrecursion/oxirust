//! Review status tracking.

use crate::{error::ReviewResult, SessionId};
use serde::{Deserialize, Serialize};

pub mod progress;
pub mod summary;
pub mod track;

pub use progress::ProgressTracker;
pub use summary::StatusSummary;
pub use track::StatusTracker;

/// Overall review status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReviewStatus {
    /// Not started.
    NotStarted,
    /// In progress.
    InProgress,
    /// Waiting for feedback.
    WaitingForFeedback,
    /// Changes requested.
    ChangesRequested,
    /// Under review.
    UnderReview,
    /// Approved.
    Approved,
    /// Rejected.
    Rejected,
    /// On hold.
    OnHold,
}

impl ReviewStatus {
    /// Check if status is final.
    #[must_use]
    pub fn is_final(self) -> bool {
        matches!(self, Self::Approved | Self::Rejected)
    }

    /// Check if status is active.
    #[must_use]
    pub fn is_active(self) -> bool {
        !matches!(self, Self::NotStarted | Self::Approved | Self::Rejected)
    }
}

/// Status metrics for a review session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusMetrics {
    /// Session ID.
    pub session_id: SessionId,
    /// Current status.
    pub status: ReviewStatus,
    /// Total comments.
    pub total_comments: usize,
    /// Unresolved comments.
    pub unresolved_comments: usize,
    /// Total tasks.
    pub total_tasks: usize,
    /// Completed tasks.
    pub completed_tasks: usize,
    /// Approval progress (0.0-1.0).
    pub approval_progress: f64,
}

impl StatusMetrics {
    /// Calculate overall progress percentage.
    #[must_use]
    pub fn overall_progress(&self) -> f64 {
        let comment_progress = if self.total_comments > 0 {
            (self.total_comments - self.unresolved_comments) as f64 / self.total_comments as f64
        } else {
            1.0
        };

        let task_progress = if self.total_tasks > 0 {
            self.completed_tasks as f64 / self.total_tasks as f64
        } else {
            1.0
        };

        ((comment_progress + task_progress + self.approval_progress) / 3.0) * 100.0
    }

    /// Check if all comments are resolved.
    #[must_use]
    pub fn all_comments_resolved(&self) -> bool {
        self.unresolved_comments == 0
    }

    /// Check if all tasks are complete.
    #[must_use]
    pub fn all_tasks_complete(&self) -> bool {
        self.total_tasks > 0 && self.completed_tasks == self.total_tasks
    }
}

/// Get status metrics for a session.
///
/// # Errors
///
/// Returns error if metrics cannot be retrieved.
pub async fn get_status_metrics(session_id: SessionId) -> ReviewResult<StatusMetrics> {
    // In a real implementation, this would aggregate data from database

    Ok(StatusMetrics {
        session_id,
        status: ReviewStatus::InProgress,
        total_comments: 0,
        unresolved_comments: 0,
        total_tasks: 0,
        completed_tasks: 0,
        approval_progress: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_review_status_is_final() {
        assert!(ReviewStatus::Approved.is_final());
        assert!(ReviewStatus::Rejected.is_final());
        assert!(!ReviewStatus::InProgress.is_final());
    }

    #[test]
    fn test_review_status_is_active() {
        assert!(ReviewStatus::InProgress.is_active());
        assert!(ReviewStatus::UnderReview.is_active());
        assert!(!ReviewStatus::Approved.is_active());
        assert!(!ReviewStatus::NotStarted.is_active());
    }

    #[test]
    fn test_status_metrics_overall_progress() {
        let metrics = StatusMetrics {
            session_id: SessionId::new(),
            status: ReviewStatus::InProgress,
            total_comments: 10,
            unresolved_comments: 5,
            total_tasks: 4,
            completed_tasks: 2,
            approval_progress: 0.5,
        };

        let progress = metrics.overall_progress();
        assert!(progress > 0.0 && progress <= 100.0);
    }

    #[test]
    fn test_status_metrics_checks() {
        let mut metrics = StatusMetrics {
            session_id: SessionId::new(),
            status: ReviewStatus::InProgress,
            total_comments: 5,
            unresolved_comments: 0,
            total_tasks: 3,
            completed_tasks: 3,
            approval_progress: 1.0,
        };

        assert!(metrics.all_comments_resolved());
        assert!(metrics.all_tasks_complete());

        metrics.unresolved_comments = 1;
        assert!(!metrics.all_comments_resolved());
    }

    #[tokio::test]
    async fn test_get_status_metrics() {
        let session_id = SessionId::new();
        let result = get_status_metrics(session_id).await;
        assert!(result.is_ok());
    }
}
