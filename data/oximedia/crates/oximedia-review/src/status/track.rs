//! Status tracking.

use crate::{
    error::ReviewResult,
    status::{ReviewStatus, StatusMetrics},
    SessionId,
};
use std::collections::HashMap;

/// Status tracker for monitoring review progress.
pub struct StatusTracker {
    metrics: HashMap<SessionId, StatusMetrics>,
}

impl StatusTracker {
    /// Create a new status tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Track metrics for a session.
    pub fn track(&mut self, metrics: StatusMetrics) {
        self.metrics.insert(metrics.session_id, metrics);
    }

    /// Get metrics for a session.
    #[must_use]
    pub fn get_metrics(&self, session_id: SessionId) -> Option<&StatusMetrics> {
        self.metrics.get(&session_id)
    }

    /// Update session status.
    pub fn update_status(&mut self, session_id: SessionId, status: ReviewStatus) {
        if let Some(metrics) = self.metrics.get_mut(&session_id) {
            metrics.status = status;
        }
    }

    /// Get all sessions with a specific status.
    #[must_use]
    pub fn sessions_with_status(&self, status: ReviewStatus) -> Vec<SessionId> {
        self.metrics
            .iter()
            .filter(|(_, m)| m.status == status)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Count sessions by status.
    #[must_use]
    pub fn count_by_status(&self, status: ReviewStatus) -> usize {
        self.metrics.values().filter(|m| m.status == status).count()
    }
}

impl Default for StatusTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Update status metrics for a session.
///
/// # Errors
///
/// Returns error if update fails.
pub async fn update_metrics(
    session_id: SessionId,
    total_comments: usize,
    unresolved_comments: usize,
    total_tasks: usize,
    completed_tasks: usize,
) -> ReviewResult<StatusMetrics> {
    Ok(StatusMetrics {
        session_id,
        status: ReviewStatus::InProgress,
        total_comments,
        unresolved_comments,
        total_tasks,
        completed_tasks,
        approval_progress: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_tracker_creation() {
        let tracker = StatusTracker::new();
        assert_eq!(tracker.metrics.len(), 0);
    }

    #[test]
    fn test_status_tracker_track() {
        let mut tracker = StatusTracker::new();
        let session_id = SessionId::new();

        let metrics = StatusMetrics {
            session_id,
            status: ReviewStatus::InProgress,
            total_comments: 5,
            unresolved_comments: 2,
            total_tasks: 3,
            completed_tasks: 1,
            approval_progress: 0.5,
        };

        tracker.track(metrics);

        assert!(tracker.get_metrics(session_id).is_some());
    }

    #[test]
    fn test_status_tracker_update_status() {
        let mut tracker = StatusTracker::new();
        let session_id = SessionId::new();

        let metrics = StatusMetrics {
            session_id,
            status: ReviewStatus::InProgress,
            total_comments: 0,
            unresolved_comments: 0,
            total_tasks: 0,
            completed_tasks: 0,
            approval_progress: 0.0,
        };

        tracker.track(metrics);
        tracker.update_status(session_id, ReviewStatus::Approved);

        let updated = tracker
            .get_metrics(session_id)
            .expect("should succeed in test");
        assert_eq!(updated.status, ReviewStatus::Approved);
    }

    #[test]
    fn test_sessions_with_status() {
        let mut tracker = StatusTracker::new();

        let session1 = SessionId::new();
        let session2 = SessionId::new();

        tracker.track(StatusMetrics {
            session_id: session1,
            status: ReviewStatus::InProgress,
            total_comments: 0,
            unresolved_comments: 0,
            total_tasks: 0,
            completed_tasks: 0,
            approval_progress: 0.0,
        });

        tracker.track(StatusMetrics {
            session_id: session2,
            status: ReviewStatus::Approved,
            total_comments: 0,
            unresolved_comments: 0,
            total_tasks: 0,
            completed_tasks: 0,
            approval_progress: 1.0,
        });

        let in_progress = tracker.sessions_with_status(ReviewStatus::InProgress);
        assert_eq!(in_progress.len(), 1);

        let approved = tracker.sessions_with_status(ReviewStatus::Approved);
        assert_eq!(approved.len(), 1);
    }

    #[test]
    fn test_count_by_status() {
        let mut tracker = StatusTracker::new();

        for _ in 0..3 {
            tracker.track(StatusMetrics {
                session_id: SessionId::new(),
                status: ReviewStatus::InProgress,
                total_comments: 0,
                unresolved_comments: 0,
                total_tasks: 0,
                completed_tasks: 0,
                approval_progress: 0.0,
            });
        }

        assert_eq!(tracker.count_by_status(ReviewStatus::InProgress), 3);
    }
}
