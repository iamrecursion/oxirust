//! Multi-stage approval.

use crate::User;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Approval stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalStage {
    /// Stage index.
    pub index: usize,
    /// Stage name.
    pub name: String,
    /// Required approvers.
    pub approvers: Vec<User>,
    /// Status.
    pub status: StageStatus,
    /// Minimum approvals required.
    pub min_approvals: usize,
    /// Current approval count.
    pub approval_count: usize,
    /// Started timestamp.
    pub started_at: Option<DateTime<Utc>>,
    /// Completed timestamp.
    pub completed_at: Option<DateTime<Utc>>,
}

/// Status of an approval stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageStatus {
    /// Stage not started.
    NotStarted,
    /// Stage in progress.
    InProgress,
    /// Stage completed.
    Completed,
    /// Stage rejected.
    Rejected,
    /// Stage skipped.
    Skipped,
}

impl ApprovalStage {
    /// Create a new approval stage.
    #[must_use]
    pub fn new(index: usize, name: String) -> Self {
        Self {
            index,
            name,
            approvers: Vec::new(),
            status: StageStatus::NotStarted,
            min_approvals: 1,
            approval_count: 0,
            started_at: None,
            completed_at: None,
        }
    }

    /// Add an approver to the stage.
    pub fn add_approver(&mut self, user: User) {
        self.approvers.push(user);
    }

    /// Set minimum approvals required.
    pub fn set_min_approvals(&mut self, min: usize) {
        self.min_approvals = min;
    }

    /// Start the stage.
    pub fn start(&mut self) {
        self.status = StageStatus::InProgress;
        self.started_at = Some(Utc::now());
    }

    /// Record an approval.
    pub fn record_approval(&mut self) {
        self.approval_count += 1;
        if self.approval_count >= self.min_approvals {
            self.complete();
        }
    }

    /// Complete the stage.
    pub fn complete(&mut self) {
        self.status = StageStatus::Completed;
        self.completed_at = Some(Utc::now());
    }

    /// Reject the stage.
    pub fn reject(&mut self) {
        self.status = StageStatus::Rejected;
        self.completed_at = Some(Utc::now());
    }

    /// Check if stage is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.status == StageStatus::Completed
    }

    /// Check if stage requires all approvers.
    #[must_use]
    pub fn requires_all_approvers(&self) -> bool {
        self.min_approvals >= self.approvers.len()
    }

    /// Get approval progress percentage.
    #[must_use]
    pub fn progress_percentage(&self) -> f64 {
        if self.min_approvals == 0 {
            return 100.0;
        }

        (self.approval_count as f64 / self.min_approvals as f64 * 100.0).min(100.0)
    }
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

    #[test]
    fn test_stage_creation() {
        let stage = ApprovalStage::new(0, "Review".to_string());
        assert_eq!(stage.index, 0);
        assert_eq!(stage.status, StageStatus::NotStarted);
        assert_eq!(stage.approval_count, 0);
    }

    #[test]
    fn test_stage_add_approver() {
        let mut stage = ApprovalStage::new(0, "Review".to_string());
        stage.add_approver(create_test_user("user1"));
        assert_eq!(stage.approvers.len(), 1);
    }

    #[test]
    fn test_stage_start() {
        let mut stage = ApprovalStage::new(0, "Review".to_string());
        stage.start();
        assert_eq!(stage.status, StageStatus::InProgress);
        assert!(stage.started_at.is_some());
    }

    #[test]
    fn test_stage_record_approval() {
        let mut stage = ApprovalStage::new(0, "Review".to_string());
        stage.set_min_approvals(2);
        stage.start();

        assert_eq!(stage.approval_count, 0);

        stage.record_approval();
        assert_eq!(stage.approval_count, 1);
        assert_eq!(stage.status, StageStatus::InProgress);

        stage.record_approval();
        assert_eq!(stage.approval_count, 2);
        assert_eq!(stage.status, StageStatus::Completed);
    }

    #[test]
    fn test_stage_progress() {
        let mut stage = ApprovalStage::new(0, "Review".to_string());
        stage.set_min_approvals(4);

        assert!((stage.progress_percentage() - 0.0).abs() < 0.001);

        stage.record_approval();
        assert!((stage.progress_percentage() - 25.0).abs() < 0.001);

        stage.record_approval();
        assert!((stage.progress_percentage() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_stage_requires_all_approvers() {
        let mut stage = ApprovalStage::new(0, "Review".to_string());
        stage.add_approver(create_test_user("user1"));
        stage.add_approver(create_test_user("user2"));
        stage.set_min_approvals(2);

        assert!(stage.requires_all_approvers());

        stage.set_min_approvals(1);
        assert!(!stage.requires_all_approvers());
    }
}
