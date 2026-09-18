//! Approval decision handling.

use crate::approval::ApprovalId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Approval decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecision {
    /// Decision ID.
    pub id: ApprovalId,
    /// Approver user ID.
    pub approver_id: String,
    /// Decision type.
    pub decision_type: DecisionType,
    /// Comments.
    pub comments: Option<String>,
    /// Conditions (if conditional approval).
    pub conditions: Vec<String>,
    /// Decision timestamp.
    pub decided_at: DateTime<Utc>,
}

/// Type of approval decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionType {
    /// Approve unconditionally.
    Approve,
    /// Reject.
    Reject,
    /// Approve with conditions.
    ApproveWithConditions,
    /// Request changes.
    RequestChanges,
    /// Abstain from decision.
    Abstain,
}

impl DecisionType {
    /// Check if decision is a form of approval.
    #[must_use]
    pub fn is_approval(self) -> bool {
        matches!(self, Self::Approve | Self::ApproveWithConditions)
    }

    /// Check if decision is a rejection.
    #[must_use]
    pub fn is_rejection(self) -> bool {
        matches!(self, Self::Reject | Self::RequestChanges)
    }
}

impl ApprovalDecision {
    /// Create a new approval decision.
    #[must_use]
    pub fn new(approver_id: String, decision_type: DecisionType) -> Self {
        Self {
            id: ApprovalId::new(),
            approver_id,
            decision_type,
            comments: None,
            conditions: Vec::new(),
            decided_at: Utc::now(),
        }
    }

    /// Add comments to the decision.
    #[must_use]
    pub fn with_comments(mut self, comments: impl Into<String>) -> Self {
        self.comments = Some(comments.into());
        self
    }

    /// Add a condition to the decision.
    #[must_use]
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.conditions.push(condition.into());
        self
    }

    /// Check if decision has conditions.
    #[must_use]
    pub fn has_conditions(&self) -> bool {
        !self.conditions.is_empty()
    }
}

/// Decision summary for a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionSummary {
    /// Total decisions.
    pub total: usize,
    /// Number of approvals.
    pub approvals: usize,
    /// Number of rejections.
    pub rejections: usize,
    /// Number of conditional approvals.
    pub conditional: usize,
    /// Number of abstentions.
    pub abstentions: usize,
}

impl DecisionSummary {
    /// Create a new decision summary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total: 0,
            approvals: 0,
            rejections: 0,
            conditional: 0,
            abstentions: 0,
        }
    }

    /// Add a decision to the summary.
    pub fn add_decision(&mut self, decision: &ApprovalDecision) {
        self.total += 1;
        match decision.decision_type {
            DecisionType::Approve => self.approvals += 1,
            DecisionType::Reject | DecisionType::RequestChanges => self.rejections += 1,
            DecisionType::ApproveWithConditions => self.conditional += 1,
            DecisionType::Abstain => self.abstentions += 1,
        }
    }

    /// Calculate approval rate.
    #[must_use]
    pub fn approval_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }

        ((self.approvals + self.conditional) as f64 / self.total as f64) * 100.0
    }

    /// Check if majority approved.
    #[must_use]
    pub fn has_majority_approval(&self) -> bool {
        (self.approvals + self.conditional) > self.rejections
    }
}

impl Default for DecisionSummary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_creation() {
        let decision = ApprovalDecision::new("user-1".to_string(), DecisionType::Approve);
        assert_eq!(decision.decision_type, DecisionType::Approve);
        assert!(decision.comments.is_none());
        assert!(!decision.has_conditions());
    }

    #[test]
    fn test_decision_with_comments() {
        let decision = ApprovalDecision::new("user-1".to_string(), DecisionType::Approve)
            .with_comments("Looks good!");

        assert_eq!(decision.comments, Some("Looks good!".to_string()));
    }

    #[test]
    fn test_decision_with_conditions() {
        let decision =
            ApprovalDecision::new("user-1".to_string(), DecisionType::ApproveWithConditions)
                .with_condition("Fix audio levels")
                .with_condition("Adjust color grading");

        assert!(decision.has_conditions());
        assert_eq!(decision.conditions.len(), 2);
    }

    #[test]
    fn test_decision_type_is_approval() {
        assert!(DecisionType::Approve.is_approval());
        assert!(DecisionType::ApproveWithConditions.is_approval());
        assert!(!DecisionType::Reject.is_approval());
    }

    #[test]
    fn test_decision_type_is_rejection() {
        assert!(DecisionType::Reject.is_rejection());
        assert!(DecisionType::RequestChanges.is_rejection());
        assert!(!DecisionType::Approve.is_rejection());
    }

    #[test]
    fn test_decision_summary() {
        let mut summary = DecisionSummary::new();

        let d1 = ApprovalDecision::new("user-1".to_string(), DecisionType::Approve);
        let d2 = ApprovalDecision::new("user-2".to_string(), DecisionType::Approve);
        let d3 = ApprovalDecision::new("user-3".to_string(), DecisionType::Reject);

        summary.add_decision(&d1);
        summary.add_decision(&d2);
        summary.add_decision(&d3);

        assert_eq!(summary.total, 3);
        assert_eq!(summary.approvals, 2);
        assert_eq!(summary.rejections, 1);
        assert!(summary.has_majority_approval());
        assert!((summary.approval_rate() - 66.666).abs() < 0.01);
    }
}
