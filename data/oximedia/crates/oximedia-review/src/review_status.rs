#![allow(dead_code)]
//! Review status tracking, state transitions, and status aggregation.
//!
//! This module provides fine-grained review status management with support
//! for status history, transition validation, and aggregation across multiple
//! reviewers and review stages.

use std::collections::HashMap;
use std::fmt;

/// The current status of a review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewStatus {
    /// Review has been created but not yet started.
    Draft,
    /// Review is open and awaiting feedback.
    Open,
    /// Review is actively being worked on by reviewers.
    InProgress,
    /// Changes have been requested by a reviewer.
    ChangesRequested,
    /// Review has been approved by all required reviewers.
    Approved,
    /// Review has been rejected.
    Rejected,
    /// Review has been closed without a decision.
    Closed,
    /// Review has been archived.
    Archived,
}

impl fmt::Display for ReviewStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Draft => write!(f, "Draft"),
            Self::Open => write!(f, "Open"),
            Self::InProgress => write!(f, "In Progress"),
            Self::ChangesRequested => write!(f, "Changes Requested"),
            Self::Approved => write!(f, "Approved"),
            Self::Rejected => write!(f, "Rejected"),
            Self::Closed => write!(f, "Closed"),
            Self::Archived => write!(f, "Archived"),
        }
    }
}

impl ReviewStatus {
    /// Returns all valid transitions from the current status.
    #[must_use]
    pub fn valid_transitions(&self) -> &[ReviewStatus] {
        match self {
            Self::Draft => &[Self::Open, Self::Closed],
            Self::Open => &[Self::InProgress, Self::Closed],
            Self::InProgress => &[
                Self::ChangesRequested,
                Self::Approved,
                Self::Rejected,
                Self::Closed,
            ],
            Self::ChangesRequested => &[Self::InProgress, Self::Closed],
            Self::Approved => &[Self::Archived, Self::Closed],
            Self::Rejected => &[Self::InProgress, Self::Archived, Self::Closed],
            Self::Closed => &[Self::Archived],
            Self::Archived => &[],
        }
    }

    /// Check whether a transition to the given status is valid.
    #[must_use]
    pub fn can_transition_to(&self, target: ReviewStatus) -> bool {
        self.valid_transitions().contains(&target)
    }

    /// Returns true if this status represents a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Archived)
    }

    /// Returns true if this status represents an active review.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Open | Self::InProgress | Self::ChangesRequested)
    }

    /// Returns true if this status represents a resolved review.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        matches!(
            self,
            Self::Approved | Self::Rejected | Self::Closed | Self::Archived
        )
    }
}

/// A single status transition event.
#[derive(Debug, Clone)]
pub struct StatusTransition {
    /// The status before the transition.
    pub from: ReviewStatus,
    /// The status after the transition.
    pub to: ReviewStatus,
    /// The user who triggered the transition.
    pub changed_by: String,
    /// Optional reason for the transition.
    pub reason: Option<String>,
    /// Timestamp of the transition in milliseconds since epoch.
    pub timestamp_ms: u64,
}

/// Tracks the full history of status changes for a review.
#[derive(Debug, Clone)]
pub struct StatusHistory {
    /// The current status.
    current: ReviewStatus,
    /// Ordered list of transitions.
    transitions: Vec<StatusTransition>,
}

impl StatusHistory {
    /// Create a new status history starting from `Draft`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: ReviewStatus::Draft,
            transitions: Vec::new(),
        }
    }

    /// Create a new status history starting from the given status.
    #[must_use]
    pub fn with_initial(status: ReviewStatus) -> Self {
        Self {
            current: status,
            transitions: Vec::new(),
        }
    }

    /// Get the current status.
    #[must_use]
    pub fn current(&self) -> ReviewStatus {
        self.current
    }

    /// Get the full list of transitions.
    #[must_use]
    pub fn transitions(&self) -> &[StatusTransition] {
        &self.transitions
    }

    /// Get the number of transitions that have occurred.
    #[must_use]
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Attempt to transition to a new status.
    ///
    /// Returns `Ok(())` if the transition is valid, or an error message otherwise.
    pub fn transition(
        &mut self,
        to: ReviewStatus,
        changed_by: &str,
        reason: Option<&str>,
        timestamp_ms: u64,
    ) -> Result<(), String> {
        if !self.current.can_transition_to(to) {
            return Err(format!(
                "Invalid transition from {} to {}",
                self.current, to
            ));
        }

        let transition = StatusTransition {
            from: self.current,
            to,
            changed_by: changed_by.to_string(),
            reason: reason.map(String::from),
            timestamp_ms,
        };

        self.transitions.push(transition);
        self.current = to;
        Ok(())
    }

    /// Get the time spent in each status (in milliseconds).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn time_in_each_status(&self, current_time_ms: u64) -> HashMap<ReviewStatus, u64> {
        let mut durations: HashMap<ReviewStatus, u64> = HashMap::new();

        if self.transitions.is_empty() {
            durations.insert(self.current, current_time_ms);
            return durations;
        }

        // Time from start to first transition
        let first_ts = self.transitions[0].timestamp_ms;
        *durations.entry(self.transitions[0].from).or_insert(0) += first_ts;

        // Time between consecutive transitions
        for window in self.transitions.windows(2) {
            let duration = window[1]
                .timestamp_ms
                .saturating_sub(window[0].timestamp_ms);
            *durations.entry(window[0].to).or_insert(0) += duration;
        }

        // Time from last transition to now
        if let Some(last) = self.transitions.last() {
            let duration = current_time_ms.saturating_sub(last.timestamp_ms);
            *durations.entry(last.to).or_insert(0) += duration;
        }

        durations
    }

    /// Get the last transition, if any.
    #[must_use]
    pub fn last_transition(&self) -> Option<&StatusTransition> {
        self.transitions.last()
    }
}

impl Default for StatusHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregates review statuses across multiple reviewers.
#[derive(Debug, Clone)]
pub struct StatusAggregator {
    /// Map of reviewer ID to their current status.
    reviewer_statuses: HashMap<String, ReviewStatus>,
    /// Minimum number of approvals required.
    required_approvals: usize,
}

impl StatusAggregator {
    /// Create a new status aggregator.
    #[must_use]
    pub fn new(required_approvals: usize) -> Self {
        Self {
            reviewer_statuses: HashMap::new(),
            required_approvals: required_approvals.max(1),
        }
    }

    /// Set the status for a reviewer.
    pub fn set_reviewer_status(&mut self, reviewer_id: &str, status: ReviewStatus) {
        self.reviewer_statuses
            .insert(reviewer_id.to_string(), status);
    }

    /// Get the status of a specific reviewer.
    #[must_use]
    pub fn get_reviewer_status(&self, reviewer_id: &str) -> Option<ReviewStatus> {
        self.reviewer_statuses.get(reviewer_id).copied()
    }

    /// Get the number of reviewers who have approved.
    #[must_use]
    pub fn approval_count(&self) -> usize {
        self.reviewer_statuses
            .values()
            .filter(|s| **s == ReviewStatus::Approved)
            .count()
    }

    /// Get the number of reviewers who have rejected.
    #[must_use]
    pub fn rejection_count(&self) -> usize {
        self.reviewer_statuses
            .values()
            .filter(|s| **s == ReviewStatus::Rejected)
            .count()
    }

    /// Get the total number of reviewers.
    #[must_use]
    pub fn reviewer_count(&self) -> usize {
        self.reviewer_statuses.len()
    }

    /// Check whether the overall review is approved (enough approvals, no rejections).
    #[must_use]
    pub fn is_approved(&self) -> bool {
        self.rejection_count() == 0 && self.approval_count() >= self.required_approvals
    }

    /// Check whether the overall review is rejected (any rejection).
    #[must_use]
    pub fn is_rejected(&self) -> bool {
        self.rejection_count() > 0
    }

    /// Check whether all reviewers have submitted their status.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.reviewer_statuses.values().all(|s| s.is_resolved())
    }

    /// Compute the aggregate status across all reviewers.
    #[must_use]
    pub fn aggregate_status(&self) -> ReviewStatus {
        if self.reviewer_statuses.is_empty() {
            return ReviewStatus::Draft;
        }

        if self.is_rejected() {
            return ReviewStatus::Rejected;
        }

        if self.is_approved() {
            return ReviewStatus::Approved;
        }

        let has_changes_requested = self
            .reviewer_statuses
            .values()
            .any(|s| *s == ReviewStatus::ChangesRequested);

        if has_changes_requested {
            return ReviewStatus::ChangesRequested;
        }

        ReviewStatus::InProgress
    }

    /// Get the required number of approvals.
    #[must_use]
    pub fn required_approvals(&self) -> usize {
        self.required_approvals
    }

    /// Get the number of remaining approvals needed.
    #[must_use]
    pub fn remaining_approvals(&self) -> usize {
        self.required_approvals
            .saturating_sub(self.approval_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_review_status_display() {
        assert_eq!(ReviewStatus::Draft.to_string(), "Draft");
        assert_eq!(ReviewStatus::InProgress.to_string(), "In Progress");
        assert_eq!(
            ReviewStatus::ChangesRequested.to_string(),
            "Changes Requested"
        );
        assert_eq!(ReviewStatus::Approved.to_string(), "Approved");
    }

    #[test]
    fn test_valid_transitions_from_draft() {
        let valid = ReviewStatus::Draft.valid_transitions();
        assert!(valid.contains(&ReviewStatus::Open));
        assert!(valid.contains(&ReviewStatus::Closed));
        assert!(!valid.contains(&ReviewStatus::Approved));
    }

    #[test]
    fn test_can_transition_to() {
        assert!(ReviewStatus::Draft.can_transition_to(ReviewStatus::Open));
        assert!(!ReviewStatus::Draft.can_transition_to(ReviewStatus::Approved));
        assert!(ReviewStatus::InProgress.can_transition_to(ReviewStatus::Approved));
        assert!(ReviewStatus::Archived.valid_transitions().is_empty());
    }

    #[test]
    fn test_is_terminal() {
        assert!(ReviewStatus::Archived.is_terminal());
        assert!(!ReviewStatus::Draft.is_terminal());
        assert!(!ReviewStatus::Approved.is_terminal());
    }

    #[test]
    fn test_is_active() {
        assert!(ReviewStatus::Open.is_active());
        assert!(ReviewStatus::InProgress.is_active());
        assert!(ReviewStatus::ChangesRequested.is_active());
        assert!(!ReviewStatus::Draft.is_active());
        assert!(!ReviewStatus::Approved.is_active());
    }

    #[test]
    fn test_is_resolved() {
        assert!(ReviewStatus::Approved.is_resolved());
        assert!(ReviewStatus::Rejected.is_resolved());
        assert!(ReviewStatus::Closed.is_resolved());
        assert!(ReviewStatus::Archived.is_resolved());
        assert!(!ReviewStatus::Open.is_resolved());
    }

    #[test]
    fn test_status_history_creation() {
        let history = StatusHistory::new();
        assert_eq!(history.current(), ReviewStatus::Draft);
        assert_eq!(history.transition_count(), 0);
    }

    #[test]
    fn test_status_history_valid_transition() {
        let mut history = StatusHistory::new();
        let result = history.transition(ReviewStatus::Open, "user1", Some("Starting review"), 1000);
        assert!(result.is_ok());
        assert_eq!(history.current(), ReviewStatus::Open);
        assert_eq!(history.transition_count(), 1);
    }

    #[test]
    fn test_status_history_invalid_transition() {
        let mut history = StatusHistory::new();
        let result = history.transition(ReviewStatus::Approved, "user1", None, 1000);
        assert!(result.is_err());
        assert_eq!(history.current(), ReviewStatus::Draft);
    }

    #[test]
    fn test_status_history_full_lifecycle() {
        let mut history = StatusHistory::new();
        assert!(history
            .transition(ReviewStatus::Open, "user1", None, 100)
            .is_ok());
        assert!(history
            .transition(ReviewStatus::InProgress, "user2", None, 200)
            .is_ok());
        assert!(history
            .transition(
                ReviewStatus::ChangesRequested,
                "user2",
                Some("Fix audio"),
                300
            )
            .is_ok());
        assert!(history
            .transition(ReviewStatus::InProgress, "user1", None, 400)
            .is_ok());
        assert!(history
            .transition(ReviewStatus::Approved, "user2", None, 500)
            .is_ok());
        assert_eq!(history.current(), ReviewStatus::Approved);
        assert_eq!(history.transition_count(), 5);
    }

    #[test]
    fn test_status_history_time_tracking() {
        let mut history = StatusHistory::new();
        assert!(history
            .transition(ReviewStatus::Open, "u1", None, 100)
            .is_ok());
        assert!(history
            .transition(ReviewStatus::InProgress, "u1", None, 300)
            .is_ok());
        let times = history.time_in_each_status(500);
        assert_eq!(*times.get(&ReviewStatus::Draft).unwrap_or(&0), 100);
        assert_eq!(*times.get(&ReviewStatus::Open).unwrap_or(&0), 200);
        assert_eq!(*times.get(&ReviewStatus::InProgress).unwrap_or(&0), 200);
    }

    #[test]
    fn test_status_aggregator_basic() {
        let mut agg = StatusAggregator::new(2);
        agg.set_reviewer_status("r1", ReviewStatus::Approved);
        agg.set_reviewer_status("r2", ReviewStatus::InProgress);
        assert_eq!(agg.approval_count(), 1);
        assert_eq!(agg.reviewer_count(), 2);
        assert!(!agg.is_approved());
    }

    #[test]
    fn test_status_aggregator_approved() {
        let mut agg = StatusAggregator::new(2);
        agg.set_reviewer_status("r1", ReviewStatus::Approved);
        agg.set_reviewer_status("r2", ReviewStatus::Approved);
        assert!(agg.is_approved());
        assert_eq!(agg.aggregate_status(), ReviewStatus::Approved);
        assert_eq!(agg.remaining_approvals(), 0);
    }

    #[test]
    fn test_status_aggregator_rejected() {
        let mut agg = StatusAggregator::new(2);
        agg.set_reviewer_status("r1", ReviewStatus::Approved);
        agg.set_reviewer_status("r2", ReviewStatus::Rejected);
        assert!(agg.is_rejected());
        assert_eq!(agg.aggregate_status(), ReviewStatus::Rejected);
    }

    #[test]
    fn test_status_aggregator_changes_requested() {
        let mut agg = StatusAggregator::new(2);
        agg.set_reviewer_status("r1", ReviewStatus::Approved);
        agg.set_reviewer_status("r2", ReviewStatus::ChangesRequested);
        assert_eq!(agg.aggregate_status(), ReviewStatus::ChangesRequested);
    }

    #[test]
    fn test_last_transition() {
        let mut history = StatusHistory::new();
        assert!(history.last_transition().is_none());
        assert!(history
            .transition(ReviewStatus::Open, "u1", Some("Opening"), 100)
            .is_ok());
        let last = history.last_transition().expect("should succeed in test");
        assert_eq!(last.from, ReviewStatus::Draft);
        assert_eq!(last.to, ReviewStatus::Open);
        assert_eq!(last.changed_by, "u1");
    }
}
