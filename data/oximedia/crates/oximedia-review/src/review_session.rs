//! Review session management with quorum logic and state tracking.

#![allow(dead_code)]

use std::collections::HashMap;

/// Role a reviewer can hold within a review session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewerRole {
    /// Full approver — their vote counts toward quorum and approval decisions.
    Approver,
    /// Standard reviewer — can comment but cannot formally approve.
    Reviewer,
    /// Stakeholder observer — read-only access, does not count in quorum.
    Observer,
    /// Lead reviewer — required to approve before others' votes are counted.
    Lead,
}

impl ReviewerRole {
    /// Returns `true` if this role has the right to cast a formal approval vote.
    #[must_use]
    pub fn can_approve(&self) -> bool {
        matches!(self, Self::Approver | Self::Lead)
    }

    /// Returns `true` if this role counts toward session quorum.
    #[must_use]
    pub fn counts_toward_quorum(&self) -> bool {
        matches!(self, Self::Approver | Self::Lead)
    }

    /// Human-readable label for the role.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Approver => "Approver",
            Self::Reviewer => "Reviewer",
            Self::Observer => "Observer",
            Self::Lead => "Lead",
        }
    }
}

/// Current state of a review session lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewSessionState {
    /// Session has been created but not yet opened for review.
    Draft,
    /// Review is actively in progress.
    InProgress,
    /// All required approvals have been received.
    Approved,
    /// Session was rejected — changes required.
    Rejected,
    /// Session was closed/cancelled without a decision.
    Closed,
    /// Session is on hold pending additional information.
    OnHold,
}

impl ReviewSessionState {
    /// Returns `true` if the session has reached a terminal state
    /// (no further state transitions are expected).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Approved | Self::Rejected | Self::Closed)
    }

    /// Returns `true` if participants can still act on the session.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Draft | Self::InProgress | Self::OnHold)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Draft => "Draft",
            Self::InProgress => "In Progress",
            Self::Approved => "Approved",
            Self::Rejected => "Rejected",
            Self::Closed => "Closed",
            Self::OnHold => "On Hold",
        }
    }
}

/// A single reviewer entry within a session.
#[derive(Debug, Clone)]
pub struct SessionReviewer {
    /// Unique reviewer identifier (e.g. user ID or email).
    pub id: String,
    /// Display name.
    pub name: String,
    /// Role assigned to this reviewer.
    pub role: ReviewerRole,
    /// Whether this reviewer has cast a vote.
    pub has_voted: bool,
    /// Whether their vote was an approval (`true`) or rejection (`false`).
    pub vote: Option<bool>,
}

impl SessionReviewer {
    /// Create a new reviewer entry with the given role.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, role: ReviewerRole) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            role,
            has_voted: false,
            vote: None,
        }
    }

    /// Record a vote for this reviewer.
    pub fn cast_vote(&mut self, approved: bool) {
        self.has_voted = true;
        self.vote = Some(approved);
    }
}

/// A review session that aggregates reviewers and tracks approval state.
#[derive(Debug, Clone)]
pub struct ReviewSession {
    /// Session identifier.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Current lifecycle state.
    pub state: ReviewSessionState,
    /// Minimum number of approver votes needed for quorum.
    pub quorum_threshold: usize,
    /// Map of reviewer ID → reviewer data.
    reviewers: HashMap<String, SessionReviewer>,
}

impl ReviewSession {
    /// Create a new review session.
    #[must_use]
    pub fn new(id: impl Into<String>, title: impl Into<String>, quorum_threshold: usize) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            state: ReviewSessionState::Draft,
            quorum_threshold,
            reviewers: HashMap::new(),
        }
    }

    /// Add a reviewer to the session.
    ///
    /// If a reviewer with the same ID already exists their entry is replaced.
    pub fn add_reviewer(&mut self, reviewer: SessionReviewer) {
        self.reviewers.insert(reviewer.id.clone(), reviewer);
    }

    /// Remove a reviewer by ID.
    pub fn remove_reviewer(&mut self, id: &str) {
        self.reviewers.remove(id);
    }

    /// Return a slice-view of all reviewers.
    #[must_use]
    pub fn reviewers(&self) -> Vec<&SessionReviewer> {
        self.reviewers.values().collect()
    }

    /// Count of approvers and leads who have voted *in favour*.
    #[must_use]
    pub fn approval_vote_count(&self) -> usize {
        self.reviewers
            .values()
            .filter(|r| r.role.can_approve() && r.vote == Some(true))
            .count()
    }

    /// Count of approvers and leads who have voted *against*.
    #[must_use]
    pub fn rejection_vote_count(&self) -> usize {
        self.reviewers
            .values()
            .filter(|r| r.role.can_approve() && r.vote == Some(false))
            .count()
    }

    /// Returns `true` when enough approvers have voted in favour to meet quorum.
    #[must_use]
    pub fn is_quorum(&self) -> bool {
        self.approval_vote_count() >= self.quorum_threshold
    }

    /// Total number of reviewers (all roles).
    #[must_use]
    pub fn reviewer_count(&self) -> usize {
        self.reviewers.len()
    }

    /// Transition the session to `InProgress`.
    ///
    /// Only valid from `Draft` state.
    pub fn start(&mut self) -> bool {
        if self.state == ReviewSessionState::Draft {
            self.state = ReviewSessionState::InProgress;
            true
        } else {
            false
        }
    }

    /// Attempt to approve the session.
    ///
    /// Succeeds only when quorum is met and the session is `InProgress`.
    pub fn approve(&mut self) -> bool {
        if self.state == ReviewSessionState::InProgress && self.is_quorum() {
            self.state = ReviewSessionState::Approved;
            true
        } else {
            false
        }
    }

    /// Reject the session.
    pub fn reject(&mut self) {
        if !self.state.is_terminal() {
            self.state = ReviewSessionState::Rejected;
        }
    }

    /// Place the session on hold.
    pub fn hold(&mut self) {
        if self.state == ReviewSessionState::InProgress {
            self.state = ReviewSessionState::OnHold;
        }
    }

    /// Resume from hold.
    pub fn resume(&mut self) {
        if self.state == ReviewSessionState::OnHold {
            self.state = ReviewSessionState::InProgress;
        }
    }
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(quorum: usize) -> ReviewSession {
        ReviewSession::new("sess-1", "Color Grade Review", quorum)
    }

    fn make_approver(id: &str) -> SessionReviewer {
        SessionReviewer::new(id, "Alice", ReviewerRole::Approver)
    }

    // 1 — ReviewerRole::can_approve
    #[test]
    fn test_approver_can_approve() {
        assert!(ReviewerRole::Approver.can_approve());
        assert!(ReviewerRole::Lead.can_approve());
        assert!(!ReviewerRole::Reviewer.can_approve());
        assert!(!ReviewerRole::Observer.can_approve());
    }

    // 2 — ReviewerRole::counts_toward_quorum
    #[test]
    fn test_counts_toward_quorum() {
        assert!(ReviewerRole::Approver.counts_toward_quorum());
        assert!(ReviewerRole::Lead.counts_toward_quorum());
        assert!(!ReviewerRole::Reviewer.counts_toward_quorum());
        assert!(!ReviewerRole::Observer.counts_toward_quorum());
    }

    // 3 — ReviewerRole::label
    #[test]
    fn test_reviewer_role_label() {
        assert_eq!(ReviewerRole::Approver.label(), "Approver");
        assert_eq!(ReviewerRole::Observer.label(), "Observer");
        assert_eq!(ReviewerRole::Lead.label(), "Lead");
    }

    // 4 — ReviewSessionState::is_terminal
    #[test]
    fn test_state_is_terminal() {
        assert!(ReviewSessionState::Approved.is_terminal());
        assert!(ReviewSessionState::Rejected.is_terminal());
        assert!(ReviewSessionState::Closed.is_terminal());
        assert!(!ReviewSessionState::Draft.is_terminal());
        assert!(!ReviewSessionState::InProgress.is_terminal());
        assert!(!ReviewSessionState::OnHold.is_terminal());
    }

    // 5 — ReviewSessionState::is_active
    #[test]
    fn test_state_is_active() {
        assert!(ReviewSessionState::Draft.is_active());
        assert!(ReviewSessionState::InProgress.is_active());
        assert!(ReviewSessionState::OnHold.is_active());
        assert!(!ReviewSessionState::Approved.is_active());
        assert!(!ReviewSessionState::Rejected.is_active());
    }

    // 6 — ReviewSession starts in Draft
    #[test]
    fn test_session_initial_state() {
        let s = make_session(2);
        assert_eq!(s.state, ReviewSessionState::Draft);
        assert_eq!(s.quorum_threshold, 2);
    }

    // 7 — add_reviewer / reviewer_count
    #[test]
    fn test_add_reviewer() {
        let mut s = make_session(1);
        s.add_reviewer(make_approver("u1"));
        s.add_reviewer(SessionReviewer::new("u2", "Bob", ReviewerRole::Reviewer));
        assert_eq!(s.reviewer_count(), 2);
    }

    // 8 — is_quorum false when no votes
    #[test]
    fn test_no_quorum_without_votes() {
        let mut s = make_session(1);
        s.add_reviewer(make_approver("u1"));
        assert!(!s.is_quorum());
    }

    // 9 — is_quorum true after approval votes
    #[test]
    fn test_quorum_reached_after_votes() {
        let mut s = make_session(2);
        let mut r1 = make_approver("u1");
        let mut r2 = make_approver("u2");
        r1.cast_vote(true);
        r2.cast_vote(true);
        s.add_reviewer(r1);
        s.add_reviewer(r2);
        assert!(s.is_quorum());
    }

    // 10 — rejection votes do not satisfy quorum
    #[test]
    fn test_rejection_votes_do_not_count_for_quorum() {
        let mut s = make_session(1);
        let mut r = make_approver("u1");
        r.cast_vote(false);
        s.add_reviewer(r);
        assert!(!s.is_quorum());
        assert_eq!(s.rejection_vote_count(), 1);
    }

    // 11 — session lifecycle: draft → in_progress → approved
    #[test]
    fn test_session_approve_lifecycle() {
        let mut s = make_session(1);
        let mut r = make_approver("u1");
        r.cast_vote(true);
        s.add_reviewer(r);
        assert!(s.start());
        assert!(s.approve());
        assert_eq!(s.state, ReviewSessionState::Approved);
        assert!(s.state.is_terminal());
    }

    // 12 — approve fails without quorum
    #[test]
    fn test_approve_fails_without_quorum() {
        let mut s = make_session(2);
        s.add_reviewer(make_approver("u1")); // no vote cast
        s.start();
        assert!(!s.approve());
        assert_eq!(s.state, ReviewSessionState::InProgress);
    }

    // 13 — hold and resume
    #[test]
    fn test_hold_resume() {
        let mut s = make_session(1);
        s.start();
        s.hold();
        assert_eq!(s.state, ReviewSessionState::OnHold);
        s.resume();
        assert_eq!(s.state, ReviewSessionState::InProgress);
    }

    // 14 — reject transitions to terminal
    #[test]
    fn test_reject() {
        let mut s = make_session(1);
        s.start();
        s.reject();
        assert_eq!(s.state, ReviewSessionState::Rejected);
        assert!(s.state.is_terminal());
    }

    // 15 — cannot re-start from approved
    #[test]
    fn test_cannot_start_from_terminal() {
        let mut s = make_session(1);
        let mut r = make_approver("u1");
        r.cast_vote(true);
        s.add_reviewer(r);
        s.start();
        s.approve();
        // start() should return false when already terminal
        assert!(!s.start());
    }
}
