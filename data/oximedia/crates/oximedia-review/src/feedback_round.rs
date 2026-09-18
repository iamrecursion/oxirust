//! Feedback round management for iterative media review.
//!
//! Manages multiple feedback rounds within a review session, tracking issues
//! raised per round, resolution status, round metrics, and round-over-round
//! progress reporting.

#![allow(dead_code)]

use std::collections::HashMap;

/// Status of a single feedback item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackStatus {
    /// Item has been raised but not yet acknowledged.
    Open,
    /// Reviewer has acknowledged the item.
    Acknowledged,
    /// Fix is in progress.
    InProgress,
    /// Fix has been applied and is awaiting verification.
    PendingVerification,
    /// Item has been resolved and verified.
    Resolved,
    /// Item was marked as not applicable.
    Rejected,
}

impl FeedbackStatus {
    /// Returns true if the item is in a terminal state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Resolved | Self::Rejected)
    }
}

/// Severity of a feedback item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Low-priority cosmetic issue.
    Minor = 1,
    /// Moderate issue that should be fixed.
    Moderate = 2,
    /// High-priority issue that must be fixed.
    Major = 3,
    /// Critical blocking issue.
    Critical = 4,
}

/// A single feedback item raised during a round.
#[derive(Debug, Clone)]
pub struct FeedbackItem {
    /// Unique identifier.
    pub id: u64,
    /// Frame the feedback refers to (None = global).
    pub frame: Option<u64>,
    /// Brief title.
    pub title: String,
    /// Detailed description.
    pub description: String,
    /// Severity.
    pub severity: Severity,
    /// Current status.
    pub status: FeedbackStatus,
    /// User ID of the person who raised the feedback.
    pub raised_by: String,
    /// User ID of the person assigned to resolve it.
    pub assignee: Option<String>,
    /// Timestamp raised in ms since epoch.
    pub raised_ms: u64,
    /// Timestamp resolved in ms since epoch.
    pub resolved_ms: Option<u64>,
    /// Labels / tags.
    pub labels: Vec<String>,
}

impl FeedbackItem {
    /// Create a new feedback item.
    #[must_use]
    pub fn new(
        id: u64,
        title: impl Into<String>,
        severity: Severity,
        raised_by: impl Into<String>,
        raised_ms: u64,
    ) -> Self {
        Self {
            id,
            frame: None,
            title: title.into(),
            description: String::new(),
            severity,
            status: FeedbackStatus::Open,
            raised_by: raised_by.into(),
            assignee: None,
            raised_ms,
            resolved_ms: None,
            labels: Vec::new(),
        }
    }

    /// Set frame reference.
    pub fn with_frame(mut self, frame: u64) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Assign to a user.
    pub fn assign_to(&mut self, user_id: impl Into<String>) {
        self.assignee = Some(user_id.into());
    }

    /// Transition to a new status.
    pub fn transition(&mut self, new_status: FeedbackStatus, now_ms: u64) {
        self.status = new_status;
        if new_status.is_terminal() {
            self.resolved_ms = Some(now_ms);
        }
    }

    /// Add a label.
    pub fn add_label(&mut self, label: impl Into<String>) {
        self.labels.push(label.into());
    }

    /// Time to resolve in milliseconds, if resolved.
    #[must_use]
    pub fn resolution_time_ms(&self) -> Option<u64> {
        self.resolved_ms.map(|r| r.saturating_sub(self.raised_ms))
    }
}

/// Summary statistics for a feedback round.
#[derive(Debug, Clone, Default)]
pub struct RoundStats {
    /// Total items raised.
    pub total: usize,
    /// Items still open (not terminal).
    pub open: usize,
    /// Items resolved.
    pub resolved: usize,
    /// Items rejected / N/A.
    pub rejected: usize,
    /// Number of critical items.
    pub critical: usize,
    /// Number of major items.
    pub major: usize,
}

impl RoundStats {
    /// Resolution rate as a fraction (0.0–1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn resolution_rate(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.resolved as f64 / self.total as f64
        }
    }

    /// Returns true if all items are terminal.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.open == 0
    }
}

/// A single feedback round within a review session.
#[derive(Debug)]
pub struct FeedbackRound {
    /// Round number (1-based).
    pub number: u32,
    /// Display label.
    pub label: String,
    /// Feedback items raised in this round.
    pub items: HashMap<u64, FeedbackItem>,
    /// Timestamp when the round opened.
    pub opened_ms: u64,
    /// Timestamp when the round closed, if it has been closed.
    pub closed_ms: Option<u64>,
    /// User who opened the round.
    pub opened_by: String,
}

impl FeedbackRound {
    /// Create a new feedback round.
    #[must_use]
    pub fn new(
        number: u32,
        label: impl Into<String>,
        opened_by: impl Into<String>,
        opened_ms: u64,
    ) -> Self {
        Self {
            number,
            label: label.into(),
            items: HashMap::new(),
            opened_ms,
            closed_ms: None,
            opened_by: opened_by.into(),
        }
    }

    /// Add a feedback item to this round.
    pub fn add_item(&mut self, item: FeedbackItem) {
        self.items.insert(item.id, item);
    }

    /// Close this round.
    pub fn close(&mut self, now_ms: u64) {
        self.closed_ms = Some(now_ms);
    }

    /// Returns true if the round is closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed_ms.is_some()
    }

    /// Compute summary statistics for this round.
    #[must_use]
    pub fn stats(&self) -> RoundStats {
        let mut stats = RoundStats::default();
        stats.total = self.items.len();
        for item in self.items.values() {
            match item.status {
                FeedbackStatus::Resolved => stats.resolved += 1,
                FeedbackStatus::Rejected => stats.rejected += 1,
                _ => stats.open += 1,
            }
            if item.severity == Severity::Critical {
                stats.critical += 1;
            }
            if item.severity == Severity::Major {
                stats.major += 1;
            }
        }
        stats
    }

    /// Duration of the round in milliseconds (open rounds use `now_ms`).
    #[must_use]
    pub fn duration_ms(&self, now_ms: u64) -> u64 {
        let end = self.closed_ms.unwrap_or(now_ms);
        end.saturating_sub(self.opened_ms)
    }
}

/// Manager for all feedback rounds in a review session.
#[derive(Debug, Default)]
pub struct FeedbackRoundManager {
    /// Ordered list of rounds (earliest first).
    pub rounds: Vec<FeedbackRound>,
}

impl FeedbackRoundManager {
    /// Create a new manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a new round.
    pub fn open_round(
        &mut self,
        label: impl Into<String>,
        opened_by: impl Into<String>,
        now_ms: u64,
    ) -> u32 {
        let number = self.rounds.len() as u32 + 1;
        self.rounds
            .push(FeedbackRound::new(number, label, opened_by, now_ms));
        number
    }

    /// Get the latest (most recently opened) round.
    #[must_use]
    pub fn latest(&self) -> Option<&FeedbackRound> {
        self.rounds.last()
    }

    /// Get the latest round mutably.
    #[must_use]
    pub fn latest_mut(&mut self) -> Option<&mut FeedbackRound> {
        self.rounds.last_mut()
    }

    /// Get a round by number (1-based).
    #[must_use]
    pub fn round(&self, number: u32) -> Option<&FeedbackRound> {
        self.rounds.iter().find(|r| r.number == number)
    }

    /// Get a round mutably by number (1-based).
    #[must_use]
    pub fn round_mut(&mut self, number: u32) -> Option<&mut FeedbackRound> {
        self.rounds.iter_mut().find(|r| r.number == number)
    }

    /// Total open items across all rounds.
    #[must_use]
    pub fn total_open(&self) -> usize {
        self.rounds.iter().map(|r| r.stats().open).sum()
    }

    /// Total resolved items across all rounds.
    #[must_use]
    pub fn total_resolved(&self) -> usize {
        self.rounds.iter().map(|r| r.stats().resolved).sum()
    }

    /// Overall resolution rate across all rounds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn overall_resolution_rate(&self) -> f64 {
        let total: usize = self.rounds.iter().map(|r| r.stats().total).sum();
        let resolved: usize = self.total_resolved();
        if total == 0 {
            1.0
        } else {
            resolved as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: u64, severity: Severity) -> FeedbackItem {
        FeedbackItem::new(id, format!("Item {id}"), severity, "alice", 1000)
    }

    #[test]
    fn test_feedback_status_terminal() {
        assert!(FeedbackStatus::Resolved.is_terminal());
        assert!(FeedbackStatus::Rejected.is_terminal());
        assert!(!FeedbackStatus::Open.is_terminal());
        assert!(!FeedbackStatus::InProgress.is_terminal());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::Major);
        assert!(Severity::Major > Severity::Moderate);
        assert!(Severity::Moderate > Severity::Minor);
    }

    #[test]
    fn test_feedback_item_creation() {
        let item = make_item(1, Severity::Major);
        assert_eq!(item.id, 1);
        assert_eq!(item.status, FeedbackStatus::Open);
        assert!(item.assignee.is_none());
    }

    #[test]
    fn test_feedback_item_with_frame() {
        let item = make_item(2, Severity::Minor).with_frame(500);
        assert_eq!(item.frame, Some(500));
    }

    #[test]
    fn test_feedback_item_transition_to_resolved() {
        let mut item = make_item(3, Severity::Critical);
        item.transition(FeedbackStatus::Resolved, 5000);
        assert_eq!(item.status, FeedbackStatus::Resolved);
        assert_eq!(item.resolved_ms, Some(5000));
    }

    #[test]
    fn test_feedback_item_resolution_time() {
        let mut item = make_item(4, Severity::Moderate);
        item.transition(FeedbackStatus::Resolved, 3000);
        assert_eq!(item.resolution_time_ms(), Some(2000));
    }

    #[test]
    fn test_feedback_item_add_label() {
        let mut item = make_item(5, Severity::Minor);
        item.add_label("color");
        item.add_label("grade");
        assert_eq!(item.labels.len(), 2);
    }

    #[test]
    fn test_round_stats_empty() {
        let round = FeedbackRound::new(1, "R1", "alice", 0);
        let stats = round.stats();
        assert_eq!(stats.total, 0);
        assert!(stats.is_clean());
        assert!((stats.resolution_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_round_stats_with_items() {
        let mut round = FeedbackRound::new(1, "R1", "alice", 0);
        round.add_item(make_item(1, Severity::Critical));
        let mut item2 = make_item(2, Severity::Major);
        item2.transition(FeedbackStatus::Resolved, 2000);
        round.add_item(item2);
        let stats = round.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.open, 1);
        assert_eq!(stats.resolved, 1);
        assert_eq!(stats.critical, 1);
        assert!(!stats.is_clean());
    }

    #[test]
    fn test_round_duration() {
        let mut round = FeedbackRound::new(1, "R1", "alice", 1000);
        round.close(6000);
        assert_eq!(round.duration_ms(9999), 5000);
    }

    #[test]
    fn test_round_manager_open_round() {
        let mut mgr = FeedbackRoundManager::new();
        let n = mgr.open_round("Round 1", "alice", 0);
        assert_eq!(n, 1);
        assert_eq!(mgr.rounds.len(), 1);
    }

    #[test]
    fn test_round_manager_latest() {
        let mut mgr = FeedbackRoundManager::new();
        mgr.open_round("R1", "alice", 0);
        mgr.open_round("R2", "bob", 1000);
        let latest = mgr.latest().expect("should succeed in test");
        assert_eq!(latest.number, 2);
    }

    #[test]
    fn test_round_manager_total_open_resolved() {
        let mut mgr = FeedbackRoundManager::new();
        mgr.open_round("R1", "alice", 0);
        if let Some(round) = mgr.latest_mut() {
            round.add_item(make_item(1, Severity::Major));
            let mut item2 = make_item(2, Severity::Minor);
            item2.transition(FeedbackStatus::Resolved, 2000);
            round.add_item(item2);
        }
        assert_eq!(mgr.total_open(), 1);
        assert_eq!(mgr.total_resolved(), 1);
    }

    #[test]
    fn test_round_manager_overall_resolution_rate() {
        let mut mgr = FeedbackRoundManager::new();
        mgr.open_round("R1", "alice", 0);
        if let Some(round) = mgr.latest_mut() {
            let mut item = make_item(1, Severity::Minor);
            item.transition(FeedbackStatus::Resolved, 1000);
            round.add_item(item);
            round.add_item(make_item(2, Severity::Minor));
        }
        let rate = mgr.overall_resolution_rate();
        assert!((rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_round_manager_find_by_number() {
        let mut mgr = FeedbackRoundManager::new();
        mgr.open_round("R1", "alice", 0);
        mgr.open_round("R2", "bob", 1000);
        assert!(mgr.round(1).is_some());
        assert!(mgr.round(3).is_none());
    }
}
