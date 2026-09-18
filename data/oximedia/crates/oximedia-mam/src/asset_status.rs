#![allow(dead_code)]
//! Asset status lifecycle management for MAM.
//!
//! Provides [`AssetStatus`] enum with lifecycle predicates, [`AssetStatusHistory`]
//! for audit-trail tracking, and [`AssetStatusFilter`] for querying by status.

use std::fmt;

// ---------------------------------------------------------------------------
// AssetStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a media asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetStatus {
    /// Newly created; not yet submitted for review.
    Draft,
    /// Submitted and awaiting review.
    InReview,
    /// Reviewed and approved for use.
    Approved,
    /// Moved to long-term archive; not actively in use.
    Archived,
    /// Soft-deleted; logically removed but data retained.
    Deleted,
}

impl AssetStatus {
    /// Returns `true` for statuses that represent an actively usable asset.
    ///
    /// `Draft`, `InReview`, and `Approved` are considered active.
    #[must_use]
    pub fn is_active(self) -> bool {
        matches!(self, Self::Draft | Self::InReview | Self::Approved)
    }

    /// Returns `true` if the asset can transition to `target`.
    ///
    /// Allowed transitions:
    /// - Draft → InReview
    /// - InReview → Approved | Draft (rejection sends back to Draft)
    /// - Approved → Archived
    /// - Archived → Approved (unarchive)
    /// - Any → Deleted
    #[must_use]
    pub fn can_transition_to(self, target: Self) -> bool {
        if target == Self::Deleted {
            return true;
        }
        match self {
            Self::Draft => matches!(target, Self::InReview),
            Self::InReview => matches!(target, Self::Approved | Self::Draft),
            Self::Approved => matches!(target, Self::Archived),
            Self::Archived => matches!(target, Self::Approved),
            Self::Deleted => false,
        }
    }

    /// Short display label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::InReview => "in_review",
            Self::Approved => "approved",
            Self::Archived => "archived",
            Self::Deleted => "deleted",
        }
    }
}

impl fmt::Display for AssetStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// AssetStatusEntry
// ---------------------------------------------------------------------------

/// A single entry in the asset status history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetStatusEntry {
    /// The status value.
    pub status: AssetStatus,
    /// Reason or comment for the transition (may be empty).
    pub reason: String,
    /// Epoch-seconds timestamp of the transition.
    pub timestamp_secs: u64,
}

impl AssetStatusEntry {
    /// Create a new entry.
    #[must_use]
    pub fn new(status: AssetStatus, reason: impl Into<String>, timestamp_secs: u64) -> Self {
        Self {
            status,
            reason: reason.into(),
            timestamp_secs,
        }
    }
}

// ---------------------------------------------------------------------------
// AssetStatusHistory
// ---------------------------------------------------------------------------

/// Append-only audit trail of asset status transitions.
#[derive(Debug, Clone, Default)]
pub struct AssetStatusHistory {
    entries: Vec<AssetStatusEntry>,
}

impl AssetStatusHistory {
    /// Create a new empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new status entry onto the history.
    pub fn push(&mut self, entry: AssetStatusEntry) {
        self.entries.push(entry);
    }

    /// Push a status directly (convenience wrapper).
    pub fn push_status(
        &mut self,
        status: AssetStatus,
        reason: impl Into<String>,
        timestamp_secs: u64,
    ) {
        self.entries
            .push(AssetStatusEntry::new(status, reason, timestamp_secs));
    }

    /// Return the most recent status, or `None` if the history is empty.
    #[must_use]
    pub fn current(&self) -> Option<AssetStatus> {
        self.entries.last().map(|e| e.status)
    }

    /// Return the second-to-last status (the one before the current), or
    /// `None` if fewer than two entries exist.
    #[must_use]
    pub fn previous(&self) -> Option<AssetStatus> {
        let len = self.entries.len();
        if len < 2 {
            return None;
        }
        Some(self.entries[len - 2].status)
    }

    /// Number of entries in the history.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if no entries have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries in chronological order.
    pub fn iter(&self) -> impl Iterator<Item = &AssetStatusEntry> {
        self.entries.iter()
    }
}

// ---------------------------------------------------------------------------
// AssetStatusFilter
// ---------------------------------------------------------------------------

/// A predicate set for filtering assets by one or more statuses.
#[derive(Debug, Clone, Default)]
pub struct AssetStatusFilter {
    statuses: Vec<AssetStatus>,
    /// When `true`, only active statuses are matched regardless of `statuses`.
    active_only: bool,
}

impl AssetStatusFilter {
    /// Create a new, empty filter (matches nothing until statuses are added).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a filter that matches all active statuses.
    #[must_use]
    pub fn active_only() -> Self {
        Self {
            statuses: vec![],
            active_only: true,
        }
    }

    /// Add a status to the filter's allow-list.
    pub fn add(&mut self, status: AssetStatus) {
        if !self.statuses.contains(&status) {
            self.statuses.push(status);
        }
    }

    /// Builder-style addition.
    #[must_use]
    pub fn with(mut self, status: AssetStatus) -> Self {
        self.add(status);
        self
    }

    /// Returns `true` if `status` satisfies this filter.
    #[must_use]
    pub fn matches(&self, status: AssetStatus) -> bool {
        if self.active_only {
            return status.is_active();
        }
        self.statuses.contains(&status)
    }

    /// Returns the number of statuses explicitly listed (not counting
    /// the `active_only` shorthand).
    #[must_use]
    pub fn len(&self) -> usize {
        self.statuses.len()
    }

    /// `true` if no explicit statuses and `active_only` is `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.active_only && self.statuses.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_active_draft() {
        assert!(AssetStatus::Draft.is_active());
    }

    #[test]
    fn test_is_active_in_review() {
        assert!(AssetStatus::InReview.is_active());
    }

    #[test]
    fn test_is_active_approved() {
        assert!(AssetStatus::Approved.is_active());
    }

    #[test]
    fn test_is_active_archived_false() {
        assert!(!AssetStatus::Archived.is_active());
    }

    #[test]
    fn test_is_active_deleted_false() {
        assert!(!AssetStatus::Deleted.is_active());
    }

    #[test]
    fn test_transition_draft_to_in_review() {
        assert!(AssetStatus::Draft.can_transition_to(AssetStatus::InReview));
    }

    #[test]
    fn test_transition_in_review_to_approved() {
        assert!(AssetStatus::InReview.can_transition_to(AssetStatus::Approved));
    }

    #[test]
    fn test_transition_in_review_rejected_to_draft() {
        assert!(AssetStatus::InReview.can_transition_to(AssetStatus::Draft));
    }

    #[test]
    fn test_transition_approved_to_archived() {
        assert!(AssetStatus::Approved.can_transition_to(AssetStatus::Archived));
    }

    #[test]
    fn test_transition_archived_to_approved() {
        assert!(AssetStatus::Archived.can_transition_to(AssetStatus::Approved));
    }

    #[test]
    fn test_transition_any_to_deleted() {
        for status in [
            AssetStatus::Draft,
            AssetStatus::InReview,
            AssetStatus::Approved,
            AssetStatus::Archived,
        ] {
            assert!(status.can_transition_to(AssetStatus::Deleted));
        }
    }

    #[test]
    fn test_transition_deleted_cannot_transition() {
        for target in [
            AssetStatus::Draft,
            AssetStatus::InReview,
            AssetStatus::Approved,
            AssetStatus::Archived,
        ] {
            assert!(!AssetStatus::Deleted.can_transition_to(target));
        }
    }

    #[test]
    fn test_history_push_and_current() {
        let mut h = AssetStatusHistory::new();
        assert!(h.current().is_none());
        h.push_status(AssetStatus::Draft, "created", 1_000);
        assert_eq!(h.current(), Some(AssetStatus::Draft));
        h.push_status(AssetStatus::InReview, "submitted", 2_000);
        assert_eq!(h.current(), Some(AssetStatus::InReview));
    }

    #[test]
    fn test_history_previous() {
        let mut h = AssetStatusHistory::new();
        h.push_status(AssetStatus::Draft, "", 1_000);
        assert!(h.previous().is_none());
        h.push_status(AssetStatus::InReview, "", 2_000);
        assert_eq!(h.previous(), Some(AssetStatus::Draft));
    }

    #[test]
    fn test_history_len_and_is_empty() {
        let mut h = AssetStatusHistory::new();
        assert!(h.is_empty());
        h.push_status(AssetStatus::Draft, "", 0);
        assert_eq!(h.len(), 1);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_filter_explicit_match() {
        let filter = AssetStatusFilter::new()
            .with(AssetStatus::Draft)
            .with(AssetStatus::Approved);
        assert!(filter.matches(AssetStatus::Draft));
        assert!(filter.matches(AssetStatus::Approved));
        assert!(!filter.matches(AssetStatus::InReview));
        assert!(!filter.matches(AssetStatus::Archived));
    }

    #[test]
    fn test_filter_active_only() {
        let filter = AssetStatusFilter::active_only();
        assert!(filter.matches(AssetStatus::Draft));
        assert!(filter.matches(AssetStatus::InReview));
        assert!(filter.matches(AssetStatus::Approved));
        assert!(!filter.matches(AssetStatus::Archived));
        assert!(!filter.matches(AssetStatus::Deleted));
    }

    #[test]
    fn test_filter_empty_matches_nothing() {
        let filter = AssetStatusFilter::new();
        assert!(filter.is_empty());
        assert!(!filter.matches(AssetStatus::Draft));
    }

    #[test]
    fn test_asset_status_display() {
        assert_eq!(AssetStatus::Draft.to_string(), "draft");
        assert_eq!(AssetStatus::InReview.to_string(), "in_review");
        assert_eq!(AssetStatus::Approved.to_string(), "approved");
        assert_eq!(AssetStatus::Archived.to_string(), "archived");
        assert_eq!(AssetStatus::Deleted.to_string(), "deleted");
    }

    #[test]
    fn test_history_iter() {
        let mut h = AssetStatusHistory::new();
        h.push_status(AssetStatus::Draft, "a", 1);
        h.push_status(AssetStatus::InReview, "b", 2);
        let statuses: Vec<_> = h.iter().map(|e| e.status).collect();
        assert_eq!(statuses, vec![AssetStatus::Draft, AssetStatus::InReview]);
    }
}
