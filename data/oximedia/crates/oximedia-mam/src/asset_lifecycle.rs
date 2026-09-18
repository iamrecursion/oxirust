//! Asset lifecycle management for MAM.
//!
//! Provides a state machine for tracking asset lifecycle stages, retention
//! policies, and lifecycle history.

/// Lifecycle stage of a media asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum LifecycleStage {
    /// Asset has just been ingested and awaits processing.
    Ingest,
    /// Asset is pending quality control review.
    QcPending,
    /// Asset has passed quality control.
    QcPassed,
    /// Asset has failed quality control.
    QcFailed,
    /// Asset is active and available for use.
    Active,
    /// Asset has been archived (low-cost storage).
    Archived,
    /// Asset has been marked for deletion.
    Deleted,
}

impl LifecycleStage {
    /// Returns `true` if transitioning to `next` is a valid state machine move.
    #[must_use]
    pub fn can_transition_to(&self, next: &Self) -> bool {
        match self {
            Self::Ingest => matches!(next, Self::QcPending),
            Self::QcPending => matches!(next, Self::QcPassed | Self::QcFailed),
            Self::QcPassed => matches!(next, Self::Active),
            Self::QcFailed => matches!(next, Self::QcPending | Self::Deleted),
            Self::Active => matches!(next, Self::Archived | Self::Deleted),
            Self::Archived => matches!(next, Self::Active | Self::Deleted),
            Self::Deleted => false,
        }
    }

    /// Human-readable label for this stage.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ingest => "Ingest",
            Self::QcPending => "QC Pending",
            Self::QcPassed => "QC Passed",
            Self::QcFailed => "QC Failed",
            Self::Active => "Active",
            Self::Archived => "Archived",
            Self::Deleted => "Deleted",
        }
    }

    /// Returns `true` if this is a terminal stage (no further transitions possible).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Deleted)
    }
}

/// Tracks the full lifecycle of a single asset.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AssetLifecycle {
    /// Unique identifier for the asset.
    pub asset_id: String,
    /// Current lifecycle stage.
    pub stage: LifecycleStage,
    /// History of (stage, timestamp_ms) entries, oldest first.
    pub history: Vec<(LifecycleStage, u64)>,
}

impl AssetLifecycle {
    /// Create a new `AssetLifecycle` starting at `Ingest`.
    #[must_use]
    pub fn new(asset_id: impl Into<String>, timestamp_ms: u64) -> Self {
        let stage = LifecycleStage::Ingest;
        let history = vec![(LifecycleStage::Ingest, timestamp_ms)];
        Self {
            asset_id: asset_id.into(),
            stage,
            history,
        }
    }

    /// Attempt to transition the asset to `new_stage`.
    ///
    /// Returns `true` when the transition is valid and was applied.
    pub fn transition(&mut self, new_stage: LifecycleStage, timestamp_ms: u64) -> bool {
        if !self.stage.can_transition_to(&new_stage) {
            return false;
        }
        self.stage = new_stage.clone();
        self.history.push((new_stage, timestamp_ms));
        true
    }

    /// Returns how long (in milliseconds) the asset has been in its current stage.
    #[must_use]
    pub fn age_in_stage_ms(&self, now_ms: u64) -> u64 {
        self.history
            .last()
            .map(|(_, ts)| now_ms.saturating_sub(*ts))
            .unwrap_or(0)
    }

    /// Returns the total number of stage transitions recorded.
    #[must_use]
    pub fn transition_count(&self) -> usize {
        // History starts with the initial stage; transitions = entries - 1.
        self.history.len().saturating_sub(1)
    }
}

/// Retention policy that dictates how long assets spend in each phase.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RetentionPolicy {
    /// Policy name.
    pub name: String,
    /// Number of days an asset remains active before archival.
    pub active_days: u32,
    /// Number of days an asset remains archived before deletion.
    pub archive_after_days: u32,
    /// Total days after ingest before deletion (active + archive).
    pub delete_after_days: u32,
}

impl RetentionPolicy {
    /// Create a new retention policy.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        active_days: u32,
        archive_after_days: u32,
        delete_after_days: u32,
    ) -> Self {
        Self {
            name: name.into(),
            active_days,
            archive_after_days,
            delete_after_days,
        }
    }

    /// Standard broadcast news retention: active 30d, archive 365d, delete after 2y.
    #[must_use]
    pub fn broadcast_news() -> Self {
        Self::new("Broadcast News", 30, 365, 730)
    }

    /// Sports content retention: active 90d, archive 5y, delete after ~6y.
    #[must_use]
    pub fn sports() -> Self {
        Self::new("Sports", 90, 1825, 2190)
    }

    /// Corporate content retention: active 365d, archive 2y, delete after 3y.
    #[must_use]
    pub fn corporate() -> Self {
        Self::new("Corporate", 365, 730, 1095)
    }

    /// Return the recommended lifecycle stage for an asset of `age_days` old.
    #[must_use]
    pub fn action_for_age(&self, age_days: u32) -> LifecycleStage {
        if age_days >= self.delete_after_days {
            LifecycleStage::Deleted
        } else if age_days >= self.active_days {
            LifecycleStage::Archived
        } else {
            LifecycleStage::Active
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- LifecycleStage transition tests ---

    #[test]
    fn test_valid_transition_ingest_to_qc_pending() {
        assert!(LifecycleStage::Ingest.can_transition_to(&LifecycleStage::QcPending));
    }

    #[test]
    fn test_invalid_transition_ingest_to_active() {
        assert!(!LifecycleStage::Ingest.can_transition_to(&LifecycleStage::Active));
    }

    #[test]
    fn test_valid_transition_qc_pending_to_passed() {
        assert!(LifecycleStage::QcPending.can_transition_to(&LifecycleStage::QcPassed));
    }

    #[test]
    fn test_valid_transition_qc_pending_to_failed() {
        assert!(LifecycleStage::QcPending.can_transition_to(&LifecycleStage::QcFailed));
    }

    #[test]
    fn test_valid_transition_qc_failed_to_pending_retry() {
        assert!(LifecycleStage::QcFailed.can_transition_to(&LifecycleStage::QcPending));
    }

    #[test]
    fn test_terminal_stage_no_transitions() {
        assert!(!LifecycleStage::Deleted.can_transition_to(&LifecycleStage::Active));
        assert!(LifecycleStage::Deleted.is_terminal());
    }

    #[test]
    fn test_archived_can_restore_to_active() {
        assert!(LifecycleStage::Archived.can_transition_to(&LifecycleStage::Active));
    }

    // --- AssetLifecycle tests ---

    #[test]
    fn test_asset_lifecycle_starts_at_ingest() {
        let lc = AssetLifecycle::new("asset-001", 1000);
        assert_eq!(lc.stage, LifecycleStage::Ingest);
        assert_eq!(lc.transition_count(), 0);
    }

    #[test]
    fn test_asset_lifecycle_valid_transition() {
        let mut lc = AssetLifecycle::new("asset-002", 0);
        let ok = lc.transition(LifecycleStage::QcPending, 1000);
        assert!(ok);
        assert_eq!(lc.stage, LifecycleStage::QcPending);
        assert_eq!(lc.transition_count(), 1);
    }

    #[test]
    fn test_asset_lifecycle_invalid_transition_rejected() {
        let mut lc = AssetLifecycle::new("asset-003", 0);
        let ok = lc.transition(LifecycleStage::Active, 1000); // invalid: Ingest -> Active
        assert!(!ok);
        assert_eq!(lc.stage, LifecycleStage::Ingest);
    }

    #[test]
    fn test_asset_lifecycle_age_in_stage() {
        let lc = AssetLifecycle::new("asset-004", 1_000);
        let age = lc.age_in_stage_ms(6_000);
        assert_eq!(age, 5_000);
    }

    #[test]
    fn test_asset_lifecycle_full_happy_path() {
        let mut lc = AssetLifecycle::new("asset-005", 0);
        assert!(lc.transition(LifecycleStage::QcPending, 100));
        assert!(lc.transition(LifecycleStage::QcPassed, 200));
        assert!(lc.transition(LifecycleStage::Active, 300));
        assert!(lc.transition(LifecycleStage::Archived, 400));
        assert_eq!(lc.stage, LifecycleStage::Archived);
        assert_eq!(lc.transition_count(), 4);
    }

    // --- RetentionPolicy tests ---

    #[test]
    fn test_broadcast_news_policy() {
        let p = RetentionPolicy::broadcast_news();
        assert_eq!(p.active_days, 30);
        assert_eq!(p.archive_after_days, 365);
        assert_eq!(p.delete_after_days, 730);
    }

    #[test]
    fn test_retention_action_for_age_active() {
        let p = RetentionPolicy::broadcast_news();
        assert_eq!(p.action_for_age(10), LifecycleStage::Active);
    }

    #[test]
    fn test_retention_action_for_age_archived() {
        let p = RetentionPolicy::broadcast_news();
        assert_eq!(p.action_for_age(50), LifecycleStage::Archived);
    }

    #[test]
    fn test_retention_action_for_age_deleted() {
        let p = RetentionPolicy::broadcast_news();
        assert_eq!(p.action_for_age(730), LifecycleStage::Deleted);
    }

    #[test]
    fn test_sports_policy_longer_retention() {
        let p = RetentionPolicy::sports();
        assert!(p.delete_after_days > RetentionPolicy::broadcast_news().delete_after_days);
    }

    #[test]
    fn test_label_for_each_stage() {
        assert_eq!(LifecycleStage::Ingest.label(), "Ingest");
        assert_eq!(LifecycleStage::Active.label(), "Active");
        assert_eq!(LifecycleStage::Deleted.label(), "Deleted");
    }
}
