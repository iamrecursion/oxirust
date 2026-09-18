//! Cold / glacier storage tier management.
//!
//! Provides an `ArchiveTier` classification for assets and policy-driven
//! automatic demotion from hot to cold/glacier tiers based on age and access
//! patterns.  Retrieval scheduling allows controlled recall of archived assets.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Tier enum
// ---------------------------------------------------------------------------

/// Storage tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArchiveTier {
    /// Fastest access; high cost.
    Hot,
    /// Moderate latency; reduced cost.
    Warm,
    /// Minutes-to-hours latency; low cost.
    Cold,
    /// Hours-to-days latency; lowest cost.
    Glacier,
}

impl ArchiveTier {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Warm => "warm",
            Self::Cold => "cold",
            Self::Glacier => "glacier",
        }
    }

    /// Numeric ordering value (lower = hotter).
    #[must_use]
    pub const fn ordinal(&self) -> u8 {
        match self {
            Self::Hot => 0,
            Self::Warm => 1,
            Self::Cold => 2,
            Self::Glacier => 3,
        }
    }

    /// Returns the next colder tier, or `None` if already at glacier.
    #[must_use]
    pub const fn demote(&self) -> Option<ArchiveTier> {
        match self {
            Self::Hot => Some(Self::Warm),
            Self::Warm => Some(Self::Cold),
            Self::Cold => Some(Self::Glacier),
            Self::Glacier => None,
        }
    }

    /// Returns the next hotter tier, or `None` if already at hot.
    #[must_use]
    pub const fn promote(&self) -> Option<ArchiveTier> {
        match self {
            Self::Glacier => Some(Self::Cold),
            Self::Cold => Some(Self::Warm),
            Self::Warm => Some(Self::Hot),
            Self::Hot => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tier policy
// ---------------------------------------------------------------------------

/// A single demotion rule: if an asset on `from_tier` has not been accessed
/// for `idle_days`, demote it to `to_tier`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemotionRule {
    /// Source tier.
    pub from_tier: ArchiveTier,
    /// Target tier.
    pub to_tier: ArchiveTier,
    /// Number of days without access before demotion triggers.
    pub idle_days: u32,
}

/// A policy that contains an ordered set of demotion rules.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierPolicy {
    /// Name of the policy.
    pub name: String,
    /// Ordered list of demotion rules (evaluated from first to last).
    pub rules: Vec<DemotionRule>,
}

impl TierPolicy {
    /// Create a new, empty policy with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rules: Vec::new(),
        }
    }

    /// Add a demotion rule to the policy.
    pub fn add_rule(&mut self, from: ArchiveTier, to: ArchiveTier, idle_days: u32) {
        self.rules.push(DemotionRule {
            from_tier: from,
            to_tier: to,
            idle_days,
        });
    }

    /// Builder: add rule and return self.
    #[must_use]
    pub fn with_rule(mut self, from: ArchiveTier, to: ArchiveTier, idle_days: u32) -> Self {
        self.add_rule(from, to, idle_days);
        self
    }

    /// Create a sensible default policy: Hot->Warm 30d, Warm->Cold 90d, Cold->Glacier 365d.
    #[must_use]
    pub fn default_policy() -> Self {
        Self::new("default")
            .with_rule(ArchiveTier::Hot, ArchiveTier::Warm, 30)
            .with_rule(ArchiveTier::Warm, ArchiveTier::Cold, 90)
            .with_rule(ArchiveTier::Cold, ArchiveTier::Glacier, 365)
    }

    /// Evaluate the policy for a single asset and return the target tier it
    /// should be demoted to, or `None` if no rule fires.
    #[must_use]
    pub fn evaluate(
        &self,
        current_tier: ArchiveTier,
        last_accessed: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> Option<ArchiveTier> {
        let idle = now.signed_duration_since(last_accessed);
        for rule in &self.rules {
            if rule.from_tier == current_tier {
                let threshold = Duration::days(i64::from(rule.idle_days));
                if idle >= threshold {
                    return Some(rule.to_tier);
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Asset tier record
// ---------------------------------------------------------------------------

/// Tracks the current tier and access time for a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetTierRecord {
    pub asset_id: Uuid,
    pub tier: ArchiveTier,
    pub last_accessed: DateTime<Utc>,
    pub tier_changed_at: DateTime<Utc>,
}

impl AssetTierRecord {
    /// Create a new record (defaults to Hot).
    #[must_use]
    pub fn new(asset_id: Uuid) -> Self {
        let now = Utc::now();
        Self {
            asset_id,
            tier: ArchiveTier::Hot,
            last_accessed: now,
            tier_changed_at: now,
        }
    }

    /// Record an access and optionally promote back to Hot.
    pub fn record_access(&mut self, promote_to_hot: bool) {
        self.last_accessed = Utc::now();
        if promote_to_hot && self.tier != ArchiveTier::Hot {
            self.tier = ArchiveTier::Hot;
            self.tier_changed_at = self.last_accessed;
        }
    }

    /// Apply a tier demotion.
    pub fn demote_to(&mut self, tier: ArchiveTier) {
        self.tier = tier;
        self.tier_changed_at = Utc::now();
    }
}

// ---------------------------------------------------------------------------
// Retrieval request
// ---------------------------------------------------------------------------

/// Priority of a retrieval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalPriority {
    /// Expedited (minutes, higher cost).
    Expedited,
    /// Standard (hours).
    Standard,
    /// Bulk (cheapest, longest).
    Bulk,
}

/// Status of a retrieval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalStatus {
    /// Waiting to be scheduled.
    Pending,
    /// In progress (being restored from archive).
    InProgress,
    /// Completed — asset is available on a warmer tier.
    Completed,
    /// Failed.
    Failed,
}

/// A request to retrieve an asset from a cold/glacier tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRequest {
    pub id: Uuid,
    pub asset_id: Uuid,
    pub source_tier: ArchiveTier,
    pub target_tier: ArchiveTier,
    pub priority: RetrievalPriority,
    pub status: RetrievalStatus,
    pub requested_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub requested_by: Option<String>,
    pub reason: Option<String>,
}

impl RetrievalRequest {
    /// Create a new pending retrieval request.
    #[must_use]
    pub fn new(
        asset_id: Uuid,
        source_tier: ArchiveTier,
        target_tier: ArchiveTier,
        priority: RetrievalPriority,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            asset_id,
            source_tier,
            target_tier,
            priority,
            status: RetrievalStatus::Pending,
            requested_at: Utc::now(),
            completed_at: None,
            requested_by: None,
            reason: None,
        }
    }

    /// Builder: set the requester.
    #[must_use]
    pub fn with_requester(mut self, requester: impl Into<String>) -> Self {
        self.requested_by = Some(requester.into());
        self
    }

    /// Builder: set a reason for the retrieval.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Transition to in-progress.
    pub fn start(&mut self) {
        self.status = RetrievalStatus::InProgress;
    }

    /// Mark as completed.
    pub fn complete(&mut self) {
        self.status = RetrievalStatus::Completed;
        self.completed_at = Some(Utc::now());
    }

    /// Mark as failed.
    pub fn fail(&mut self) {
        self.status = RetrievalStatus::Failed;
        self.completed_at = Some(Utc::now());
    }
}

// ---------------------------------------------------------------------------
// Tier manager
// ---------------------------------------------------------------------------

/// In-memory tier manager that evaluates policies against a set of assets.
#[derive(Debug)]
pub struct TierManager {
    policy: TierPolicy,
    records: HashMap<Uuid, AssetTierRecord>,
    retrieval_queue: Vec<RetrievalRequest>,
}

impl TierManager {
    /// Create a new tier manager with the given policy.
    #[must_use]
    pub fn new(policy: TierPolicy) -> Self {
        Self {
            policy,
            records: HashMap::new(),
            retrieval_queue: Vec::new(),
        }
    }

    /// Register a new asset (defaults to Hot).
    pub fn register_asset(&mut self, asset_id: Uuid) {
        self.records
            .entry(asset_id)
            .or_insert_with(|| AssetTierRecord::new(asset_id));
    }

    /// Record an access for the given asset.
    ///
    /// If `promote_to_hot` is true the asset is moved back to the Hot tier.
    pub fn record_access(&mut self, asset_id: Uuid, promote_to_hot: bool) {
        if let Some(rec) = self.records.get_mut(&asset_id) {
            rec.record_access(promote_to_hot);
        }
    }

    /// Get the current tier for an asset.
    #[must_use]
    pub fn current_tier(&self, asset_id: &Uuid) -> Option<ArchiveTier> {
        self.records.get(asset_id).map(|r| r.tier)
    }

    /// Evaluate all assets and apply demotion rules. Returns a list of
    /// `(asset_id, old_tier, new_tier)` for assets that were demoted.
    pub fn apply_policy(&mut self) -> Vec<(Uuid, ArchiveTier, ArchiveTier)> {
        let now = Utc::now();
        let mut demotions = Vec::new();

        for rec in self.records.values_mut() {
            if let Some(target) = self.policy.evaluate(rec.tier, rec.last_accessed, now) {
                let old = rec.tier;
                rec.demote_to(target);
                demotions.push((rec.asset_id, old, target));
            }
        }

        demotions
    }

    /// Submit a retrieval request for an archived asset.
    pub fn request_retrieval(&mut self, request: RetrievalRequest) {
        self.retrieval_queue.push(request);
    }

    /// Get all pending retrieval requests.
    #[must_use]
    pub fn pending_retrievals(&self) -> Vec<&RetrievalRequest> {
        self.retrieval_queue
            .iter()
            .filter(|r| r.status == RetrievalStatus::Pending)
            .collect()
    }

    /// Number of managed assets.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.records.len()
    }

    /// Number of retrieval requests in the queue (all statuses).
    #[must_use]
    pub fn retrieval_count(&self) -> usize {
        self.retrieval_queue.len()
    }

    /// Distribution of assets across tiers.
    #[must_use]
    pub fn tier_distribution(&self) -> HashMap<ArchiveTier, usize> {
        let mut dist = HashMap::new();
        for rec in self.records.values() {
            *dist.entry(rec.tier).or_insert(0) += 1;
        }
        dist
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archive_tier_labels() {
        assert_eq!(ArchiveTier::Hot.label(), "hot");
        assert_eq!(ArchiveTier::Warm.label(), "warm");
        assert_eq!(ArchiveTier::Cold.label(), "cold");
        assert_eq!(ArchiveTier::Glacier.label(), "glacier");
    }

    #[test]
    fn test_archive_tier_ordinal() {
        assert!(ArchiveTier::Hot.ordinal() < ArchiveTier::Warm.ordinal());
        assert!(ArchiveTier::Cold.ordinal() < ArchiveTier::Glacier.ordinal());
    }

    #[test]
    fn test_demote_chain() {
        assert_eq!(ArchiveTier::Hot.demote(), Some(ArchiveTier::Warm));
        assert_eq!(ArchiveTier::Warm.demote(), Some(ArchiveTier::Cold));
        assert_eq!(ArchiveTier::Cold.demote(), Some(ArchiveTier::Glacier));
        assert_eq!(ArchiveTier::Glacier.demote(), None);
    }

    #[test]
    fn test_promote_chain() {
        assert_eq!(ArchiveTier::Glacier.promote(), Some(ArchiveTier::Cold));
        assert_eq!(ArchiveTier::Cold.promote(), Some(ArchiveTier::Warm));
        assert_eq!(ArchiveTier::Warm.promote(), Some(ArchiveTier::Hot));
        assert_eq!(ArchiveTier::Hot.promote(), None);
    }

    #[test]
    fn test_tier_policy_default() {
        let pol = TierPolicy::default_policy();
        assert_eq!(pol.rules.len(), 3);
        assert_eq!(pol.rules[0].from_tier, ArchiveTier::Hot);
        assert_eq!(pol.rules[0].to_tier, ArchiveTier::Warm);
    }

    #[test]
    fn test_tier_policy_evaluate_no_demotion() {
        let pol = TierPolicy::default_policy();
        let now = Utc::now();
        let last = now - Duration::days(5); // only 5 days idle
        assert!(pol.evaluate(ArchiveTier::Hot, last, now).is_none());
    }

    #[test]
    fn test_tier_policy_evaluate_demotion() {
        let pol = TierPolicy::default_policy();
        let now = Utc::now();
        let last = now - Duration::days(31);
        assert_eq!(
            pol.evaluate(ArchiveTier::Hot, last, now),
            Some(ArchiveTier::Warm)
        );
    }

    #[test]
    fn test_tier_policy_evaluate_glacier_no_rule() {
        let pol = TierPolicy::default_policy();
        let now = Utc::now();
        let last = now - Duration::days(9999);
        assert!(pol.evaluate(ArchiveTier::Glacier, last, now).is_none());
    }

    #[test]
    fn test_asset_tier_record_new() {
        let rec = AssetTierRecord::new(Uuid::new_v4());
        assert_eq!(rec.tier, ArchiveTier::Hot);
    }

    #[test]
    fn test_asset_tier_record_access_promote() {
        let mut rec = AssetTierRecord::new(Uuid::new_v4());
        rec.tier = ArchiveTier::Cold;
        rec.record_access(true);
        assert_eq!(rec.tier, ArchiveTier::Hot);
    }

    #[test]
    fn test_asset_tier_record_access_no_promote() {
        let mut rec = AssetTierRecord::new(Uuid::new_v4());
        rec.tier = ArchiveTier::Cold;
        rec.record_access(false);
        assert_eq!(rec.tier, ArchiveTier::Cold);
    }

    #[test]
    fn test_retrieval_request_lifecycle() {
        let mut req = RetrievalRequest::new(
            Uuid::new_v4(),
            ArchiveTier::Glacier,
            ArchiveTier::Hot,
            RetrievalPriority::Standard,
        );
        assert_eq!(req.status, RetrievalStatus::Pending);

        req.start();
        assert_eq!(req.status, RetrievalStatus::InProgress);

        req.complete();
        assert_eq!(req.status, RetrievalStatus::Completed);
        assert!(req.completed_at.is_some());
    }

    #[test]
    fn test_retrieval_request_fail() {
        let mut req = RetrievalRequest::new(
            Uuid::new_v4(),
            ArchiveTier::Cold,
            ArchiveTier::Hot,
            RetrievalPriority::Expedited,
        );
        req.start();
        req.fail();
        assert_eq!(req.status, RetrievalStatus::Failed);
    }

    #[test]
    fn test_retrieval_request_builder() {
        let req = RetrievalRequest::new(
            Uuid::new_v4(),
            ArchiveTier::Cold,
            ArchiveTier::Hot,
            RetrievalPriority::Bulk,
        )
        .with_requester("alice")
        .with_reason("client delivery");

        assert_eq!(req.requested_by.as_deref(), Some("alice"));
        assert_eq!(req.reason.as_deref(), Some("client delivery"));
    }

    #[test]
    fn test_tier_manager_register_and_tier() {
        let mut mgr = TierManager::new(TierPolicy::default_policy());
        let id = Uuid::new_v4();
        mgr.register_asset(id);
        assert_eq!(mgr.current_tier(&id), Some(ArchiveTier::Hot));
        assert_eq!(mgr.asset_count(), 1);
    }

    #[test]
    fn test_tier_manager_apply_policy() {
        let mut mgr = TierManager::new(TierPolicy::default_policy());
        let id = Uuid::new_v4();
        mgr.register_asset(id);

        // Manually backdate last_accessed
        if let Some(rec) = mgr.records.get_mut(&id) {
            rec.last_accessed = Utc::now() - Duration::days(40);
        }

        let demotions = mgr.apply_policy();
        assert_eq!(demotions.len(), 1);
        assert_eq!(demotions[0].0, id);
        assert_eq!(demotions[0].1, ArchiveTier::Hot);
        assert_eq!(demotions[0].2, ArchiveTier::Warm);
        assert_eq!(mgr.current_tier(&id), Some(ArchiveTier::Warm));
    }

    #[test]
    fn test_tier_manager_retrieval() {
        let mut mgr = TierManager::new(TierPolicy::default_policy());
        let id = Uuid::new_v4();
        mgr.register_asset(id);

        let req = RetrievalRequest::new(
            id,
            ArchiveTier::Glacier,
            ArchiveTier::Hot,
            RetrievalPriority::Standard,
        );
        mgr.request_retrieval(req);

        assert_eq!(mgr.retrieval_count(), 1);
        assert_eq!(mgr.pending_retrievals().len(), 1);
    }

    #[test]
    fn test_tier_manager_distribution() {
        let mut mgr = TierManager::new(TierPolicy::default_policy());
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        mgr.register_asset(a);
        mgr.register_asset(b);
        if let Some(rec) = mgr.records.get_mut(&b) {
            rec.tier = ArchiveTier::Cold;
        }

        let dist = mgr.tier_distribution();
        assert_eq!(dist.get(&ArchiveTier::Hot).copied().unwrap_or(0), 1);
        assert_eq!(dist.get(&ArchiveTier::Cold).copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_tier_manager_record_access_promotes() {
        let mut mgr = TierManager::new(TierPolicy::default_policy());
        let id = Uuid::new_v4();
        mgr.register_asset(id);
        if let Some(rec) = mgr.records.get_mut(&id) {
            rec.tier = ArchiveTier::Cold;
        }
        mgr.record_access(id, true);
        assert_eq!(mgr.current_tier(&id), Some(ArchiveTier::Hot));
    }

    #[test]
    fn test_demotion_rule_serialization() {
        let rule = DemotionRule {
            from_tier: ArchiveTier::Hot,
            to_tier: ArchiveTier::Warm,
            idle_days: 30,
        };
        let json = serde_json::to_string(&rule).expect("ser");
        let deser: DemotionRule = serde_json::from_str(&json).expect("deser");
        assert_eq!(deser.from_tier, ArchiveTier::Hot);
        assert_eq!(deser.idle_days, 30);
    }
}
