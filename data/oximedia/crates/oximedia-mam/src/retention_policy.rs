#![allow(dead_code)]
//! Data retention policy engine for automated asset lifecycle management.
//!
//! Provides configurable retention rules that determine when assets should be
//! archived, moved to cold storage, or permanently deleted based on age,
//! access patterns, and metadata criteria.

use std::collections::HashMap;
use std::fmt;

/// Unique identifier for a retention policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolicyId(u64);

impl PolicyId {
    /// Create a new policy identifier.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the numeric value of the identifier.
    pub fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for PolicyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "policy-{}", self.0)
    }
}

/// The action to take when a retention rule triggers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionAction {
    /// Move the asset to cold / archive storage.
    Archive,
    /// Permanently delete the asset.
    Delete,
    /// Move the asset to a lower-cost storage tier.
    TierDown,
    /// Flag the asset for manual review.
    FlagForReview,
    /// Notify administrators about the asset.
    Notify,
}

impl fmt::Display for RetentionAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Archive => write!(f, "archive"),
            Self::Delete => write!(f, "delete"),
            Self::TierDown => write!(f, "tier_down"),
            Self::FlagForReview => write!(f, "flag_for_review"),
            Self::Notify => write!(f, "notify"),
        }
    }
}

/// Criteria that determine whether a retention rule matches an asset.
#[derive(Debug, Clone, PartialEq)]
pub struct RetentionCriteria {
    /// Maximum asset age in days before the rule triggers.
    pub max_age_days: Option<u64>,
    /// Minimum days since last access before the rule triggers.
    pub min_idle_days: Option<u64>,
    /// Asset must have one of these media types (e.g. `"video"`, `"audio"`).
    pub media_types: Vec<String>,
    /// Asset must have one of these tags.
    pub required_tags: Vec<String>,
    /// Asset must NOT have any of these tags.
    pub excluded_tags: Vec<String>,
    /// Minimum file size in bytes.
    pub min_size_bytes: Option<u64>,
    /// Maximum file size in bytes.
    pub max_size_bytes: Option<u64>,
}

impl Default for RetentionCriteria {
    fn default() -> Self {
        Self {
            max_age_days: None,
            min_idle_days: None,
            media_types: Vec::new(),
            required_tags: Vec::new(),
            excluded_tags: Vec::new(),
            min_size_bytes: None,
            max_size_bytes: None,
        }
    }
}

impl RetentionCriteria {
    /// Create an empty criteria set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum age in days.
    pub fn with_max_age(mut self, days: u64) -> Self {
        self.max_age_days = Some(days);
        self
    }

    /// Set minimum idle days (days since last access).
    pub fn with_min_idle(mut self, days: u64) -> Self {
        self.min_idle_days = Some(days);
        self
    }

    /// Restrict to specific media types.
    pub fn with_media_types(mut self, types: Vec<String>) -> Self {
        self.media_types = types;
        self
    }

    /// Restrict to assets with specific tags.
    pub fn with_required_tags(mut self, tags: Vec<String>) -> Self {
        self.required_tags = tags;
        self
    }

    /// Exclude assets with specific tags.
    pub fn with_excluded_tags(mut self, tags: Vec<String>) -> Self {
        self.excluded_tags = tags;
        self
    }

    /// Set minimum file size filter.
    pub fn with_min_size(mut self, bytes: u64) -> Self {
        self.min_size_bytes = Some(bytes);
        self
    }
}

/// Lightweight snapshot of an asset for retention evaluation.
#[derive(Debug, Clone)]
pub struct AssetSnapshot {
    /// Asset identifier string.
    pub id: String,
    /// Age of the asset in days.
    pub age_days: u64,
    /// Days since last access.
    pub idle_days: u64,
    /// Media type label.
    pub media_type: String,
    /// Tags associated with the asset.
    pub tags: Vec<String>,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// Check whether a set of criteria matches an asset snapshot.
#[allow(clippy::cast_precision_loss)]
pub fn matches_criteria(criteria: &RetentionCriteria, asset: &AssetSnapshot) -> bool {
    if let Some(max_age) = criteria.max_age_days {
        if asset.age_days < max_age {
            return false;
        }
    }
    if let Some(min_idle) = criteria.min_idle_days {
        if asset.idle_days < min_idle {
            return false;
        }
    }
    if !criteria.media_types.is_empty() && !criteria.media_types.contains(&asset.media_type) {
        return false;
    }
    if !criteria.required_tags.is_empty()
        && !criteria
            .required_tags
            .iter()
            .any(|t| asset.tags.contains(t))
    {
        return false;
    }
    if criteria
        .excluded_tags
        .iter()
        .any(|t| asset.tags.contains(t))
    {
        return false;
    }
    if let Some(min_size) = criteria.min_size_bytes {
        if asset.size_bytes < min_size {
            return false;
        }
    }
    if let Some(max_size) = criteria.max_size_bytes {
        if asset.size_bytes > max_size {
            return false;
        }
    }
    true
}

/// A single retention rule that pairs criteria with an action.
#[derive(Debug, Clone)]
pub struct RetentionRule {
    /// Human-readable name.
    pub name: String,
    /// Evaluation priority (lower = higher priority).
    pub priority: u32,
    /// Whether the rule is currently enabled.
    pub enabled: bool,
    /// Criteria that must match.
    pub criteria: RetentionCriteria,
    /// Action to apply when matched.
    pub action: RetentionAction,
}

impl RetentionRule {
    /// Create a new retention rule.
    pub fn new(
        name: impl Into<String>,
        criteria: RetentionCriteria,
        action: RetentionAction,
    ) -> Self {
        Self {
            name: name.into(),
            priority: 100,
            enabled: true,
            criteria,
            action,
        }
    }

    /// Set the priority of this rule.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Enable or disable this rule.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// The result of evaluating a single asset against the policy engine.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyEvalResult {
    /// Asset identifier.
    pub asset_id: String,
    /// Name of the rule that matched (if any).
    pub matched_rule: Option<String>,
    /// Action to take.
    pub action: Option<RetentionAction>,
}

// ---------------------------------------------------------------------------
// Legal hold
// ---------------------------------------------------------------------------

/// A legal hold that overrides retention rules, preventing any destructive
/// action (Delete, Archive, TierDown) on affected assets until the hold is
/// released.
#[derive(Debug, Clone, PartialEq)]
pub struct LegalHold {
    /// Human-readable name or case reference (e.g. "Case #12345").
    pub name: String,
    /// Free-text reason describing why the hold was placed.
    pub reason: String,
    /// Who placed the hold (username or system identifier).
    pub placed_by: String,
    /// Unix-epoch seconds when the hold was placed.
    pub placed_at: u64,
    /// Optional expiry timestamp. `None` = indefinite.
    pub expires_at: Option<u64>,
    /// Asset IDs covered by this hold.
    pub asset_ids: Vec<String>,
    /// Tag-based selector: any asset with one of these tags is covered.
    pub tag_selectors: Vec<String>,
    /// Media-type selector: any asset of these media types is covered.
    pub media_type_selectors: Vec<String>,
    /// Whether the hold is currently active.
    pub active: bool,
}

impl LegalHold {
    /// Create a new active legal hold.
    pub fn new(
        name: impl Into<String>,
        reason: impl Into<String>,
        placed_by: impl Into<String>,
        placed_at: u64,
    ) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
            placed_by: placed_by.into(),
            placed_at,
            expires_at: None,
            asset_ids: Vec::new(),
            tag_selectors: Vec::new(),
            media_type_selectors: Vec::new(),
            active: true,
        }
    }

    /// Add specific asset IDs to this hold.
    pub fn with_asset_ids(mut self, ids: Vec<String>) -> Self {
        self.asset_ids = ids;
        self
    }

    /// Add tag selectors: any asset tagged with one of these is held.
    pub fn with_tag_selectors(mut self, tags: Vec<String>) -> Self {
        self.tag_selectors = tags;
        self
    }

    /// Add media-type selectors.
    pub fn with_media_type_selectors(mut self, types: Vec<String>) -> Self {
        self.media_type_selectors = types;
        self
    }

    /// Set an expiry timestamp (Unix epoch seconds).
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Check whether this hold covers the given asset, taking expiry into
    /// account. `now` is the current Unix-epoch time for expiry checking.
    pub fn covers_asset(&self, asset: &AssetSnapshot, now: u64) -> bool {
        if !self.active {
            return false;
        }
        // Check expiry.
        if let Some(exp) = self.expires_at {
            if now >= exp {
                return false;
            }
        }
        // Explicit asset ID match.
        if self.asset_ids.contains(&asset.id) {
            return true;
        }
        // Tag selector match.
        if !self.tag_selectors.is_empty()
            && self.tag_selectors.iter().any(|t| asset.tags.contains(t))
        {
            return true;
        }
        // Media-type selector match.
        if !self.media_type_selectors.is_empty()
            && self
                .media_type_selectors
                .iter()
                .any(|m| m == &asset.media_type)
        {
            return true;
        }
        false
    }
}

/// Unique identifier for a legal hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LegalHoldId(u64);

impl LegalHoldId {
    /// Create a new legal hold identifier.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the numeric value.
    pub fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for LegalHoldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hold-{}", self.0)
    }
}

/// Actions considered "destructive" that legal holds can block.
fn is_destructive(action: &RetentionAction) -> bool {
    matches!(
        action,
        RetentionAction::Delete | RetentionAction::Archive | RetentionAction::TierDown
    )
}

/// The retention policy engine that holds all rules and evaluates assets.
#[derive(Debug)]
pub struct RetentionPolicyEngine {
    /// All registered rules, keyed by policy ID.
    rules: HashMap<PolicyId, RetentionRule>,
    /// Auto-incrementing ID counter.
    next_id: u64,
    /// Active legal holds that override retention rules.
    legal_holds: HashMap<LegalHoldId, LegalHold>,
    /// Auto-incrementing hold ID counter.
    next_hold_id: u64,
}

impl Default for RetentionPolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RetentionPolicyEngine {
    /// Create a new empty policy engine.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            next_id: 1,
            legal_holds: HashMap::new(),
            next_hold_id: 1,
        }
    }

    /// Add a rule and return its assigned policy ID.
    pub fn add_rule(&mut self, rule: RetentionRule) -> PolicyId {
        let id = PolicyId::new(self.next_id);
        self.next_id += 1;
        self.rules.insert(id, rule);
        id
    }

    /// Remove a rule by its policy ID. Returns `true` if it existed.
    pub fn remove_rule(&mut self, id: PolicyId) -> bool {
        self.rules.remove(&id).is_some()
    }

    /// Return the number of rules in the engine.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    // -----------------------------------------------------------------------
    // Legal hold management
    // -----------------------------------------------------------------------

    /// Place a legal hold and return its ID.
    pub fn place_hold(&mut self, hold: LegalHold) -> LegalHoldId {
        let id = LegalHoldId::new(self.next_hold_id);
        self.next_hold_id += 1;
        self.legal_holds.insert(id, hold);
        id
    }

    /// Release (deactivate) a legal hold by ID. Returns `true` if found.
    pub fn release_hold(&mut self, id: LegalHoldId) -> bool {
        if let Some(hold) = self.legal_holds.get_mut(&id) {
            hold.active = false;
            true
        } else {
            false
        }
    }

    /// Remove a legal hold entirely. Returns `true` if it existed.
    pub fn remove_hold(&mut self, id: LegalHoldId) -> bool {
        self.legal_holds.remove(&id).is_some()
    }

    /// Get a reference to a legal hold.
    pub fn get_hold(&self, id: LegalHoldId) -> Option<&LegalHold> {
        self.legal_holds.get(&id)
    }

    /// Return all legal hold IDs.
    pub fn hold_ids(&self) -> Vec<LegalHoldId> {
        self.legal_holds.keys().copied().collect()
    }

    /// Return the number of active legal holds.
    pub fn active_hold_count(&self) -> usize {
        self.legal_holds.values().filter(|h| h.active).count()
    }

    /// Check whether any active hold covers the given asset at time `now`.
    pub fn is_under_hold(&self, asset: &AssetSnapshot, now: u64) -> bool {
        self.legal_holds
            .values()
            .any(|h| h.covers_asset(asset, now))
    }

    /// Return all holds that cover a given asset at time `now`.
    pub fn holds_for_asset(&self, asset: &AssetSnapshot, now: u64) -> Vec<LegalHoldId> {
        self.legal_holds
            .iter()
            .filter(|(_, h)| h.covers_asset(asset, now))
            .map(|(id, _)| *id)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Evaluation (hold-aware)
    // -----------------------------------------------------------------------

    /// Evaluate a single asset against all enabled rules, respecting legal
    /// holds.  If a destructive action would be triggered but the asset is
    /// under a legal hold, the action is replaced with
    /// [`RetentionAction::FlagForReview`] and the result indicates the hold.
    ///
    /// `now` is the current Unix-epoch time used for hold expiry checks.
    pub fn evaluate_with_holds(&self, asset: &AssetSnapshot, now: u64) -> PolicyEvalResult {
        let base = self.evaluate(asset);

        // If the matched action is destructive and the asset is under hold,
        // override the action.
        if let Some(action) = &base.action {
            if is_destructive(action) && self.is_under_hold(asset, now) {
                return PolicyEvalResult {
                    asset_id: base.asset_id,
                    matched_rule: base.matched_rule,
                    action: Some(RetentionAction::FlagForReview),
                };
            }
        }

        base
    }

    /// Evaluate a batch of assets, respecting legal holds.
    pub fn evaluate_batch_with_holds(
        &self,
        assets: &[AssetSnapshot],
        now: u64,
    ) -> Vec<PolicyEvalResult> {
        assets
            .iter()
            .map(|a| self.evaluate_with_holds(a, now))
            .collect()
    }

    /// Evaluate a single asset against all enabled rules.
    /// Returns the first matching rule by priority.
    pub fn evaluate(&self, asset: &AssetSnapshot) -> PolicyEvalResult {
        let mut sorted: Vec<_> = self.rules.values().filter(|r| r.enabled).collect();
        sorted.sort_by_key(|r| r.priority);

        for rule in sorted {
            if matches_criteria(&rule.criteria, asset) {
                return PolicyEvalResult {
                    asset_id: asset.id.clone(),
                    matched_rule: Some(rule.name.clone()),
                    action: Some(rule.action),
                };
            }
        }

        PolicyEvalResult {
            asset_id: asset.id.clone(),
            matched_rule: None,
            action: None,
        }
    }

    /// Evaluate a batch of assets. Returns one result per asset.
    pub fn evaluate_batch(&self, assets: &[AssetSnapshot]) -> Vec<PolicyEvalResult> {
        assets.iter().map(|a| self.evaluate(a)).collect()
    }

    /// Get a reference to a rule by ID.
    pub fn get_rule(&self, id: PolicyId) -> Option<&RetentionRule> {
        self.rules.get(&id)
    }

    /// Get a mutable reference to a rule by ID.
    pub fn get_rule_mut(&mut self, id: PolicyId) -> Option<&mut RetentionRule> {
        self.rules.get_mut(&id)
    }

    /// Return all policy IDs.
    pub fn policy_ids(&self) -> Vec<PolicyId> {
        self.rules.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_asset(id: &str, age: u64, idle: u64, media: &str, size: u64) -> AssetSnapshot {
        AssetSnapshot {
            id: id.to_string(),
            age_days: age,
            idle_days: idle,
            media_type: media.to_string(),
            tags: vec!["production".to_string()],
            size_bytes: size,
        }
    }

    #[test]
    fn test_policy_id_display() {
        let id = PolicyId::new(42);
        assert_eq!(id.to_string(), "policy-42");
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_retention_action_display() {
        assert_eq!(RetentionAction::Archive.to_string(), "archive");
        assert_eq!(RetentionAction::Delete.to_string(), "delete");
        assert_eq!(RetentionAction::TierDown.to_string(), "tier_down");
        assert_eq!(
            RetentionAction::FlagForReview.to_string(),
            "flag_for_review"
        );
        assert_eq!(RetentionAction::Notify.to_string(), "notify");
    }

    #[test]
    fn test_criteria_default() {
        let c = RetentionCriteria::default();
        assert!(c.max_age_days.is_none());
        assert!(c.min_idle_days.is_none());
        assert!(c.media_types.is_empty());
    }

    #[test]
    fn test_criteria_builder() {
        let c = RetentionCriteria::new()
            .with_max_age(365)
            .with_min_idle(90)
            .with_media_types(vec!["video".to_string()])
            .with_min_size(1024);
        assert_eq!(c.max_age_days, Some(365));
        assert_eq!(c.min_idle_days, Some(90));
        assert_eq!(c.media_types, vec!["video"]);
        assert_eq!(c.min_size_bytes, Some(1024));
    }

    #[test]
    fn test_matches_criteria_age() {
        let c = RetentionCriteria::new().with_max_age(30);
        let young = sample_asset("a1", 10, 5, "video", 1000);
        let old = sample_asset("a2", 60, 5, "video", 1000);
        assert!(!matches_criteria(&c, &young));
        assert!(matches_criteria(&c, &old));
    }

    #[test]
    fn test_matches_criteria_idle() {
        let c = RetentionCriteria::new().with_min_idle(90);
        let active = sample_asset("a1", 100, 10, "video", 1000);
        let idle = sample_asset("a2", 100, 120, "video", 1000);
        assert!(!matches_criteria(&c, &active));
        assert!(matches_criteria(&c, &idle));
    }

    #[test]
    fn test_matches_criteria_media_type() {
        let c = RetentionCriteria::new().with_media_types(vec!["audio".to_string()]);
        let video = sample_asset("a1", 100, 100, "video", 1000);
        let audio = sample_asset("a2", 100, 100, "audio", 1000);
        assert!(!matches_criteria(&c, &video));
        assert!(matches_criteria(&c, &audio));
    }

    #[test]
    fn test_matches_criteria_excluded_tags() {
        let c = RetentionCriteria::new().with_excluded_tags(vec!["production".to_string()]);
        let asset = sample_asset("a1", 100, 100, "video", 1000);
        assert!(!matches_criteria(&c, &asset));
    }

    #[test]
    fn test_matches_criteria_size_range() {
        let mut c = RetentionCriteria::new().with_min_size(500);
        c.max_size_bytes = Some(2000);
        let small = sample_asset("a1", 100, 100, "video", 100);
        let mid = sample_asset("a2", 100, 100, "video", 1000);
        let big = sample_asset("a3", 100, 100, "video", 5000);
        assert!(!matches_criteria(&c, &small));
        assert!(matches_criteria(&c, &mid));
        assert!(!matches_criteria(&c, &big));
    }

    #[test]
    fn test_engine_add_remove_rules() {
        let mut engine = RetentionPolicyEngine::new();
        assert_eq!(engine.rule_count(), 0);
        let id = engine.add_rule(RetentionRule::new(
            "test",
            RetentionCriteria::new(),
            RetentionAction::Archive,
        ));
        assert_eq!(engine.rule_count(), 1);
        assert!(engine.remove_rule(id));
        assert_eq!(engine.rule_count(), 0);
        assert!(!engine.remove_rule(id));
    }

    #[test]
    fn test_engine_evaluate_no_match() {
        let engine = RetentionPolicyEngine::new();
        let asset = sample_asset("a1", 10, 5, "video", 1000);
        let result = engine.evaluate(&asset);
        assert!(result.matched_rule.is_none());
        assert!(result.action.is_none());
    }

    #[test]
    fn test_engine_evaluate_match() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "archive_old",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Archive,
        ));
        let asset = sample_asset("a1", 60, 5, "video", 1000);
        let result = engine.evaluate(&asset);
        assert_eq!(result.matched_rule.as_deref(), Some("archive_old"));
        assert_eq!(result.action, Some(RetentionAction::Archive));
    }

    #[test]
    fn test_engine_priority_ordering() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(
            RetentionRule::new(
                "low_priority",
                RetentionCriteria::new().with_max_age(30),
                RetentionAction::Notify,
            )
            .with_priority(200),
        );
        engine.add_rule(
            RetentionRule::new(
                "high_priority",
                RetentionCriteria::new().with_max_age(30),
                RetentionAction::Delete,
            )
            .with_priority(10),
        );
        let asset = sample_asset("a1", 60, 5, "video", 1000);
        let result = engine.evaluate(&asset);
        assert_eq!(result.matched_rule.as_deref(), Some("high_priority"));
        assert_eq!(result.action, Some(RetentionAction::Delete));
    }

    #[test]
    fn test_engine_disabled_rule_skipped() {
        let mut engine = RetentionPolicyEngine::new();
        let id = engine.add_rule(RetentionRule::new(
            "disabled",
            RetentionCriteria::new().with_max_age(1),
            RetentionAction::Delete,
        ));
        engine
            .get_rule_mut(id)
            .expect("should succeed in test")
            .set_enabled(false);
        let asset = sample_asset("a1", 999, 999, "video", 1000);
        let result = engine.evaluate(&asset);
        assert!(result.action.is_none());
    }

    #[test]
    fn test_evaluate_batch() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "archive",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Archive,
        ));
        let assets = vec![
            sample_asset("a1", 10, 5, "video", 1000),
            sample_asset("a2", 60, 5, "video", 1000),
        ];
        let results = engine.evaluate_batch(&assets);
        assert_eq!(results.len(), 2);
        assert!(results[0].action.is_none());
        assert_eq!(results[1].action, Some(RetentionAction::Archive));
    }

    #[test]
    fn test_policy_ids() {
        let mut engine = RetentionPolicyEngine::new();
        let id1 = engine.add_rule(RetentionRule::new(
            "r1",
            RetentionCriteria::new(),
            RetentionAction::Archive,
        ));
        let id2 = engine.add_rule(RetentionRule::new(
            "r2",
            RetentionCriteria::new(),
            RetentionAction::Delete,
        ));
        let ids = engine.policy_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    // -----------------------------------------------------------------------
    // Legal hold tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_legal_hold_id_display() {
        let id = LegalHoldId::new(7);
        assert_eq!(id.to_string(), "hold-7");
        assert_eq!(id.value(), 7);
    }

    #[test]
    fn test_legal_hold_creation() {
        let hold = LegalHold::new("Case #1", "Litigation", "admin", 1000);
        assert_eq!(hold.name, "Case #1");
        assert!(hold.active);
        assert!(hold.expires_at.is_none());
    }

    #[test]
    fn test_legal_hold_builders() {
        let hold = LegalHold::new("Case #2", "Audit", "legal", 2000)
            .with_asset_ids(vec!["a1".to_string()])
            .with_tag_selectors(vec!["confidential".to_string()])
            .with_media_type_selectors(vec!["video".to_string()])
            .with_expiry(5000);
        assert_eq!(hold.asset_ids, vec!["a1"]);
        assert_eq!(hold.tag_selectors, vec!["confidential"]);
        assert_eq!(hold.media_type_selectors, vec!["video"]);
        assert_eq!(hold.expires_at, Some(5000));
    }

    #[test]
    fn test_hold_covers_asset_by_id() {
        let hold = LegalHold::new("H", "r", "u", 100).with_asset_ids(vec!["a1".to_string()]);
        let asset = sample_asset("a1", 100, 100, "video", 1000);
        assert!(hold.covers_asset(&asset, 200));
        let other = sample_asset("a2", 100, 100, "video", 1000);
        assert!(!hold.covers_asset(&other, 200));
    }

    #[test]
    fn test_hold_covers_asset_by_tag() {
        let hold =
            LegalHold::new("H", "r", "u", 100).with_tag_selectors(vec!["production".to_string()]);
        let asset = sample_asset("a1", 100, 100, "video", 1000);
        assert!(hold.covers_asset(&asset, 200));
    }

    #[test]
    fn test_hold_covers_asset_by_media_type() {
        let hold =
            LegalHold::new("H", "r", "u", 100).with_media_type_selectors(vec!["audio".to_string()]);
        let video = sample_asset("a1", 100, 100, "video", 1000);
        let audio = sample_asset("a2", 100, 100, "audio", 1000);
        assert!(!hold.covers_asset(&video, 200));
        assert!(hold.covers_asset(&audio, 200));
    }

    #[test]
    fn test_hold_expired_does_not_cover() {
        let hold = LegalHold::new("H", "r", "u", 100)
            .with_asset_ids(vec!["a1".to_string()])
            .with_expiry(500);
        let asset = sample_asset("a1", 100, 100, "video", 1000);
        assert!(hold.covers_asset(&asset, 400)); // before expiry
        assert!(!hold.covers_asset(&asset, 500)); // at expiry
        assert!(!hold.covers_asset(&asset, 600)); // after expiry
    }

    #[test]
    fn test_hold_inactive_does_not_cover() {
        let mut hold = LegalHold::new("H", "r", "u", 100).with_asset_ids(vec!["a1".to_string()]);
        hold.active = false;
        let asset = sample_asset("a1", 100, 100, "video", 1000);
        assert!(!hold.covers_asset(&asset, 200));
    }

    #[test]
    fn test_engine_place_and_release_hold() {
        let mut engine = RetentionPolicyEngine::new();
        assert_eq!(engine.active_hold_count(), 0);
        let hid = engine.place_hold(LegalHold::new("H1", "r", "u", 100));
        assert_eq!(engine.active_hold_count(), 1);
        assert!(engine.get_hold(hid).is_some());
        assert!(engine.release_hold(hid));
        assert_eq!(engine.active_hold_count(), 0);
        // Hold still exists but is inactive.
        assert!(engine.get_hold(hid).is_some());
    }

    #[test]
    fn test_engine_remove_hold() {
        let mut engine = RetentionPolicyEngine::new();
        let hid = engine.place_hold(LegalHold::new("H1", "r", "u", 100));
        assert!(engine.remove_hold(hid));
        assert!(!engine.remove_hold(hid)); // already gone
        assert!(engine.get_hold(hid).is_none());
    }

    #[test]
    fn test_engine_hold_ids() {
        let mut engine = RetentionPolicyEngine::new();
        let h1 = engine.place_hold(LegalHold::new("H1", "r", "u", 100));
        let h2 = engine.place_hold(LegalHold::new("H2", "r", "u", 200));
        let ids = engine.hold_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&h1));
        assert!(ids.contains(&h2));
    }

    #[test]
    fn test_is_under_hold() {
        let mut engine = RetentionPolicyEngine::new();
        engine
            .place_hold(LegalHold::new("H", "r", "u", 100).with_asset_ids(vec!["a1".to_string()]));
        let a1 = sample_asset("a1", 100, 100, "video", 1000);
        let a2 = sample_asset("a2", 100, 100, "video", 1000);
        assert!(engine.is_under_hold(&a1, 200));
        assert!(!engine.is_under_hold(&a2, 200));
    }

    #[test]
    fn test_holds_for_asset() {
        let mut engine = RetentionPolicyEngine::new();
        let h1 = engine
            .place_hold(LegalHold::new("H1", "r", "u", 100).with_asset_ids(vec!["a1".to_string()]));
        let h2 = engine.place_hold(
            LegalHold::new("H2", "r", "u", 200).with_tag_selectors(vec!["production".to_string()]),
        );
        let a1 = sample_asset("a1", 100, 100, "video", 1000);
        let holds = engine.holds_for_asset(&a1, 300);
        assert!(holds.contains(&h1));
        assert!(holds.contains(&h2)); // a1 has "production" tag
    }

    #[test]
    fn test_evaluate_with_holds_blocks_delete() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "delete_old",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Delete,
        ));
        engine.place_hold(
            LegalHold::new("Litigation", "case", "legal", 100)
                .with_asset_ids(vec!["a1".to_string()]),
        );
        let a1 = sample_asset("a1", 60, 5, "video", 1000);
        let result = engine.evaluate_with_holds(&a1, 200);
        // Delete should be overridden to FlagForReview.
        assert_eq!(result.action, Some(RetentionAction::FlagForReview));
        assert_eq!(result.matched_rule.as_deref(), Some("delete_old"));
    }

    #[test]
    fn test_evaluate_with_holds_blocks_archive() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "archive_old",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Archive,
        ));
        engine.place_hold(
            LegalHold::new("Audit", "reason", "admin", 100).with_asset_ids(vec!["a1".to_string()]),
        );
        let a1 = sample_asset("a1", 60, 5, "video", 1000);
        let result = engine.evaluate_with_holds(&a1, 200);
        assert_eq!(result.action, Some(RetentionAction::FlagForReview));
    }

    #[test]
    fn test_evaluate_with_holds_allows_non_destructive() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "notify_old",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Notify,
        ));
        engine
            .place_hold(LegalHold::new("H", "r", "u", 100).with_asset_ids(vec!["a1".to_string()]));
        let a1 = sample_asset("a1", 60, 5, "video", 1000);
        let result = engine.evaluate_with_holds(&a1, 200);
        // Notify is not destructive, so it passes through.
        assert_eq!(result.action, Some(RetentionAction::Notify));
    }

    #[test]
    fn test_evaluate_with_holds_no_hold_passes_through() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "delete_old",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Delete,
        ));
        let a1 = sample_asset("a1", 60, 5, "video", 1000);
        let result = engine.evaluate_with_holds(&a1, 200);
        assert_eq!(result.action, Some(RetentionAction::Delete));
    }

    #[test]
    fn test_evaluate_with_holds_expired_hold_allows_delete() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "delete_old",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Delete,
        ));
        engine.place_hold(
            LegalHold::new("H", "r", "u", 100)
                .with_asset_ids(vec!["a1".to_string()])
                .with_expiry(500),
        );
        let a1 = sample_asset("a1", 60, 5, "video", 1000);
        // Before expiry: blocked.
        let r1 = engine.evaluate_with_holds(&a1, 400);
        assert_eq!(r1.action, Some(RetentionAction::FlagForReview));
        // After expiry: allowed.
        let r2 = engine.evaluate_with_holds(&a1, 600);
        assert_eq!(r2.action, Some(RetentionAction::Delete));
    }

    #[test]
    fn test_evaluate_batch_with_holds() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "delete",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Delete,
        ));
        engine
            .place_hold(LegalHold::new("H", "r", "u", 100).with_asset_ids(vec!["a1".to_string()]));
        let assets = vec![
            sample_asset("a1", 60, 5, "video", 1000),
            sample_asset("a2", 60, 5, "video", 1000),
        ];
        let results = engine.evaluate_batch_with_holds(&assets, 200);
        assert_eq!(results[0].action, Some(RetentionAction::FlagForReview));
        assert_eq!(results[1].action, Some(RetentionAction::Delete));
    }

    #[test]
    fn test_evaluate_with_holds_blocks_tier_down() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "tier_down",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::TierDown,
        ));
        engine
            .place_hold(LegalHold::new("H", "r", "u", 100).with_asset_ids(vec!["a1".to_string()]));
        let a1 = sample_asset("a1", 60, 5, "video", 1000);
        let result = engine.evaluate_with_holds(&a1, 200);
        assert_eq!(result.action, Some(RetentionAction::FlagForReview));
    }

    #[test]
    fn test_released_hold_does_not_block() {
        let mut engine = RetentionPolicyEngine::new();
        engine.add_rule(RetentionRule::new(
            "delete",
            RetentionCriteria::new().with_max_age(30),
            RetentionAction::Delete,
        ));
        let hid = engine
            .place_hold(LegalHold::new("H", "r", "u", 100).with_asset_ids(vec!["a1".to_string()]));
        engine.release_hold(hid);
        let a1 = sample_asset("a1", 60, 5, "video", 1000);
        let result = engine.evaluate_with_holds(&a1, 200);
        assert_eq!(result.action, Some(RetentionAction::Delete));
    }
}
