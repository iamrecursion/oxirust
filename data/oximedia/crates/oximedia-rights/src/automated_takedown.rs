//! Automated takedown module for triggering actions when rights expire.
//!
//! Provides a rule-based engine that evaluates rights records against
//! configurable triggers and generates [`TakedownAction`]s that downstream
//! systems can execute (e.g. CDN purge, CMS unpublish, DRM revocation).
//!
//! # Design
//!
//! 1. Register [`TakedownRule`]s describing **when** to act.
//! 2. Call [`TakedownEngine::evaluate`] with a snapshot of rights records and
//!    the current timestamp.
//! 3. The engine returns a list of [`TakedownAction`]s that are ready to fire.
//! 4. Callers execute or queue the actions; the engine itself is stateless.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── ActionKind ────────────────────────────────────────────────────────────────

/// The type of automated action to take.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TakedownActionKind {
    /// Purge the asset from all CDN edge caches.
    PurgeCdn,
    /// Unpublish the asset from the CMS / media platform.
    UnpublishCms,
    /// Revoke DRM licenses associated with the asset.
    RevokeDrm,
    /// Send a notification e-mail to the rights holder.
    NotifyRightsHolder,
    /// Send a notification to the operations team.
    NotifyOps,
    /// Mark the asset as restricted in the database.
    MarkRestricted,
    /// Custom action identified by a string.
    Custom(String),
}

impl TakedownActionKind {
    /// Human-readable description.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::PurgeCdn => "Purge CDN caches",
            Self::UnpublishCms => "Unpublish from CMS",
            Self::RevokeDrm => "Revoke DRM licenses",
            Self::NotifyRightsHolder => "Notify rights holder",
            Self::NotifyOps => "Notify operations team",
            Self::MarkRestricted => "Mark asset restricted",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ── TriggerCondition ──────────────────────────────────────────────────────────

/// Conditions under which a takedown rule fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Fire when the right has already expired (expired_at <= now).
    OnExpiry,
    /// Fire N seconds before expiry (lookahead warning).
    BeforeExpiry {
        /// Seconds before expiry to fire.
        seconds_before: u64,
    },
    /// Fire when the record is explicitly marked inactive.
    OnDeactivation,
    /// Fire when expiry is unknown / absent.
    OnMissingExpiry,
}

impl TriggerCondition {
    /// Evaluate whether this condition is satisfied for a given record.
    ///
    /// `expires_at` is `None` for records with no expiry.
    /// `active` is the current active flag of the record.
    #[must_use]
    pub fn is_satisfied(&self, expires_at: Option<u64>, active: bool, now: u64) -> bool {
        match self {
            Self::OnExpiry => expires_at.map_or(false, |exp| now >= exp),
            Self::BeforeExpiry { seconds_before } => expires_at.map_or(false, |exp| {
                now < exp && exp.saturating_sub(*seconds_before) <= now
            }),
            Self::OnDeactivation => !active,
            Self::OnMissingExpiry => expires_at.is_none(),
        }
    }
}

// ── TakedownRule ──────────────────────────────────────────────────────────────

/// A rule that maps a [`TriggerCondition`] to one or more [`TakedownActionKind`]s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakedownRule {
    /// Unique rule identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// The condition that triggers this rule.
    pub condition: TriggerCondition,
    /// Actions to perform when the condition is satisfied.
    pub actions: Vec<TakedownActionKind>,
    /// If `true`, the rule only applies to assets matching one of these IDs.
    /// Empty = applies to all assets.
    pub asset_filter: Vec<String>,
    /// If `true`, the rule only fires for active records.
    pub active_only: bool,
    /// Priority (lower number = evaluated first).
    pub priority: u32,
}

impl TakedownRule {
    /// Create a new rule.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        condition: TriggerCondition,
        actions: Vec<TakedownActionKind>,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            condition,
            actions,
            asset_filter: Vec::new(),
            active_only: false,
            priority: 100,
        }
    }

    /// Builder: restrict to specific assets.
    #[must_use]
    pub fn with_asset_filter(mut self, assets: Vec<String>) -> Self {
        self.asset_filter = assets;
        self
    }

    /// Builder: only fire for active records.
    #[must_use]
    pub fn active_only(mut self) -> Self {
        self.active_only = true;
        self
    }

    /// Builder: set priority.
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Whether this rule applies to a given asset.
    #[must_use]
    pub fn applies_to_asset(&self, asset_id: &str) -> bool {
        self.asset_filter.is_empty() || self.asset_filter.iter().any(|a| a == asset_id)
    }
}

// ── TakedownAction ────────────────────────────────────────────────────────────

/// A generated action ready to be executed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakedownAction {
    /// ID of the rule that generated this action.
    pub rule_id: String,
    /// Asset to act on.
    pub asset_id: String,
    /// Rights record that triggered the action.
    pub record_id: String,
    /// The action to perform.
    pub kind: TakedownActionKind,
    /// Timestamp at which this action was generated.
    pub generated_at: u64,
    /// Human-readable reason string.
    pub reason: String,
}

// ── RightsSnapshot ────────────────────────────────────────────────────────────

/// A snapshot of a rights record for engine evaluation.
#[derive(Debug, Clone)]
pub struct RightsSnapshot {
    /// Record identifier.
    pub record_id: String,
    /// Asset identifier.
    pub asset_id: String,
    /// Whether the record is currently active.
    pub active: bool,
    /// Optional expiry timestamp (Unix seconds).
    pub expires_at: Option<u64>,
}

impl RightsSnapshot {
    /// Create a new snapshot.
    #[must_use]
    pub fn new(
        record_id: impl Into<String>,
        asset_id: impl Into<String>,
        active: bool,
        expires_at: Option<u64>,
    ) -> Self {
        Self {
            record_id: record_id.into(),
            asset_id: asset_id.into(),
            active,
            expires_at,
        }
    }
}

// ── TakedownEngine ────────────────────────────────────────────────────────────

/// Stateless rules engine that evaluates rights snapshots against takedown rules.
#[derive(Debug, Default)]
pub struct TakedownEngine {
    rules: Vec<TakedownRule>,
    /// Tracks which (record_id, rule_id) pairs have already fired to avoid
    /// duplicate actions on repeated calls.
    fired: HashMap<(String, String), u64>,
}

impl TakedownEngine {
    /// Create a new engine with no rules.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a rule.
    pub fn add_rule(&mut self, rule: TakedownRule) {
        self.rules.push(rule);
        // Keep sorted by priority (ascending).
        self.rules.sort_by_key(|r| r.priority);
    }

    /// Number of registered rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Evaluate all snapshots against all rules at `now`.
    ///
    /// Returns all [`TakedownAction`]s that should be executed.
    /// Actions for (record_id, rule_id) pairs that have already fired since the
    /// last [`clear_fired`](TakedownEngine::clear_fired) call are suppressed.
    pub fn evaluate(&mut self, snapshots: &[RightsSnapshot], now: u64) -> Vec<TakedownAction> {
        let mut actions = Vec::new();

        for snapshot in snapshots {
            for rule in &self.rules {
                // Asset filter
                if !rule.applies_to_asset(&snapshot.asset_id) {
                    continue;
                }

                // Active-only filter
                if rule.active_only && !snapshot.active {
                    continue;
                }

                // Condition evaluation
                if !rule
                    .condition
                    .is_satisfied(snapshot.expires_at, snapshot.active, now)
                {
                    continue;
                }

                // Deduplication check
                let key = (snapshot.record_id.clone(), rule.id.clone());
                if self.fired.contains_key(&key) {
                    continue;
                }
                self.fired.insert(key, now);

                let reason = format!(
                    "Rule '{}' triggered for record '{}' (asset '{}') at ts={}",
                    rule.id, snapshot.record_id, snapshot.asset_id, now,
                );

                for action_kind in &rule.actions {
                    actions.push(TakedownAction {
                        rule_id: rule.id.clone(),
                        asset_id: snapshot.asset_id.clone(),
                        record_id: snapshot.record_id.clone(),
                        kind: action_kind.clone(),
                        generated_at: now,
                        reason: reason.clone(),
                    });
                }
            }
        }

        actions
    }

    /// Clear the deduplication state so previously-fired rules can fire again.
    pub fn clear_fired(&mut self) {
        self.fired.clear();
    }

    /// Whether a given (record_id, rule_id) pair has already fired.
    #[must_use]
    pub fn has_fired(&self, record_id: &str, rule_id: &str) -> bool {
        self.fired
            .contains_key(&(record_id.to_string(), rule_id.to_string()))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn expiry_rule() -> TakedownRule {
        TakedownRule::new(
            "rule-expire",
            "Takedown on expiry",
            TriggerCondition::OnExpiry,
            vec![
                TakedownActionKind::PurgeCdn,
                TakedownActionKind::UnpublishCms,
            ],
        )
    }

    fn warning_rule() -> TakedownRule {
        TakedownRule::new(
            "rule-warn",
            "Warn 24h before expiry",
            TriggerCondition::BeforeExpiry {
                seconds_before: 86_400,
            },
            vec![TakedownActionKind::NotifyOps],
        )
    }

    fn active_only_rule() -> TakedownRule {
        TakedownRule::new(
            "rule-deactivated",
            "Notify when deactivated",
            TriggerCondition::OnDeactivation,
            vec![TakedownActionKind::NotifyRightsHolder],
        )
        .active_only()
    }

    // ── TriggerCondition ──

    #[test]
    fn test_on_expiry_fires_after() {
        let cond = TriggerCondition::OnExpiry;
        assert!(cond.is_satisfied(Some(100), true, 100));
        assert!(cond.is_satisfied(Some(100), true, 200));
    }

    #[test]
    fn test_on_expiry_does_not_fire_before() {
        let cond = TriggerCondition::OnExpiry;
        assert!(!cond.is_satisfied(Some(100), true, 50));
    }

    #[test]
    fn test_on_expiry_no_expiry() {
        let cond = TriggerCondition::OnExpiry;
        assert!(!cond.is_satisfied(None, true, 9999));
    }

    #[test]
    fn test_before_expiry_in_window() {
        let cond = TriggerCondition::BeforeExpiry {
            seconds_before: 86_400,
        };
        // Expiry at 200_000; now at 200_000 - 50_000 (within 86400 window)
        assert!(cond.is_satisfied(Some(200_000), true, 200_000 - 50_000));
    }

    #[test]
    fn test_before_expiry_too_early() {
        let cond = TriggerCondition::BeforeExpiry {
            seconds_before: 86_400,
        };
        // Expiry at 200_000; now at 200_000 - 100_000 (outside 86400 window)
        assert!(!cond.is_satisfied(Some(200_000), true, 200_000 - 100_000));
    }

    #[test]
    fn test_before_expiry_already_expired() {
        let cond = TriggerCondition::BeforeExpiry {
            seconds_before: 86_400,
        };
        assert!(!cond.is_satisfied(Some(100), true, 200));
    }

    #[test]
    fn test_on_deactivation() {
        let cond = TriggerCondition::OnDeactivation;
        assert!(cond.is_satisfied(None, false, 0));
        assert!(!cond.is_satisfied(None, true, 0));
    }

    #[test]
    fn test_on_missing_expiry() {
        let cond = TriggerCondition::OnMissingExpiry;
        assert!(cond.is_satisfied(None, true, 0));
        assert!(!cond.is_satisfied(Some(1000), true, 0));
    }

    // ── TakedownRule ──

    #[test]
    fn test_rule_asset_filter_empty_applies_all() {
        let rule = expiry_rule();
        assert!(rule.applies_to_asset("any-asset"));
    }

    #[test]
    fn test_rule_asset_filter_specific() {
        let rule = expiry_rule().with_asset_filter(vec!["asset-A".to_string()]);
        assert!(rule.applies_to_asset("asset-A"));
        assert!(!rule.applies_to_asset("asset-B"));
    }

    // ── TakedownEngine ──

    #[test]
    fn test_evaluate_expired_record_fires() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(expiry_rule());

        let snap = RightsSnapshot::new("r1", "asset-A", true, Some(100));
        let actions = engine.evaluate(&[snap], 200);
        assert_eq!(actions.len(), 2); // PurgeCdn + UnpublishCms
    }

    #[test]
    fn test_evaluate_not_expired_no_actions() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(expiry_rule());

        let snap = RightsSnapshot::new("r1", "asset-A", true, Some(10_000));
        let actions = engine.evaluate(&[snap], 100);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_evaluate_deduplication() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(expiry_rule());

        let snap = RightsSnapshot::new("r1", "asset-A", true, Some(100));
        let first = engine.evaluate(std::slice::from_ref(&snap), 200);
        let second = engine.evaluate(std::slice::from_ref(&snap), 200);
        assert_eq!(first.len(), 2);
        assert!(second.is_empty()); // already fired
    }

    #[test]
    fn test_clear_fired_re_enables_rule() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(expiry_rule());

        let snap = RightsSnapshot::new("r1", "asset-A", true, Some(100));
        engine.evaluate(std::slice::from_ref(&snap), 200);
        engine.clear_fired();
        let second = engine.evaluate(std::slice::from_ref(&snap), 200);
        assert_eq!(second.len(), 2);
    }

    #[test]
    fn test_has_fired() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(expiry_rule());

        let snap = RightsSnapshot::new("r1", "asset-A", true, Some(100));
        assert!(!engine.has_fired("r1", "rule-expire"));
        engine.evaluate(&[snap], 200);
        assert!(engine.has_fired("r1", "rule-expire"));
    }

    #[test]
    fn test_warning_rule_fires_in_window() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(warning_rule());

        // Expires at 200_000; now at 200_000 - 50_000 (within 86400s window)
        let snap = RightsSnapshot::new("r1", "asset-A", true, Some(200_000));
        let actions = engine.evaluate(&[snap], 200_000 - 50_000);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, TakedownActionKind::NotifyOps);
    }

    #[test]
    fn test_active_only_rule_skips_active_records() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(active_only_rule());

        // active_only rule uses OnDeactivation, so it only fires for inactive.
        // But with active_only=true and record.active=true, it should be skipped.
        let snap = RightsSnapshot::new("r1", "asset-A", true, None);
        let actions = engine.evaluate(&[snap], 0);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_action_kind_description() {
        assert_eq!(
            TakedownActionKind::PurgeCdn.description(),
            "Purge CDN caches"
        );
        assert_eq!(
            TakedownActionKind::NotifyOps.description(),
            "Notify operations team"
        );
        assert_eq!(
            TakedownActionKind::Custom("foo".into()).description(),
            "foo"
        );
    }

    #[test]
    fn test_rule_count() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(expiry_rule());
        engine.add_rule(warning_rule());
        assert_eq!(engine.rule_count(), 2);
    }

    #[test]
    fn test_multiple_rules_multiple_actions() {
        let mut engine = TakedownEngine::new();
        engine.add_rule(expiry_rule());
        engine.add_rule(warning_rule());

        // Expired record – only expiry rule fires (warning fires before expiry, not after)
        let snap = RightsSnapshot::new("r1", "asset-A", true, Some(100));
        let actions = engine.evaluate(&[snap], 200);
        // Only expiry rule: PurgeCdn + UnpublishCms
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_priority_ordering_affects_dedup() {
        // Two rules: one priority 10, one priority 200
        let r1 = TakedownRule::new(
            "high-prio",
            "High priority",
            TriggerCondition::OnExpiry,
            vec![TakedownActionKind::MarkRestricted],
        )
        .with_priority(10);

        let r2 = TakedownRule::new(
            "low-prio",
            "Low priority",
            TriggerCondition::OnExpiry,
            vec![TakedownActionKind::NotifyOps],
        )
        .with_priority(200);

        let mut engine = TakedownEngine::new();
        engine.add_rule(r2);
        engine.add_rule(r1);

        let snap = RightsSnapshot::new("r1", "a", true, Some(100));
        let actions = engine.evaluate(&[snap], 200);
        // Both rules fire (different rule IDs)
        assert_eq!(actions.len(), 2);
        // First action should be from the higher-priority rule
        assert_eq!(actions[0].rule_id, "high-prio");
    }
}
