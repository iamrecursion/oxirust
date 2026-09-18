//! Storage lifecycle management
//!
//! Manages automatic object transitions between storage tiers and calculates
//! potential cost savings from lifecycle policies.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Storage tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StorageTier {
    /// Frequently accessed data – highest cost, instant retrieval
    Hot,
    /// Infrequently accessed data
    Warm,
    /// Rarely accessed data – low cost, minutes-to-hours retrieval
    Cold,
    /// Long-term archival – lowest cost, hours retrieval
    Archive,
}

impl StorageTier {
    /// Approximate cost per GB per month (USD)
    #[must_use]
    pub fn cost_per_gb_month(self) -> f64 {
        match self {
            StorageTier::Hot => 0.023,
            StorageTier::Warm => 0.0125,
            StorageTier::Cold => 0.004,
            StorageTier::Archive => 0.00099,
        }
    }

    /// Typical retrieval latency in seconds
    #[must_use]
    pub fn retrieval_time_secs(self) -> u64 {
        match self {
            StorageTier::Hot => 0,
            StorageTier::Warm => 5,
            StorageTier::Cold => 300,       // ~5 minutes
            StorageTier::Archive => 14_400, // ~4 hours
        }
    }

    /// Human-readable label
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            StorageTier::Hot => "Hot",
            StorageTier::Warm => "Warm",
            StorageTier::Cold => "Cold",
            StorageTier::Archive => "Archive",
        }
    }
}

/// A transition from the current tier to a target tier after a number of days
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierTransition {
    /// Days since object creation before the transition takes effect
    pub after_days: u32,
    /// Target storage tier
    pub target_tier: StorageTier,
}

impl TierTransition {
    /// Create a new tier transition
    #[must_use]
    pub fn new(after_days: u32, target_tier: StorageTier) -> Self {
        Self {
            after_days,
            target_tier,
        }
    }
}

/// A single lifecycle rule, scoped to an object key prefix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleRule {
    /// Unique rule identifier
    pub id: String,
    /// Key prefix this rule applies to (empty string matches all keys)
    pub prefix: String,
    /// Ordered list of tier transitions
    pub transitions: Vec<TierTransition>,
    /// Delete objects after this many days (optional)
    pub expiration_days: Option<u32>,
    /// Whether the rule is active
    pub enabled: bool,
}

impl LifecycleRule {
    /// Create a new enabled rule
    #[must_use]
    pub fn new(id: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prefix: prefix.into(),
            transitions: Vec::new(),
            expiration_days: None,
            enabled: true,
        }
    }

    /// Add a tier transition to this rule
    pub fn add_transition(&mut self, transition: TierTransition) {
        self.transitions.push(transition);
        // Keep transitions sorted ascending by after_days
        self.transitions.sort_by_key(|t| t.after_days);
    }

    /// Set expiration
    pub fn set_expiration(&mut self, days: u32) {
        self.expiration_days = Some(days);
    }

    /// Whether this rule applies to the given object key
    #[must_use]
    pub fn applies_to(&self, key: &str) -> bool {
        self.enabled && key.starts_with(&self.prefix)
    }
}

/// A full lifecycle policy containing one or more rules
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecyclePolicy {
    /// The rules that make up this policy
    pub rules: Vec<LifecycleRule>,
}

impl LifecyclePolicy {
    /// Create an empty policy
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule to the policy
    pub fn add_rule(&mut self, rule: LifecycleRule) {
        self.rules.push(rule);
    }

    /// Remove a rule by ID.  Returns `true` if a rule was removed.
    pub fn remove_rule(&mut self, rule_id: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != rule_id);
        self.rules.len() < before
    }

    /// Return all rules that apply to the given object key
    #[must_use]
    pub fn find_applicable_rules(&self, key: &str) -> Vec<&LifecycleRule> {
        self.rules.iter().filter(|r| r.applies_to(key)).collect()
    }
}

/// Represents a stored object whose tier may need to be managed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageObject {
    /// Object key (path)
    pub key: String,
    /// Object size in bytes
    pub size_bytes: u64,
    /// How many days ago the object was last modified
    pub last_modified_days_ago: u32,
    /// Tier the object is currently stored in
    pub current_tier: StorageTier,
}

impl StorageObject {
    /// Create a new storage object
    #[must_use]
    pub fn new(
        key: impl Into<String>,
        size_bytes: u64,
        last_modified_days_ago: u32,
        current_tier: StorageTier,
    ) -> Self {
        Self {
            key: key.into(),
            size_bytes,
            last_modified_days_ago,
            current_tier,
        }
    }

    /// Size in GB
    #[must_use]
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Monthly storage cost in USD at the current tier
    #[must_use]
    pub fn current_monthly_cost(&self) -> f64 {
        self.size_gb() * self.current_tier.cost_per_gb_month()
    }
}

/// Evaluates lifecycle policies against objects and estimates savings
pub struct LifecycleEngine;

impl LifecycleEngine {
    /// Determine the target tier for an object given a policy, or `None` if no
    /// transition should be applied yet.
    ///
    /// The highest-priority applicable transition (the one with the largest
    /// `after_days` that still applies) is selected.
    #[must_use]
    pub fn evaluate_object(obj: &StorageObject, policy: &LifecyclePolicy) -> Option<StorageTier> {
        let age = obj.last_modified_days_ago;
        let rules = policy.find_applicable_rules(&obj.key);

        // Collect all transitions that have been reached and pick the target
        // with the highest relative cost reduction (i.e. lowest cost tier
        // among eligible transitions).
        let mut best: Option<StorageTier> = None;
        for rule in rules {
            for transition in &rule.transitions {
                if age >= transition.after_days {
                    // Only move to a *cooler* tier (avoid unintended promotion)
                    if transition.target_tier > obj.current_tier {
                        match best {
                            None => best = Some(transition.target_tier),
                            Some(current_best) if transition.target_tier > current_best => {
                                best = Some(transition.target_tier);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        best
    }

    /// Estimate the monthly cost savings (USD) if the given policy were applied
    /// to all objects.
    #[must_use]
    pub fn estimate_savings(objects: &[StorageObject], policy: &LifecyclePolicy) -> f64 {
        objects
            .iter()
            .filter_map(|obj| {
                let target = Self::evaluate_object(obj, policy)?;
                if target == obj.current_tier {
                    return None;
                }
                let current_cost = obj.size_gb() * obj.current_tier.cost_per_gb_month();
                let future_cost = obj.size_gb() * target.cost_per_gb_month();
                let saving = current_cost - future_cost;
                if saving > 0.0 {
                    Some(saving)
                } else {
                    None
                }
            })
            .sum()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. StorageTier ordering
    #[test]
    fn test_tier_ordering() {
        assert!(StorageTier::Hot < StorageTier::Warm);
        assert!(StorageTier::Warm < StorageTier::Cold);
        assert!(StorageTier::Cold < StorageTier::Archive);
    }

    // 2. cost_per_gb_month descends with tier
    #[test]
    fn test_tier_cost_descends() {
        assert!(StorageTier::Hot.cost_per_gb_month() > StorageTier::Warm.cost_per_gb_month());
        assert!(StorageTier::Warm.cost_per_gb_month() > StorageTier::Cold.cost_per_gb_month());
        assert!(StorageTier::Cold.cost_per_gb_month() > StorageTier::Archive.cost_per_gb_month());
    }

    // 3. retrieval_time_secs increases with tier coldness
    #[test]
    fn test_retrieval_time_increases() {
        assert!(StorageTier::Hot.retrieval_time_secs() < StorageTier::Warm.retrieval_time_secs());
        assert!(StorageTier::Warm.retrieval_time_secs() < StorageTier::Cold.retrieval_time_secs());
        assert!(
            StorageTier::Cold.retrieval_time_secs() < StorageTier::Archive.retrieval_time_secs()
        );
    }

    // 4. LifecycleRule::applies_to prefix matching
    #[test]
    fn test_rule_prefix_match() {
        let rule = LifecycleRule::new("r1", "videos/");
        assert!(rule.applies_to("videos/2024/film.mp4"));
        assert!(!rule.applies_to("images/photo.jpg"));
    }

    // 5. Empty prefix matches everything
    #[test]
    fn test_rule_empty_prefix_matches_all() {
        let rule = LifecycleRule::new("r-all", "");
        assert!(rule.applies_to("anything/at/all"));
    }

    // 6. Disabled rule does not match
    #[test]
    fn test_rule_disabled() {
        let mut rule = LifecycleRule::new("r1", "");
        rule.enabled = false;
        assert!(!rule.applies_to("file.mp4"));
    }

    // 7. LifecyclePolicy add / remove rule
    #[test]
    fn test_policy_add_remove() {
        let mut policy = LifecyclePolicy::new();
        policy.add_rule(LifecycleRule::new("r1", "videos/"));
        policy.add_rule(LifecycleRule::new("r2", "audio/"));
        assert_eq!(policy.rules.len(), 2);

        let removed = policy.remove_rule("r1");
        assert!(removed);
        assert_eq!(policy.rules.len(), 1);

        // Removing non-existent rule returns false
        let not_removed = policy.remove_rule("does-not-exist");
        assert!(!not_removed);
    }

    // 8. find_applicable_rules
    #[test]
    fn test_find_applicable_rules() {
        let mut policy = LifecyclePolicy::new();
        policy.add_rule(LifecycleRule::new("r1", "videos/"));
        policy.add_rule(LifecycleRule::new("r2", "images/"));

        let applicable = policy.find_applicable_rules("videos/2024/clip.mp4");
        assert_eq!(applicable.len(), 1);
        assert_eq!(applicable[0].id, "r1");
    }

    // 9. LifecycleEngine::evaluate_object – transition applied
    #[test]
    fn test_evaluate_object_transition() {
        let mut rule = LifecycleRule::new("r1", "");
        rule.add_transition(TierTransition::new(30, StorageTier::Cold));

        let mut policy = LifecyclePolicy::new();
        policy.add_rule(rule);

        let obj = StorageObject::new("old/file.mp4", 1_000_000, 45, StorageTier::Hot);
        let target = LifecycleEngine::evaluate_object(&obj, &policy);
        assert_eq!(target, Some(StorageTier::Cold));
    }

    // 10. evaluate_object – no transition when object is too young
    #[test]
    fn test_evaluate_object_no_transition_young() {
        let mut rule = LifecycleRule::new("r1", "");
        rule.add_transition(TierTransition::new(30, StorageTier::Cold));

        let mut policy = LifecyclePolicy::new();
        policy.add_rule(rule);

        let obj = StorageObject::new("new/file.mp4", 1_000_000, 10, StorageTier::Hot);
        assert!(LifecycleEngine::evaluate_object(&obj, &policy).is_none());
    }

    // 11. estimate_savings returns a positive value when transitions apply
    #[test]
    fn test_estimate_savings_positive() {
        let mut rule = LifecycleRule::new("r1", "");
        rule.add_transition(TierTransition::new(30, StorageTier::Archive));

        let mut policy = LifecyclePolicy::new();
        policy.add_rule(rule);

        // 100 GB of hot data older than 30 days
        let objects = vec![StorageObject::new(
            "archive/big_file.mp4",
            100 * 1024 * 1024 * 1024,
            60,
            StorageTier::Hot,
        )];

        let savings = LifecycleEngine::estimate_savings(&objects, &policy);
        assert!(savings > 0.0, "Expected positive savings, got {savings}");
    }

    // 12. StorageObject size_gb and monthly cost
    #[test]
    fn test_storage_object_cost() {
        // 1 GB at hot tier
        let obj = StorageObject::new("file.mp4", 1024 * 1024 * 1024, 0, StorageTier::Hot);
        let cost = obj.current_monthly_cost();
        assert!((cost - 0.023).abs() < 1e-9);
    }
}
