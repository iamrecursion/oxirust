#![allow(dead_code)]
//! Cloud object lifecycle management.
//!
//! Defines lifecycle rules for automated object transitions between storage
//! classes, expiration policies, and archival scheduling for cloud-stored
//! media assets.

use std::collections::HashMap;

/// Cloud storage tier for lifecycle transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageTier {
    /// Hot / standard access — lowest latency, highest cost.
    Hot,
    /// Warm / infrequent access — moderate latency and cost.
    Warm,
    /// Cool / archive access — higher latency, lower cost.
    Cool,
    /// Cold / deep archive — highest latency, lowest cost.
    Cold,
    /// Glacier-like deep freeze — very high latency, minimal cost.
    DeepArchive,
}

impl StorageTier {
    /// Returns the estimated monthly cost per TB in dollars (approximate).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_cost_per_tb_month(&self) -> f64 {
        match self {
            Self::Hot => 23.0,
            Self::Warm => 12.5,
            Self::Cool => 4.0,
            Self::Cold => 1.0,
            Self::DeepArchive => 0.4,
        }
    }

    /// Returns the estimated retrieval time in hours.
    #[must_use]
    pub fn estimated_retrieval_hours(&self) -> f64 {
        match self {
            Self::Hot => 0.0,
            Self::Warm => 0.0,
            Self::Cool => 0.1,
            Self::Cold => 3.0,
            Self::DeepArchive => 12.0,
        }
    }
}

impl std::fmt::Display for StorageTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hot => write!(f, "hot"),
            Self::Warm => write!(f, "warm"),
            Self::Cool => write!(f, "cool"),
            Self::Cold => write!(f, "cold"),
            Self::DeepArchive => write!(f, "deep-archive"),
        }
    }
}

/// Lifecycle action to perform on matching objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleAction {
    /// Transition to a different storage tier.
    Transition(StorageTier),
    /// Delete the object.
    Expire,
    /// Mark the object as non-current (for versioned buckets).
    MarkNonCurrent,
    /// Tag the object with specified metadata.
    Tag(String, String),
}

impl std::fmt::Display for LifecycleAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transition(tier) => write!(f, "transition to {tier}"),
            Self::Expire => write!(f, "expire/delete"),
            Self::MarkNonCurrent => write!(f, "mark non-current"),
            Self::Tag(k, v) => write!(f, "tag {k}={v}"),
        }
    }
}

/// Condition that triggers a lifecycle rule.
#[derive(Debug, Clone)]
pub struct LifecycleCondition {
    /// Minimum age in days since creation.
    pub age_days: Option<u64>,
    /// Minimum age in days since last access.
    pub days_since_last_access: Option<u64>,
    /// Object size threshold in bytes (apply rule if object is larger).
    pub min_size_bytes: Option<u64>,
    /// Object size threshold in bytes (apply rule if object is smaller).
    pub max_size_bytes: Option<u64>,
    /// Path prefix filter.
    pub prefix: Option<String>,
    /// Required tag key-value match.
    pub tag_match: Option<(String, String)>,
}

impl LifecycleCondition {
    /// Creates a new empty condition (matches everything).
    #[must_use]
    pub fn new() -> Self {
        Self {
            age_days: None,
            days_since_last_access: None,
            min_size_bytes: None,
            max_size_bytes: None,
            prefix: None,
            tag_match: None,
        }
    }

    /// Sets the minimum age in days.
    #[must_use]
    pub fn with_age_days(mut self, days: u64) -> Self {
        self.age_days = Some(days);
        self
    }

    /// Sets the days since last access threshold.
    #[must_use]
    pub fn with_days_since_access(mut self, days: u64) -> Self {
        self.days_since_last_access = Some(days);
        self
    }

    /// Sets the minimum size filter.
    #[must_use]
    pub fn with_min_size(mut self, bytes: u64) -> Self {
        self.min_size_bytes = Some(bytes);
        self
    }

    /// Sets the prefix filter.
    #[must_use]
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = Some(prefix.to_string());
        self
    }

    /// Evaluates whether an object matches this condition.
    #[must_use]
    pub fn matches(&self, obj: &ObjectState) -> bool {
        if let Some(age) = self.age_days {
            if obj.age_days < age {
                return false;
            }
        }
        if let Some(access_days) = self.days_since_last_access {
            if obj.days_since_last_access < access_days {
                return false;
            }
        }
        if let Some(min_size) = self.min_size_bytes {
            if obj.size_bytes < min_size {
                return false;
            }
        }
        if let Some(max_size) = self.max_size_bytes {
            if obj.size_bytes > max_size {
                return false;
            }
        }
        if let Some(ref prefix) = self.prefix {
            if !obj.key.starts_with(prefix) {
                return false;
            }
        }
        if let Some((ref tag_key, ref tag_value)) = self.tag_match {
            match obj.tags.get(tag_key.as_str()) {
                Some(v) if v == tag_value => {}
                _ => return false,
            }
        }
        true
    }
}

impl Default for LifecycleCondition {
    fn default() -> Self {
        Self::new()
    }
}

/// A single lifecycle rule.
#[derive(Debug, Clone)]
pub struct LifecycleRule {
    /// Rule name / identifier.
    pub name: String,
    /// Whether the rule is enabled.
    pub enabled: bool,
    /// Condition that triggers the rule.
    pub condition: LifecycleCondition,
    /// Action to perform when condition matches.
    pub action: LifecycleAction,
    /// Priority (lower = higher priority, evaluated first).
    pub priority: u32,
}

impl LifecycleRule {
    /// Creates a new lifecycle rule.
    #[must_use]
    pub fn new(name: &str, condition: LifecycleCondition, action: LifecycleAction) -> Self {
        Self {
            name: name.to_string(),
            enabled: true,
            condition,
            action,
            priority: 100,
        }
    }

    /// Sets the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Disables this rule.
    #[must_use]
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// Current state of a cloud object for lifecycle evaluation.
#[derive(Debug, Clone)]
pub struct ObjectState {
    /// Object key / path.
    pub key: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Current storage tier.
    pub current_tier: StorageTier,
    /// Age in days since creation.
    pub age_days: u64,
    /// Days since last access.
    pub days_since_last_access: u64,
    /// Object tags.
    pub tags: HashMap<String, String>,
}

impl ObjectState {
    /// Creates a new object state.
    #[must_use]
    pub fn new(key: &str, size_bytes: u64, current_tier: StorageTier) -> Self {
        Self {
            key: key.to_string(),
            size_bytes,
            current_tier,
            age_days: 0,
            days_since_last_access: 0,
            tags: HashMap::new(),
        }
    }

    /// Sets the age in days.
    #[must_use]
    pub fn with_age(mut self, days: u64) -> Self {
        self.age_days = days;
        self
    }

    /// Sets the days since last access.
    #[must_use]
    pub fn with_last_access(mut self, days: u64) -> Self {
        self.days_since_last_access = days;
        self
    }

    /// Adds a tag.
    #[must_use]
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }
}

/// Result of evaluating lifecycle rules against an object.
#[derive(Debug, Clone)]
pub struct LifecycleEvaluation {
    /// Object key.
    pub key: String,
    /// Actions to apply (rule name -> action).
    pub actions: Vec<(String, LifecycleAction)>,
}

impl LifecycleEvaluation {
    /// Returns whether any actions were triggered.
    #[must_use]
    pub fn has_actions(&self) -> bool {
        !self.actions.is_empty()
    }

    /// Returns the number of actions.
    #[must_use]
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }
}

/// Lifecycle policy engine.
///
/// Evaluates lifecycle rules against cloud objects and determines which
/// actions should be applied.
#[derive(Debug, Clone)]
pub struct LifecycleEngine {
    /// Rules managed by this engine, sorted by priority.
    rules: Vec<LifecycleRule>,
}

impl LifecycleEngine {
    /// Creates a new empty lifecycle engine.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Adds a rule to the engine.
    pub fn add_rule(&mut self, rule: LifecycleRule) {
        self.rules.push(rule);
        self.rules.sort_by_key(|r| r.priority);
    }

    /// Returns the number of rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Evaluates all rules against a single object.
    #[must_use]
    pub fn evaluate(&self, obj: &ObjectState) -> LifecycleEvaluation {
        let mut eval = LifecycleEvaluation {
            key: obj.key.clone(),
            actions: Vec::new(),
        };

        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            if rule.condition.matches(obj) {
                // Skip transitions to the same or lower tier
                if let LifecycleAction::Transition(target) = &rule.action {
                    if *target <= obj.current_tier {
                        continue;
                    }
                }
                eval.actions.push((rule.name.clone(), rule.action.clone()));
            }
        }

        eval
    }

    /// Evaluates all rules against a batch of objects.
    #[must_use]
    pub fn evaluate_batch(&self, objects: &[ObjectState]) -> Vec<LifecycleEvaluation> {
        objects.iter().map(|obj| self.evaluate(obj)).collect()
    }

    /// Estimates monthly cost savings from applying lifecycle transitions.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_savings(&self, objects: &[ObjectState]) -> f64 {
        let evals = self.evaluate_batch(objects);
        let mut savings = 0.0_f64;

        for (eval, obj) in evals.iter().zip(objects.iter()) {
            for (_, action) in &eval.actions {
                if let LifecycleAction::Transition(target) = action {
                    let current_cost = obj.current_tier.estimated_cost_per_tb_month();
                    let target_cost = target.estimated_cost_per_tb_month();
                    let size_tb = obj.size_bytes as f64 / 1_099_511_627_776.0;
                    savings += (current_cost - target_cost) * size_tb;
                }
            }
        }

        savings
    }
}

impl Default for LifecycleEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a standard media lifecycle policy with typical tier transitions.
#[must_use]
pub fn standard_media_policy() -> LifecycleEngine {
    let mut engine = LifecycleEngine::new();

    // Move to warm after 30 days of no access
    engine.add_rule(
        LifecycleRule::new(
            "warm-after-30d",
            LifecycleCondition::new().with_days_since_access(30),
            LifecycleAction::Transition(StorageTier::Warm),
        )
        .with_priority(10),
    );

    // Move to cool after 90 days of no access
    engine.add_rule(
        LifecycleRule::new(
            "cool-after-90d",
            LifecycleCondition::new().with_days_since_access(90),
            LifecycleAction::Transition(StorageTier::Cool),
        )
        .with_priority(20),
    );

    // Move to cold after 180 days of no access
    engine.add_rule(
        LifecycleRule::new(
            "cold-after-180d",
            LifecycleCondition::new().with_days_since_access(180),
            LifecycleAction::Transition(StorageTier::Cold),
        )
        .with_priority(30),
    );

    // Deep archive after 365 days
    engine.add_rule(
        LifecycleRule::new(
            "archive-after-365d",
            LifecycleCondition::new().with_days_since_access(365),
            LifecycleAction::Transition(StorageTier::DeepArchive),
        )
        .with_priority(40),
    );

    engine
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_tier_ordering() {
        assert!(StorageTier::Hot < StorageTier::Warm);
        assert!(StorageTier::Warm < StorageTier::Cool);
        assert!(StorageTier::Cool < StorageTier::Cold);
        assert!(StorageTier::Cold < StorageTier::DeepArchive);
    }

    #[test]
    fn test_storage_tier_display() {
        assert_eq!(StorageTier::Hot.to_string(), "hot");
        assert_eq!(StorageTier::Warm.to_string(), "warm");
        assert_eq!(StorageTier::Cool.to_string(), "cool");
        assert_eq!(StorageTier::Cold.to_string(), "cold");
        assert_eq!(StorageTier::DeepArchive.to_string(), "deep-archive");
    }

    #[test]
    fn test_storage_tier_cost() {
        assert!(
            StorageTier::Hot.estimated_cost_per_tb_month()
                > StorageTier::Cold.estimated_cost_per_tb_month()
        );
        assert!(
            StorageTier::Cold.estimated_cost_per_tb_month()
                > StorageTier::DeepArchive.estimated_cost_per_tb_month()
        );
    }

    #[test]
    fn test_lifecycle_action_display() {
        assert_eq!(
            LifecycleAction::Transition(StorageTier::Warm).to_string(),
            "transition to warm"
        );
        assert_eq!(LifecycleAction::Expire.to_string(), "expire/delete");
        assert_eq!(
            LifecycleAction::Tag("env".to_string(), "prod".to_string()).to_string(),
            "tag env=prod"
        );
    }

    #[test]
    fn test_condition_age_days() {
        let cond = LifecycleCondition::new().with_age_days(30);
        let young = ObjectState::new("a.mp4", 100, StorageTier::Hot).with_age(10);
        let old = ObjectState::new("b.mp4", 100, StorageTier::Hot).with_age(60);
        assert!(!cond.matches(&young));
        assert!(cond.matches(&old));
    }

    #[test]
    fn test_condition_prefix() {
        let cond = LifecycleCondition::new().with_prefix("media/");
        let matching = ObjectState::new("media/video.mp4", 100, StorageTier::Hot);
        let non_matching = ObjectState::new("logs/app.log", 100, StorageTier::Hot);
        assert!(cond.matches(&matching));
        assert!(!cond.matches(&non_matching));
    }

    #[test]
    fn test_condition_size_filter() {
        let cond = LifecycleCondition::new().with_min_size(1000);
        let small = ObjectState::new("small.txt", 500, StorageTier::Hot);
        let large = ObjectState::new("large.mp4", 5000, StorageTier::Hot);
        assert!(!cond.matches(&small));
        assert!(cond.matches(&large));
    }

    #[test]
    fn test_condition_empty_matches_all() {
        let cond = LifecycleCondition::new();
        let obj = ObjectState::new("anything.mp4", 100, StorageTier::Hot);
        assert!(cond.matches(&obj));
    }

    #[test]
    fn test_engine_evaluate_transition() {
        let mut engine = LifecycleEngine::new();
        engine.add_rule(LifecycleRule::new(
            "to-warm",
            LifecycleCondition::new().with_days_since_access(30),
            LifecycleAction::Transition(StorageTier::Warm),
        ));

        let obj = ObjectState::new("vid.mp4", 1000, StorageTier::Hot).with_last_access(60);
        let eval = engine.evaluate(&obj);
        assert!(eval.has_actions());
        assert_eq!(eval.action_count(), 1);
    }

    #[test]
    fn test_engine_skips_downgrade() {
        let mut engine = LifecycleEngine::new();
        engine.add_rule(LifecycleRule::new(
            "to-warm",
            LifecycleCondition::new().with_days_since_access(30),
            LifecycleAction::Transition(StorageTier::Warm),
        ));

        // Object already in Cool (higher than Warm) — should skip
        let obj = ObjectState::new("vid.mp4", 1000, StorageTier::Cool).with_last_access(60);
        let eval = engine.evaluate(&obj);
        assert!(!eval.has_actions());
    }

    #[test]
    fn test_engine_disabled_rule() {
        let mut engine = LifecycleEngine::new();
        engine.add_rule(
            LifecycleRule::new(
                "to-warm",
                LifecycleCondition::new(),
                LifecycleAction::Transition(StorageTier::Warm),
            )
            .disabled(),
        );

        let obj = ObjectState::new("vid.mp4", 1000, StorageTier::Hot);
        let eval = engine.evaluate(&obj);
        assert!(!eval.has_actions());
    }

    #[test]
    fn test_engine_batch_evaluate() {
        let mut engine = LifecycleEngine::new();
        engine.add_rule(LifecycleRule::new(
            "to-warm",
            LifecycleCondition::new().with_days_since_access(30),
            LifecycleAction::Transition(StorageTier::Warm),
        ));

        let objects = vec![
            ObjectState::new("a.mp4", 1000, StorageTier::Hot).with_last_access(10),
            ObjectState::new("b.mp4", 2000, StorageTier::Hot).with_last_access(60),
            ObjectState::new("c.mp4", 3000, StorageTier::Hot).with_last_access(90),
        ];
        let evals = engine.evaluate_batch(&objects);
        assert_eq!(evals.len(), 3);
        assert!(!evals[0].has_actions()); // too recent
        assert!(evals[1].has_actions());
        assert!(evals[2].has_actions());
    }

    #[test]
    fn test_standard_media_policy() {
        let engine = standard_media_policy();
        assert_eq!(engine.rule_count(), 4);

        // Object accessed 100 days ago should transition to cool (90d rule matches)
        let obj = ObjectState::new("old.mp4", 1_000_000, StorageTier::Hot).with_last_access(100);
        let eval = engine.evaluate(&obj);
        assert!(eval.has_actions());
    }

    #[test]
    fn test_estimate_savings() {
        let mut engine = LifecycleEngine::new();
        engine.add_rule(LifecycleRule::new(
            "to-cold",
            LifecycleCondition::new().with_days_since_access(30),
            LifecycleAction::Transition(StorageTier::Cold),
        ));

        let tb: u64 = 1_099_511_627_776; // 1 TB
        let objects = vec![ObjectState::new("a.mp4", tb, StorageTier::Hot).with_last_access(60)];
        let savings = engine.estimate_savings(&objects);
        // Hot=$23/TB/mo, Cold=$1/TB/mo => savings = $22/TB/mo
        assert!((savings - 22.0).abs() < 0.1);
    }

    #[test]
    fn test_rule_priority_ordering() {
        let mut engine = LifecycleEngine::new();
        engine.add_rule(
            LifecycleRule::new(
                "low-priority",
                LifecycleCondition::new(),
                LifecycleAction::Transition(StorageTier::Cold),
            )
            .with_priority(50),
        );
        engine.add_rule(
            LifecycleRule::new(
                "high-priority",
                LifecycleCondition::new(),
                LifecycleAction::Transition(StorageTier::Warm),
            )
            .with_priority(10),
        );

        // The high-priority rule should appear first in the engine's rules
        assert_eq!(engine.rules[0].name, "high-priority");
        assert_eq!(engine.rules[1].name, "low-priority");
    }

    #[test]
    fn test_object_state_with_tag() {
        let obj = ObjectState::new("vid.mp4", 100, StorageTier::Hot)
            .with_tag("env", "production")
            .with_tag("team", "media");
        assert_eq!(obj.tags.len(), 2);
        assert_eq!(obj.tags.get("env"), Some(&"production".to_string()));
    }

    #[test]
    fn test_retrieval_hours() {
        assert!((StorageTier::Hot.estimated_retrieval_hours()).abs() < f64::EPSILON);
        assert!(StorageTier::DeepArchive.estimated_retrieval_hours() > 0.0);
    }
}
