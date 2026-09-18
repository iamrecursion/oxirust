#![allow(dead_code)]
//! Affinity and anti-affinity rules for render node assignment.
//!
//! Controls which render nodes a job can or cannot be scheduled on, based on
//! labels, hardware capabilities, geographic zones, or explicit node lists.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// A label key-value pair used for matching nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeLabel {
    /// Label key (e.g., "gpu_type", "zone", "os").
    pub key: String,
    /// Label value (e.g., "a100", "us-east-1", "linux").
    pub value: String,
}

impl NodeLabel {
    /// Creates a new node label.
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

impl fmt::Display for NodeLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.key, self.value)
    }
}

/// Operator for matching a label value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchOperator {
    /// Exact equality.
    Equals,
    /// Not equal.
    NotEquals,
    /// Value is in a set of allowed values.
    In(Vec<String>),
    /// Value is not in a set of disallowed values.
    NotIn(Vec<String>),
    /// Key must exist (any value).
    Exists,
}

impl fmt::Display for MatchOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equals => write!(f, "=="),
            Self::NotEquals => write!(f, "!="),
            Self::In(vals) => write!(f, "in [{}]", vals.join(", ")),
            Self::NotIn(vals) => write!(f, "not in [{}]", vals.join(", ")),
            Self::Exists => write!(f, "exists"),
        }
    }
}

/// A single label-matching expression.
#[derive(Debug, Clone)]
pub struct LabelSelector {
    /// The label key to match on.
    pub key: String,
    /// The match operator.
    pub operator: MatchOperator,
    /// The value to compare against (used with Equals/NotEquals).
    pub value: String,
}

impl LabelSelector {
    /// Creates an equality selector.
    pub fn equals(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            operator: MatchOperator::Equals,
            value: value.into(),
        }
    }

    /// Creates a not-equal selector.
    pub fn not_equals(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            operator: MatchOperator::NotEquals,
            value: value.into(),
        }
    }

    /// Creates an "in" selector.
    pub fn is_in(key: impl Into<String>, values: Vec<String>) -> Self {
        Self {
            key: key.into(),
            operator: MatchOperator::In(values),
            value: String::new(),
        }
    }

    /// Creates an "exists" selector.
    pub fn exists(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            operator: MatchOperator::Exists,
            value: String::new(),
        }
    }

    /// Tests whether a set of node labels matches this selector.
    #[must_use]
    pub fn matches(&self, labels: &HashMap<String, String>) -> bool {
        match &self.operator {
            MatchOperator::Exists => labels.contains_key(&self.key),
            MatchOperator::Equals => labels.get(&self.key).map_or(false, |v| *v == self.value),
            MatchOperator::NotEquals => labels.get(&self.key).map_or(true, |v| *v != self.value),
            MatchOperator::In(allowed) => {
                labels.get(&self.key).map_or(false, |v| allowed.contains(v))
            }
            MatchOperator::NotIn(disallowed) => labels
                .get(&self.key)
                .map_or(true, |v| !disallowed.contains(v)),
        }
    }
}

/// Type of affinity rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinityType {
    /// Job prefers to run on matching nodes (soft).
    Preferred,
    /// Job must run on matching nodes (hard).
    Required,
}

impl fmt::Display for AffinityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preferred => write!(f, "preferred"),
            Self::Required => write!(f, "required"),
        }
    }
}

/// A complete affinity or anti-affinity rule.
#[derive(Debug, Clone)]
pub struct AffinityRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// Whether this is an affinity (attract) or anti-affinity (repel) rule.
    pub is_anti_affinity: bool,
    /// Strength of the rule (required vs preferred).
    pub affinity_type: AffinityType,
    /// Label selectors that must all match (AND logic).
    pub selectors: Vec<LabelSelector>,
    /// Weight for preferred rules (higher = stronger preference, 1..100).
    pub weight: u32,
}

impl AffinityRule {
    /// Creates a new required affinity rule.
    pub fn required_affinity(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            is_anti_affinity: false,
            affinity_type: AffinityType::Required,
            selectors: Vec::new(),
            weight: 100,
        }
    }

    /// Creates a new preferred affinity rule with a weight.
    pub fn preferred_affinity(name: impl Into<String>, weight: u32) -> Self {
        Self {
            name: name.into(),
            is_anti_affinity: false,
            affinity_type: AffinityType::Preferred,
            selectors: Vec::new(),
            weight: weight.clamp(1, 100),
        }
    }

    /// Creates a new required anti-affinity rule.
    pub fn required_anti_affinity(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            is_anti_affinity: true,
            affinity_type: AffinityType::Required,
            selectors: Vec::new(),
            weight: 100,
        }
    }

    /// Adds a label selector to this rule.
    pub fn add_selector(&mut self, selector: LabelSelector) {
        self.selectors.push(selector);
    }

    /// Evaluates this rule against a set of node labels.
    ///
    /// For affinity rules: returns `true` if all selectors match.
    /// For anti-affinity rules: returns `true` if any selector does NOT match
    /// (i.e., the node is acceptable because it doesn't match the exclusion).
    #[must_use]
    pub fn evaluate(&self, labels: &HashMap<String, String>) -> bool {
        let all_match = self.selectors.iter().all(|s| s.matches(labels));
        if self.is_anti_affinity {
            !all_match
        } else {
            all_match
        }
    }
}

/// A set of affinity rules applied to a job for node selection.
#[derive(Debug, Clone)]
pub struct AffinityRuleSet {
    /// All rules in this set.
    rules: Vec<AffinityRule>,
    /// Explicit node allowlist (if non-empty, only these nodes are considered).
    allowed_nodes: HashSet<String>,
    /// Explicit node blocklist.
    blocked_nodes: HashSet<String>,
}

impl AffinityRuleSet {
    /// Creates a new empty rule set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            allowed_nodes: HashSet::new(),
            blocked_nodes: HashSet::new(),
        }
    }

    /// Adds a rule.
    pub fn add_rule(&mut self, rule: AffinityRule) {
        self.rules.push(rule);
    }

    /// Adds a node to the explicit allowlist.
    pub fn allow_node(&mut self, node_id: impl Into<String>) {
        self.allowed_nodes.insert(node_id.into());
    }

    /// Adds a node to the explicit blocklist.
    pub fn block_node(&mut self, node_id: impl Into<String>) {
        self.blocked_nodes.insert(node_id.into());
    }

    /// Checks if a node is eligible based on all rules.
    ///
    /// Returns `true` if the node passes all required rules and is not blocked.
    #[must_use]
    pub fn is_eligible(&self, node_id: &str, labels: &HashMap<String, String>) -> bool {
        // Check blocklist
        if self.blocked_nodes.contains(node_id) {
            return false;
        }
        // Check allowlist
        if !self.allowed_nodes.is_empty() && !self.allowed_nodes.contains(node_id) {
            return false;
        }
        // Check required rules
        for rule in &self.rules {
            if rule.affinity_type == AffinityType::Required && !rule.evaluate(labels) {
                return false;
            }
        }
        true
    }

    /// Scores a node based on preferred rules. Higher = better match.
    #[must_use]
    pub fn score(&self, labels: &HashMap<String, String>) -> u32 {
        let mut total = 0u32;
        for rule in &self.rules {
            if rule.affinity_type == AffinityType::Preferred && rule.evaluate(labels) {
                total = total.saturating_add(rule.weight);
            }
        }
        total
    }

    /// Returns the number of rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl Default for AffinityRuleSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn test_node_label_display() {
        let label = NodeLabel::new("gpu", "a100");
        assert_eq!(format!("{label}"), "gpu=a100");
    }

    #[test]
    fn test_selector_equals_match() {
        let sel = LabelSelector::equals("zone", "us-east-1");
        let labels = make_labels(&[("zone", "us-east-1")]);
        assert!(sel.matches(&labels));
    }

    #[test]
    fn test_selector_equals_no_match() {
        let sel = LabelSelector::equals("zone", "us-east-1");
        let labels = make_labels(&[("zone", "eu-west-1")]);
        assert!(!sel.matches(&labels));
    }

    #[test]
    fn test_selector_not_equals() {
        let sel = LabelSelector::not_equals("os", "windows");
        let labels = make_labels(&[("os", "linux")]);
        assert!(sel.matches(&labels));
        let labels2 = make_labels(&[("os", "windows")]);
        assert!(!sel.matches(&labels2));
    }

    #[test]
    fn test_selector_in() {
        let sel = LabelSelector::is_in("gpu", vec!["a100".to_string(), "h100".to_string()]);
        let labels = make_labels(&[("gpu", "a100")]);
        assert!(sel.matches(&labels));
        let labels2 = make_labels(&[("gpu", "rtx3090")]);
        assert!(!sel.matches(&labels2));
    }

    #[test]
    fn test_selector_exists() {
        let sel = LabelSelector::exists("gpu");
        let labels = make_labels(&[("gpu", "any")]);
        assert!(sel.matches(&labels));
        let labels2 = make_labels(&[("cpu", "x86")]);
        assert!(!sel.matches(&labels2));
    }

    #[test]
    fn test_affinity_rule_required() {
        let mut rule = AffinityRule::required_affinity("need-gpu");
        rule.add_selector(LabelSelector::equals("gpu", "a100"));
        let labels = make_labels(&[("gpu", "a100")]);
        assert!(rule.evaluate(&labels));
        let labels2 = make_labels(&[("gpu", "rtx3090")]);
        assert!(!rule.evaluate(&labels2));
    }

    #[test]
    fn test_anti_affinity_rule() {
        let mut rule = AffinityRule::required_anti_affinity("avoid-spot");
        rule.add_selector(LabelSelector::equals("instance_type", "spot"));
        // Node IS spot -> anti-affinity says NO
        let labels = make_labels(&[("instance_type", "spot")]);
        assert!(!rule.evaluate(&labels));
        // Node is on-demand -> anti-affinity says YES
        let labels2 = make_labels(&[("instance_type", "on-demand")]);
        assert!(rule.evaluate(&labels2));
    }

    #[test]
    fn test_rule_set_eligibility() {
        let mut set = AffinityRuleSet::new();
        let mut rule = AffinityRule::required_affinity("gpu-rule");
        rule.add_selector(LabelSelector::equals("gpu", "a100"));
        set.add_rule(rule);

        let labels = make_labels(&[("gpu", "a100")]);
        assert!(set.is_eligible("node-1", &labels));

        let labels2 = make_labels(&[("gpu", "rtx3090")]);
        assert!(!set.is_eligible("node-2", &labels2));
    }

    #[test]
    fn test_rule_set_blocklist() {
        let mut set = AffinityRuleSet::new();
        set.block_node("bad-node");
        let labels = make_labels(&[]);
        assert!(!set.is_eligible("bad-node", &labels));
        assert!(set.is_eligible("good-node", &labels));
    }

    #[test]
    fn test_rule_set_allowlist() {
        let mut set = AffinityRuleSet::new();
        set.allow_node("vip-node");
        let labels = make_labels(&[]);
        assert!(set.is_eligible("vip-node", &labels));
        assert!(!set.is_eligible("other-node", &labels));
    }

    #[test]
    fn test_rule_set_scoring() {
        let mut set = AffinityRuleSet::new();
        let mut r1 = AffinityRule::preferred_affinity("fast-gpu", 50);
        r1.add_selector(LabelSelector::equals("gpu", "a100"));
        let mut r2 = AffinityRule::preferred_affinity("local-storage", 30);
        r2.add_selector(LabelSelector::exists("ssd"));
        set.add_rule(r1);
        set.add_rule(r2);

        let labels = make_labels(&[("gpu", "a100"), ("ssd", "nvme")]);
        assert_eq!(set.score(&labels), 80);

        let labels2 = make_labels(&[("gpu", "rtx3090"), ("ssd", "nvme")]);
        assert_eq!(set.score(&labels2), 30);
    }

    #[test]
    fn test_rule_set_empty() {
        let set = AffinityRuleSet::new();
        assert_eq!(set.rule_count(), 0);
        let labels = make_labels(&[]);
        assert!(set.is_eligible("any-node", &labels));
    }
}
