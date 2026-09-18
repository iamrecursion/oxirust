//! Geographic fencing for DRM content distribution.
//!
//! Restricts content playback based on geographic regions using ISO 3166-1
//! country codes. Supports allow-lists, deny-lists, and region groupings.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// ISO 3166-1 alpha-2 country code (e.g., "US", "JP", "DE").
pub type CountryCode = String;

/// A named group of countries (e.g., "EU", "APAC", "LATAM").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionGroup {
    /// Unique name of the region group.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Set of country codes in this group.
    pub countries: HashSet<CountryCode>,
}

impl RegionGroup {
    /// Create a new region group.
    #[must_use]
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            countries: HashSet::new(),
        }
    }

    /// Add a country code to this group.
    pub fn add_country(&mut self, code: &str) {
        self.countries.insert(code.to_uppercase());
    }

    /// Remove a country code from this group.
    pub fn remove_country(&mut self, code: &str) -> bool {
        self.countries.remove(&code.to_uppercase())
    }

    /// Check if a country is in this group.
    #[must_use]
    pub fn contains(&self, code: &str) -> bool {
        self.countries.contains(&code.to_uppercase())
    }

    /// Number of countries in this group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.countries.len()
    }

    /// Whether the group is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.countries.is_empty()
    }
}

/// Policy mode for geographic fencing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeoFenceMode {
    /// Only listed countries are allowed (allowlist).
    AllowList,
    /// Listed countries are blocked; all others allowed (denylist).
    DenyList,
    /// No geographic restrictions.
    Unrestricted,
}

/// A geographic fence rule applied to a piece of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoFenceRule {
    /// Unique identifier for the rule.
    pub rule_id: String,
    /// Content identifier this rule applies to.
    pub content_id: String,
    /// Policy mode (allow-list or deny-list).
    pub mode: GeoFenceMode,
    /// Set of country codes in the rule.
    pub countries: HashSet<CountryCode>,
    /// Optional start timestamp (epoch seconds) for temporal fencing.
    pub valid_from: Option<u64>,
    /// Optional end timestamp (epoch seconds) for temporal fencing.
    pub valid_until: Option<u64>,
    /// Priority (higher value wins when rules conflict).
    pub priority: u32,
}

impl GeoFenceRule {
    /// Create a new geo-fence rule.
    #[must_use]
    pub fn new(rule_id: &str, content_id: &str, mode: GeoFenceMode) -> Self {
        Self {
            rule_id: rule_id.to_string(),
            content_id: content_id.to_string(),
            mode,
            countries: HashSet::new(),
            valid_from: None,
            valid_until: None,
            priority: 0,
        }
    }

    /// Add a country code to this rule.
    pub fn add_country(&mut self, code: &str) {
        self.countries.insert(code.to_uppercase());
    }

    /// Add all countries from a region group to this rule.
    pub fn add_region_group(&mut self, group: &RegionGroup) {
        for c in &group.countries {
            self.countries.insert(c.clone());
        }
    }

    /// Set temporal validity window.
    pub fn set_validity(&mut self, from: Option<u64>, until: Option<u64>) {
        self.valid_from = from;
        self.valid_until = until;
    }

    /// Set the priority of this rule.
    pub fn set_priority(&mut self, priority: u32) {
        self.priority = priority;
    }

    /// Check if this rule is temporally valid at the given epoch timestamp.
    #[must_use]
    pub fn is_valid_at(&self, epoch_secs: u64) -> bool {
        if let Some(from) = self.valid_from {
            if epoch_secs < from {
                return false;
            }
        }
        if let Some(until) = self.valid_until {
            if epoch_secs > until {
                return false;
            }
        }
        true
    }

    /// Check if playback is allowed for the given country code.
    #[must_use]
    pub fn is_allowed(&self, country: &str) -> bool {
        let upper = country.to_uppercase();
        match self.mode {
            GeoFenceMode::AllowList => self.countries.contains(&upper),
            GeoFenceMode::DenyList => !self.countries.contains(&upper),
            GeoFenceMode::Unrestricted => true,
        }
    }
}

/// Result of a geo-fence evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeoFenceVerdict {
    /// Playback is allowed.
    Allowed,
    /// Playback is denied; contains the rule ID that blocked it.
    Denied {
        /// The rule ID that caused the denial.
        rule_id: String,
    },
    /// No applicable rule was found; default behaviour applies.
    NoRule,
}

/// Manager for geographic fence rules.
#[derive(Debug, Clone, Default)]
pub struct GeoFenceManager {
    /// Rules keyed by content ID.
    rules: HashMap<String, Vec<GeoFenceRule>>,
    /// Named region groups.
    groups: HashMap<String, RegionGroup>,
}

impl GeoFenceManager {
    /// Create a new manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a region group.
    pub fn register_group(&mut self, group: RegionGroup) {
        self.groups.insert(group.name.clone(), group);
    }

    /// Get a registered region group by name.
    #[must_use]
    pub fn get_group(&self, name: &str) -> Option<&RegionGroup> {
        self.groups.get(name)
    }

    /// Add a rule.
    pub fn add_rule(&mut self, rule: GeoFenceRule) {
        self.rules
            .entry(rule.content_id.clone())
            .or_default()
            .push(rule);
    }

    /// Remove all rules for a content ID.
    pub fn remove_rules(&mut self, content_id: &str) {
        self.rules.remove(content_id);
    }

    /// Evaluate geo-fence rules for the given content and country.
    ///
    /// Returns the verdict from the highest-priority applicable rule.
    #[must_use]
    pub fn evaluate(&self, content_id: &str, country: &str, epoch_secs: u64) -> GeoFenceVerdict {
        let Some(rules) = self.rules.get(content_id) else {
            return GeoFenceVerdict::NoRule;
        };

        let mut best: Option<&GeoFenceRule> = None;
        for rule in rules {
            if !rule.is_valid_at(epoch_secs) {
                continue;
            }
            if let Some(current) = best {
                if rule.priority > current.priority {
                    best = Some(rule);
                }
            } else {
                best = Some(rule);
            }
        }

        match best {
            None => GeoFenceVerdict::NoRule,
            Some(rule) => {
                if rule.is_allowed(country) {
                    GeoFenceVerdict::Allowed
                } else {
                    GeoFenceVerdict::Denied {
                        rule_id: rule.rule_id.clone(),
                    }
                }
            }
        }
    }

    /// Number of content IDs with rules.
    #[must_use]
    pub fn content_count(&self) -> usize {
        self.rules.len()
    }

    /// Total number of rules across all content.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.values().map(Vec::len).sum()
    }

    /// Create a standard EU region group.
    #[must_use]
    pub fn eu_region() -> RegionGroup {
        let mut g = RegionGroup::new("EU", "European Union member states");
        for code in &[
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
            "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        ] {
            g.add_country(code);
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_group_new() {
        let g = RegionGroup::new("TEST", "Test group");
        assert_eq!(g.name, "TEST");
        assert!(g.is_empty());
    }

    #[test]
    fn test_region_group_add_remove() {
        let mut g = RegionGroup::new("G", "g");
        g.add_country("us");
        g.add_country("jp");
        assert_eq!(g.len(), 2);
        assert!(g.contains("US"));
        assert!(g.contains("jp"));
        assert!(g.remove_country("US"));
        assert_eq!(g.len(), 1);
        assert!(!g.contains("us"));
    }

    #[test]
    fn test_rule_allow_list() {
        let mut rule = GeoFenceRule::new("r1", "c1", GeoFenceMode::AllowList);
        rule.add_country("US");
        rule.add_country("CA");
        assert!(rule.is_allowed("US"));
        assert!(rule.is_allowed("ca"));
        assert!(!rule.is_allowed("JP"));
    }

    #[test]
    fn test_rule_deny_list() {
        let mut rule = GeoFenceRule::new("r2", "c1", GeoFenceMode::DenyList);
        rule.add_country("CN");
        assert!(rule.is_allowed("US"));
        assert!(!rule.is_allowed("CN"));
    }

    #[test]
    fn test_rule_unrestricted() {
        let rule = GeoFenceRule::new("r3", "c1", GeoFenceMode::Unrestricted);
        assert!(rule.is_allowed("US"));
        assert!(rule.is_allowed("CN"));
    }

    #[test]
    fn test_temporal_validity() {
        let mut rule = GeoFenceRule::new("r4", "c1", GeoFenceMode::AllowList);
        rule.set_validity(Some(100), Some(200));
        assert!(!rule.is_valid_at(50));
        assert!(rule.is_valid_at(100));
        assert!(rule.is_valid_at(150));
        assert!(rule.is_valid_at(200));
        assert!(!rule.is_valid_at(201));
    }

    #[test]
    fn test_temporal_open_ended() {
        let mut rule = GeoFenceRule::new("r5", "c1", GeoFenceMode::AllowList);
        rule.set_validity(Some(100), None);
        assert!(!rule.is_valid_at(50));
        assert!(rule.is_valid_at(100));
        assert!(rule.is_valid_at(u64::MAX));
    }

    #[test]
    fn test_add_region_group_to_rule() {
        let mut g = RegionGroup::new("G", "g");
        g.add_country("FR");
        g.add_country("DE");
        let mut rule = GeoFenceRule::new("r6", "c1", GeoFenceMode::AllowList);
        rule.add_region_group(&g);
        assert!(rule.is_allowed("FR"));
        assert!(rule.is_allowed("DE"));
        assert!(!rule.is_allowed("US"));
    }

    #[test]
    fn test_manager_evaluate_no_rules() {
        let mgr = GeoFenceManager::new();
        let v = mgr.evaluate("c1", "US", 0);
        assert_eq!(v, GeoFenceVerdict::NoRule);
    }

    #[test]
    fn test_manager_evaluate_allow() {
        let mut mgr = GeoFenceManager::new();
        let mut rule = GeoFenceRule::new("r1", "c1", GeoFenceMode::AllowList);
        rule.add_country("US");
        mgr.add_rule(rule);
        assert_eq!(mgr.evaluate("c1", "US", 0), GeoFenceVerdict::Allowed);
        assert!(matches!(
            mgr.evaluate("c1", "JP", 0),
            GeoFenceVerdict::Denied { .. }
        ));
    }

    #[test]
    fn test_manager_priority() {
        let mut mgr = GeoFenceManager::new();
        let mut deny = GeoFenceRule::new("deny", "c1", GeoFenceMode::DenyList);
        deny.add_country("US");
        deny.set_priority(1);
        mgr.add_rule(deny);

        let mut allow = GeoFenceRule::new("allow", "c1", GeoFenceMode::AllowList);
        allow.add_country("US");
        allow.set_priority(10);
        mgr.add_rule(allow);

        // Higher priority allow rule wins.
        assert_eq!(mgr.evaluate("c1", "US", 0), GeoFenceVerdict::Allowed);
    }

    #[test]
    fn test_manager_remove_rules() {
        let mut mgr = GeoFenceManager::new();
        let rule = GeoFenceRule::new("r1", "c1", GeoFenceMode::Unrestricted);
        mgr.add_rule(rule);
        assert_eq!(mgr.content_count(), 1);
        mgr.remove_rules("c1");
        assert_eq!(mgr.content_count(), 0);
    }

    #[test]
    fn test_eu_region_group() {
        let eu = GeoFenceManager::eu_region();
        assert!(eu.contains("DE"));
        assert!(eu.contains("FR"));
        assert!(!eu.contains("US"));
        assert_eq!(eu.len(), 27);
    }

    #[test]
    fn test_manager_register_group() {
        let mut mgr = GeoFenceManager::new();
        let g = RegionGroup::new("APAC", "Asia-Pacific");
        mgr.register_group(g);
        assert!(mgr.get_group("APAC").is_some());
        assert!(mgr.get_group("NONE").is_none());
    }

    #[test]
    fn test_rule_count() {
        let mut mgr = GeoFenceManager::new();
        mgr.add_rule(GeoFenceRule::new("a", "c1", GeoFenceMode::Unrestricted));
        mgr.add_rule(GeoFenceRule::new("b", "c1", GeoFenceMode::Unrestricted));
        mgr.add_rule(GeoFenceRule::new("c", "c2", GeoFenceMode::Unrestricted));
        assert_eq!(mgr.rule_count(), 3);
        assert_eq!(mgr.content_count(), 2);
    }
}
