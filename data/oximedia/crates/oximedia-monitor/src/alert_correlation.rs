#![allow(dead_code)]
//! Alert correlation engine: groups related alerts that fire within a time window.
//!
//! When multiple alerts fire in rapid succession they often share a common root cause
//! (e.g. a failing host triggers CPU, memory, and disk alerts simultaneously).
//! [`AlertCorrelationEngine`] clusters such co-incident alerts into [`AlertGroup`]s,
//! surfacing the root-cause alert and the set of correlated followers so that on-call
//! engineers see one actionable incident rather than a flood of individual alerts.
//!
//! ## Grouping strategy
//!
//! Alerts are considered related when:
//! 1. They fire within `correlation_window_secs` of the group's `first_fired` timestamp, **and**
//! 2. They share at least one tag key-value pair with the root-cause alert (tag-based affinity),
//!    **or** there is no active group at all (first alert becomes root cause).
//!
//! Groups are scored by tag-overlap: a new alert is added to the group with which it shares
//! the most tag key-value pairs.  If two groups tie, the most recently created group wins.
//!
//! ## Expiry
//!
//! Call [`AlertCorrelationEngine::flush_expired`] periodically to remove groups whose
//! correlation window has elapsed (`first_fired + correlation_window_secs < now`).

use std::collections::HashMap;

use crate::alert::AlertSeverity;

// ── AlertGroup ────────────────────────────────────────────────────────────────

/// A correlated group of alerts sharing a common root cause.
#[derive(Debug, Clone)]
pub struct AlertGroup {
    /// The name of the first / primary alert that opened this group.
    pub root_cause: String,
    /// Names of alerts that were subsequently correlated into this group
    /// (excludes `root_cause`).
    pub related: Vec<String>,
    /// Unix timestamp (seconds) at which the root-cause alert fired.
    pub first_fired: u64,
    /// Severity of the root-cause alert.
    pub severity: AlertSeverity,
    /// Tags attached to the root-cause alert (key, value pairs).
    pub tags: Vec<(String, String)>,
}

impl AlertGroup {
    /// Create a new group seeded by a single root-cause alert.
    fn new(
        root_cause: impl Into<String>,
        first_fired: u64,
        severity: AlertSeverity,
        tags: &[(&str, &str)],
    ) -> Self {
        Self {
            root_cause: root_cause.into(),
            related: Vec::new(),
            first_fired,
            severity,
            tags: tags
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    /// Total size of this group (root cause + related).
    #[must_use]
    pub fn size(&self) -> usize {
        1 + self.related.len()
    }

    /// Count how many tag key-value pairs this group shares with the given tags.
    #[must_use]
    fn tag_overlap(&self, other_tags: &[(&str, &str)]) -> usize {
        let mut count = 0usize;
        for (ok, ov) in other_tags {
            for (sk, sv) in &self.tags {
                if sk == ok && sv == ov {
                    count += 1;
                }
            }
        }
        count
    }

    /// Whether the alert `name` is already tracked in this group.
    #[must_use]
    fn contains(&self, name: &str) -> bool {
        self.root_cause == name || self.related.iter().any(|r| r == name)
    }
}

// ── AlertCorrelationEngine ────────────────────────────────────────────────────

/// Groups related alerts that fire within a configurable time window.
///
/// # Example
///
/// ```
/// use oximedia_monitor::alert_correlation::AlertCorrelationEngine;
/// use oximedia_monitor::AlertSeverity;
///
/// let mut engine = AlertCorrelationEngine::new(60);
/// let tags = [("host", "web-01")];
///
/// // First alert creates a new group.
/// let group = engine.add_alert("high_cpu", 1000, AlertSeverity::Critical, &tags);
/// assert!(group.is_some());
///
/// // Second alert within the window and same tag is correlated.
/// let group2 = engine.add_alert("high_mem", 1010, AlertSeverity::Warning, &tags);
/// assert!(group2.is_some());
/// assert_eq!(engine.active_groups()[0].related.len(), 1);
/// ```
pub struct AlertCorrelationEngine {
    /// Width of the correlation window in seconds.
    pub correlation_window_secs: u64,
    /// Maximum total size (root + related) a group may reach.
    pub max_group_size: usize,
    /// Active groups, keyed by root-cause alert name.
    groups: HashMap<String, AlertGroup>,
}

impl AlertCorrelationEngine {
    /// Create a new engine with the given window and a default max group size of 20.
    #[must_use]
    pub fn new(window_secs: u64) -> Self {
        Self {
            correlation_window_secs: window_secs,
            max_group_size: 20,
            groups: HashMap::new(),
        }
    }

    /// Create a new engine with an explicit max group size.
    #[must_use]
    pub fn new_with_max_group_size(window_secs: u64, max_group_size: usize) -> Self {
        Self {
            correlation_window_secs: window_secs,
            max_group_size: max_group_size.max(1),
            groups: HashMap::new(),
        }
    }

    /// Submit an alert to the engine.
    ///
    /// Returns `Some(group_snapshot)` when:
    /// - A brand-new group is created (first alert in window), or
    /// - The alert is successfully correlated into an existing group.
    ///
    /// Returns `None` when the alert is a duplicate or all candidate groups are full.
    pub fn add_alert(
        &mut self,
        name: &str,
        fired_at: u64,
        severity: AlertSeverity,
        tags: &[(&str, &str)],
    ) -> Option<AlertGroup> {
        // ── Check if this alert is already tracked in any active group ────────
        for group in self.groups.values() {
            if group.contains(name) {
                return None; // duplicate — already tracked
            }
        }

        // ── Find the best candidate group to join ─────────────────────────────
        // Criteria:
        //   1. Group must still be within the correlation window.
        //   2. Group must not be full.
        //   3. Prefer the group with the highest tag overlap; break ties by
        //      most-recently opened group (largest first_fired).
        // Collect all candidate groups (within window, not full).
        let mut candidates: Vec<(&str, usize, u64)> = Vec::new(); // (root_cause, overlap, first_fired)
        for (key, group) in &self.groups {
            // Window check: fired_at must be within [first_fired, first_fired + window].
            let in_window = fired_at >= group.first_fired
                && fired_at.saturating_sub(group.first_fired) <= self.correlation_window_secs;
            if !in_window {
                continue;
            }
            // Capacity check.
            if group.size() >= self.max_group_size {
                continue;
            }
            let overlap = group.tag_overlap(tags);
            candidates.push((key.as_str(), overlap, group.first_fired));
        }

        // Pick the best candidate:
        // - Prefer groups with the highest tag overlap (overlap > 0 required when tags are present).
        // - Break ties by most-recently created group (largest first_fired).
        // - If the incoming alert has tags but no candidate has any overlap, start a new group.
        let best_key: Option<String> = if candidates.is_empty() {
            None
        } else {
            let has_tags = !tags.is_empty();
            let max_overlap = candidates.iter().map(|&(_, o, _)| o).max().unwrap_or(0);

            // If the alert carries tags, require at least one shared tag to join a group.
            if has_tags && max_overlap == 0 {
                None
            } else {
                // Among candidates with the best overlap, pick the most recently opened.
                candidates
                    .iter()
                    .filter(|&&(_, o, _)| o == max_overlap)
                    .max_by_key(|&&(_, _, fired)| fired)
                    .map(|&(k, _, _)| k.to_string())
            }
        };

        if let Some(key) = best_key {
            // Correlate into an existing group.
            if let Some(group) = self.groups.get_mut(&key) {
                group.related.push(name.to_string());
                return Some(group.clone());
            }
        }

        // ── No suitable group found — start a new one ────────────────────────
        let new_group = AlertGroup::new(name, fired_at, severity, tags);
        let snapshot = new_group.clone();
        self.groups.insert(name.to_string(), new_group);
        Some(snapshot)
    }

    /// Remove groups whose correlation window has expired (`first_fired + window < now`).
    pub fn flush_expired(&mut self, now: u64) {
        self.groups.retain(|_, group| {
            group
                .first_fired
                .saturating_add(self.correlation_window_secs)
                >= now
        });
    }

    /// Return snapshots of all currently active groups.
    #[must_use]
    pub fn active_groups(&self) -> Vec<&AlertGroup> {
        self.groups.values().collect()
    }

    /// Number of currently active groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn no_tags() -> Vec<(&'static str, &'static str)> {
        vec![]
    }

    fn host_tag(h: &'static str) -> Vec<(&'static str, &'static str)> {
        vec![("host", h)]
    }

    // ── basic creation ────────────────────────────────────────────────────────

    #[test]
    fn test_new_group_created_for_first_alert() {
        let mut engine = AlertCorrelationEngine::new(60);
        let result = engine.add_alert("cpu_high", 1000, AlertSeverity::Critical, &host_tag("h1"));
        assert!(result.is_some());
        let group = result.expect("should return group");
        assert_eq!(group.root_cause, "cpu_high");
        assert!(group.related.is_empty());
        assert_eq!(group.first_fired, 1000);
        assert_eq!(group.severity, AlertSeverity::Critical);
        assert_eq!(engine.group_count(), 1);
    }

    // ── correlation ───────────────────────────────────────────────────────────

    #[test]
    fn test_correlated_alert_joins_existing_group() {
        let mut engine = AlertCorrelationEngine::new(60);
        let tags = host_tag("h1");
        engine
            .add_alert("cpu_high", 1000, AlertSeverity::Critical, &tags)
            .expect("first alert should create group");

        let result = engine.add_alert("mem_high", 1010, AlertSeverity::Warning, &tags);
        assert!(result.is_some());
        let group = result.expect("should return group");
        assert_eq!(group.root_cause, "cpu_high");
        assert_eq!(group.related, vec!["mem_high".to_string()]);
        assert_eq!(engine.group_count(), 1); // still one group
    }

    // ── max group size ────────────────────────────────────────────────────────

    #[test]
    fn test_max_group_size_enforced() {
        let mut engine = AlertCorrelationEngine::new_with_max_group_size(60, 2);
        let tags = host_tag("h1");
        // First alert → creates group (size 1).
        engine
            .add_alert("a1", 1000, AlertSeverity::Warning, &tags)
            .expect("should create group");
        // Second alert → joins group (size 2 = max).
        engine
            .add_alert("a2", 1010, AlertSeverity::Warning, &tags)
            .expect("should join group");
        // Third alert → group is full; no group with capacity → creates new group.
        let result = engine.add_alert("a3", 1020, AlertSeverity::Warning, &tags);
        assert!(result.is_some());
        // a3 should be root of a new group since the only candidate was full.
        assert_eq!(engine.group_count(), 2);
    }

    // ── flush ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_flush_expired_removes_old_groups() {
        let mut engine = AlertCorrelationEngine::new(30);
        engine.add_alert("old_alert", 1000, AlertSeverity::Info, &no_tags());
        engine.add_alert("new_alert", 2000, AlertSeverity::Info, &no_tags());
        assert_eq!(engine.group_count(), 2);

        // flush at t=1032 — "old_alert" (first_fired=1000, window=30) expires (1000+30=1030 < 1032)
        // "new_alert" (first_fired=2000) does NOT expire
        engine.flush_expired(1032);
        assert_eq!(engine.group_count(), 1);
        let keys: Vec<&str> = engine
            .active_groups()
            .iter()
            .map(|g| g.root_cause.as_str())
            .collect();
        assert!(keys.contains(&"new_alert"));
    }

    // ── active_groups ─────────────────────────────────────────────────────────

    #[test]
    fn test_active_groups_returns_all() {
        let mut engine = AlertCorrelationEngine::new(60);
        engine.add_alert("a1", 1000, AlertSeverity::Info, &no_tags());
        engine.add_alert("a2", 5000, AlertSeverity::Info, &no_tags());
        engine.add_alert("a3", 9000, AlertSeverity::Info, &no_tags());
        let groups = engine.active_groups();
        assert_eq!(groups.len(), 3);
    }

    // ── tag-based grouping ────────────────────────────────────────────────────

    #[test]
    fn test_tag_based_grouping_preference() {
        let mut engine = AlertCorrelationEngine::new(120);
        // Two existing groups, each with a different host tag.
        engine.add_alert("cpu_h1", 1000, AlertSeverity::Warning, &host_tag("h1"));
        engine.add_alert("cpu_h2", 1005, AlertSeverity::Warning, &host_tag("h2"));

        // New alert tagged for h2 should join the h2 group.
        let result = engine.add_alert("mem_h2", 1010, AlertSeverity::Warning, &host_tag("h2"));
        assert!(result.is_some());
        let group = result.expect("should join h2 group");
        assert_eq!(group.root_cause, "cpu_h2");
        assert!(group.related.contains(&"mem_h2".to_string()));

        // Total groups should still be 2.
        assert_eq!(engine.group_count(), 2);
    }

    // ── group_count ───────────────────────────────────────────────────────────

    #[test]
    fn test_group_count() {
        let mut engine = AlertCorrelationEngine::new(10);
        assert_eq!(engine.group_count(), 0);
        engine.add_alert("x", 0, AlertSeverity::Info, &no_tags());
        assert_eq!(engine.group_count(), 1);
        engine.add_alert("y", 100, AlertSeverity::Info, &no_tags()); // outside window
        assert_eq!(engine.group_count(), 2);
    }

    // ── alerts outside window create new group ────────────────────────────────

    #[test]
    fn test_alerts_outside_window_create_new_group() {
        let mut engine = AlertCorrelationEngine::new(60);
        engine.add_alert("first", 1000, AlertSeverity::Warning, &host_tag("h1"));
        // fired_at = 1000 + 61 = 1061, which is outside the window of the first group.
        let result = engine.add_alert("second", 1061, AlertSeverity::Warning, &host_tag("h1"));
        assert!(result.is_some());
        let group = result.expect("should create new group");
        assert_eq!(group.root_cause, "second");
        assert_eq!(engine.group_count(), 2);
    }

    // ── duplicate alert suppression ───────────────────────────────────────────

    #[test]
    fn test_duplicate_alert_returns_none() {
        let mut engine = AlertCorrelationEngine::new(60);
        let tags = host_tag("h1");
        engine
            .add_alert("cpu_high", 1000, AlertSeverity::Critical, &tags)
            .expect("should create group");
        // Submit the exact same alert name again.
        let result = engine.add_alert("cpu_high", 1005, AlertSeverity::Critical, &tags);
        assert!(result.is_none());
        assert_eq!(engine.group_count(), 1);
    }
}
