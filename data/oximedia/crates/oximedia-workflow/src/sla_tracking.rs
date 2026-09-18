//! SLA tracking for workflow deadlines, breach detection, and escalation rules.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// SLA tier / severity level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlaLevel {
    /// Bronze tier — standard turnaround.
    Bronze,
    /// Silver tier — expedited turnaround.
    Silver,
    /// Gold tier — same-day turnaround.
    Gold,
    /// Platinum tier — mission-critical.
    Platinum,
}

/// Status of an SLA entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlaStatus {
    /// SLA is active and within deadline.
    Active,
    /// SLA is at risk (within warning threshold).
    AtRisk,
    /// SLA has been breached.
    Breached,
    /// SLA has been met successfully.
    Met,
}

/// An escalation rule triggered on SLA breach.
#[derive(Debug, Clone)]
pub struct EscalationRule {
    /// Rule identifier.
    pub id: String,
    /// Minimum level that triggers this rule.
    pub trigger_level: SlaLevel,
    /// Contacts to notify (email addresses, usernames, etc.).
    pub notify_contacts: Vec<String>,
    /// Delay before escalating after breach is detected.
    pub escalation_delay: Duration,
    /// Whether this rule has already fired.
    pub fired: bool,
}

impl EscalationRule {
    /// Create a new escalation rule.
    #[must_use]
    pub fn new(id: &str, trigger_level: SlaLevel, contacts: Vec<String>, delay: Duration) -> Self {
        Self {
            id: id.to_string(),
            trigger_level,
            notify_contacts: contacts,
            escalation_delay: delay,
            fired: false,
        }
    }

    /// Mark this rule as fired.
    pub fn fire(&mut self) {
        self.fired = true;
    }
}

/// Internal SLA entry stored in the tracker.
#[derive(Debug, Clone)]
struct SlaEntryInternal {
    id: String,
    name: String,
    deadline: Instant,
    warning_threshold: Duration,
    status: SlaStatus,
    level: SlaLevel,
    escalation_rules: Vec<EscalationRule>,
    created_at: Instant,
    breach_detected_at: Option<Instant>,
}

impl SlaEntryInternal {
    fn new(id: &str, name: &str, ttd: Duration, level: SlaLevel) -> Self {
        let warning = if ttd.as_secs() == 0 {
            Duration::ZERO
        } else {
            ttd / 5
        };
        Self {
            id: id.to_string(),
            name: name.to_string(),
            deadline: Instant::now() + ttd,
            warning_threshold: warning,
            status: SlaStatus::Active,
            level,
            escalation_rules: Vec::new(),
            created_at: Instant::now(),
            breach_detected_at: None,
        }
    }

    fn remaining(&self) -> Duration {
        let now = Instant::now();
        if now >= self.deadline {
            Duration::ZERO
        } else {
            self.deadline - now
        }
    }

    fn refresh_status(&mut self) {
        if self.status == SlaStatus::Met {
            return;
        }
        let now = Instant::now();
        if now >= self.deadline {
            if self.status != SlaStatus::Breached {
                self.breach_detected_at = Some(now);
            }
            self.status = SlaStatus::Breached;
        } else if self.remaining() <= self.warning_threshold {
            self.status = SlaStatus::AtRisk;
        }
    }
}

/// SLA tracker managing multiple SLA entries.
#[derive(Debug, Default)]
pub struct SlaTracker {
    entries: HashMap<String, SlaEntryInternal>,
}

impl SlaTracker {
    /// Create a new empty SLA tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new SLA entry with a time-to-deadline and level.
    pub fn register(&mut self, id: &str, name: &str, ttd: Duration, level: SlaLevel) {
        let entry = SlaEntryInternal::new(id, name, ttd, level);
        self.entries.insert(id.to_string(), entry);
    }

    /// Add an escalation rule to an existing entry.
    pub fn add_escalation_rule(&mut self, entry_id: &str, rule: EscalationRule) -> bool {
        if let Some(e) = self.entries.get_mut(entry_id) {
            e.escalation_rules.push(rule);
            true
        } else {
            false
        }
    }

    /// Mark an SLA as successfully met.
    pub fn mark_met(&mut self, id: &str) -> bool {
        if let Some(e) = self.entries.get_mut(id) {
            e.status = SlaStatus::Met;
            true
        } else {
            false
        }
    }

    /// Tick: refresh all statuses and return list of newly breached IDs.
    pub fn tick(&mut self) -> Vec<String> {
        let mut breached = Vec::new();
        for entry in self.entries.values_mut() {
            let was_breached = entry.status == SlaStatus::Breached;
            entry.refresh_status();
            if !was_breached && entry.status == SlaStatus::Breached {
                breached.push(entry.id.clone());
            }
        }
        breached
    }

    /// Return all at-risk entry IDs.
    #[must_use]
    pub fn at_risk_ids(&self) -> Vec<String> {
        self.entries
            .values()
            .filter(|e| e.status == SlaStatus::AtRisk)
            .map(|e| e.id.clone())
            .collect()
    }

    /// Return all breached entry IDs.
    #[must_use]
    pub fn breached_ids(&self) -> Vec<String> {
        self.entries
            .values()
            .filter(|e| e.status == SlaStatus::Breached)
            .map(|e| e.id.clone())
            .collect()
    }

    /// Return status of a specific entry.
    #[must_use]
    pub fn status(&self, id: &str) -> Option<SlaStatus> {
        self.entries.get(id).map(|e| e.status.clone())
    }

    /// Return level of a specific entry.
    #[must_use]
    pub fn level(&self, id: &str) -> Option<&SlaLevel> {
        self.entries.get(id).map(|e| &e.level)
    }

    /// Return remaining duration for a specific entry.
    #[must_use]
    pub fn remaining(&self, id: &str) -> Option<Duration> {
        self.entries.get(id).map(SlaEntryInternal::remaining)
    }

    /// Count total entries.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Count entries by status.
    #[must_use]
    pub fn count_by_status(&self, status: &SlaStatus) -> usize {
        self.entries
            .values()
            .filter(|e| &e.status == status)
            .count()
    }

    /// Breach rate (breached / total), returns 0.0 if no entries.
    #[must_use]
    pub fn breach_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let breached = self.count_by_status(&SlaStatus::Breached) as f64;
        let total = self.entries.len() as f64;
        breached / total
    }

    /// Compliance rate (met / total), returns 1.0 if no entries.
    #[must_use]
    pub fn compliance_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 1.0;
        }
        let met = self.count_by_status(&SlaStatus::Met) as f64;
        let total = self.entries.len() as f64;
        met / total
    }

    /// Fire pending escalation rules for breached entries.
    pub fn fire_escalations(&mut self) -> Vec<String> {
        let mut fired_ids = Vec::new();
        for entry in self.entries.values_mut() {
            if entry.status != SlaStatus::Breached {
                continue;
            }
            if let Some(breach_time) = entry.breach_detected_at {
                let elapsed = Instant::now().duration_since(breach_time);
                for rule in &mut entry.escalation_rules {
                    if !rule.fired && elapsed >= rule.escalation_delay {
                        rule.fired = true;
                        fired_ids.push(rule.id.clone());
                    }
                }
            }
        }
        fired_ids
    }
}

/// Breach detection report.
#[derive(Debug, Clone)]
pub struct BreachReport {
    /// ID of the breached entry.
    pub entry_id: String,
    /// Name of the breached entry.
    pub entry_name: String,
    /// Level of the SLA.
    pub level: SlaLevel,
    /// How long ago the breach occurred (approximate).
    pub breach_age: Duration,
}

/// Build a breach report for all currently breached entries.
#[must_use]
pub fn build_breach_report(tracker: &SlaTracker) -> Vec<BreachReport> {
    tracker
        .entries
        .values()
        .filter(|e| e.status == SlaStatus::Breached)
        .map(|e| BreachReport {
            entry_id: e.id.clone(),
            entry_name: e.name.clone(),
            level: e.level.clone(),
            breach_age: e
                .breach_detected_at
                .map_or(Duration::ZERO, |t| Instant::now().duration_since(t)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_register_and_count() {
        let mut tracker = SlaTracker::new();
        tracker.register(
            "w1",
            "Workflow 1",
            Duration::from_secs(60),
            SlaLevel::Bronze,
        );
        tracker.register("w2", "Workflow 2", Duration::from_secs(120), SlaLevel::Gold);
        assert_eq!(tracker.count(), 2);
    }

    #[test]
    fn test_initial_status_active() {
        let mut tracker = SlaTracker::new();
        tracker.register(
            "w1",
            "Workflow 1",
            Duration::from_secs(60),
            SlaLevel::Silver,
        );
        assert_eq!(tracker.status("w1"), Some(SlaStatus::Active));
    }

    #[test]
    fn test_mark_met() {
        let mut tracker = SlaTracker::new();
        tracker.register("w1", "Workflow 1", Duration::from_secs(60), SlaLevel::Gold);
        assert!(tracker.mark_met("w1"));
        assert_eq!(tracker.status("w1"), Some(SlaStatus::Met));
    }

    #[test]
    fn test_mark_met_nonexistent() {
        let mut tracker = SlaTracker::new();
        assert!(!tracker.mark_met("nonexistent"));
    }

    #[test]
    fn test_status_unknown_id() {
        let tracker = SlaTracker::new();
        assert_eq!(tracker.status("missing"), None);
    }

    #[test]
    fn test_breach_on_zero_duration() {
        let mut tracker = SlaTracker::new();
        tracker.register("w1", "Workflow 1", Duration::ZERO, SlaLevel::Platinum);
        tracker.tick();
        assert_eq!(tracker.status("w1"), Some(SlaStatus::Breached));
    }

    #[test]
    fn test_compliance_rate_all_met() {
        let mut tracker = SlaTracker::new();
        tracker.register("w1", "W1", Duration::from_secs(60), SlaLevel::Bronze);
        tracker.register("w2", "W2", Duration::from_secs(60), SlaLevel::Bronze);
        tracker.mark_met("w1");
        tracker.mark_met("w2");
        assert!((tracker.compliance_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_breach_rate_none_breached() {
        let mut tracker = SlaTracker::new();
        tracker.register("w1", "W1", Duration::from_secs(3600), SlaLevel::Silver);
        assert!((tracker.breach_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_breach_rate_empty() {
        let tracker = SlaTracker::new();
        assert_eq!(tracker.breach_rate(), 0.0);
    }

    #[test]
    fn test_compliance_rate_empty() {
        let tracker = SlaTracker::new();
        assert_eq!(tracker.compliance_rate(), 1.0);
    }

    #[test]
    fn test_at_risk_ids_empty() {
        let tracker = SlaTracker::new();
        assert!(tracker.at_risk_ids().is_empty());
    }

    #[test]
    fn test_breached_ids_empty() {
        let tracker = SlaTracker::new();
        assert!(tracker.breached_ids().is_empty());
    }

    #[test]
    fn test_count_by_status() {
        let mut tracker = SlaTracker::new();
        tracker.register("w1", "W1", Duration::from_secs(60), SlaLevel::Bronze);
        tracker.register("w2", "W2", Duration::from_secs(60), SlaLevel::Bronze);
        tracker.mark_met("w1");
        assert_eq!(tracker.count_by_status(&SlaStatus::Met), 1);
        assert_eq!(tracker.count_by_status(&SlaStatus::Active), 1);
    }

    #[test]
    fn test_escalation_rule_new() {
        let rule = EscalationRule::new(
            "rule1",
            SlaLevel::Gold,
            vec!["ops@example.com".to_string()],
            Duration::from_secs(300),
        );
        assert_eq!(rule.id, "rule1");
        assert!(!rule.fired);
        assert_eq!(rule.notify_contacts.len(), 1);
    }

    #[test]
    fn test_escalation_rule_fire() {
        let mut rule = EscalationRule::new(
            "rule2",
            SlaLevel::Platinum,
            vec!["cto@example.com".to_string()],
            Duration::ZERO,
        );
        rule.fire();
        assert!(rule.fired);
    }

    #[test]
    fn test_add_escalation_rule() {
        let mut tracker = SlaTracker::new();
        tracker.register("w1", "W1", Duration::from_secs(60), SlaLevel::Gold);
        let rule = EscalationRule::new("r1", SlaLevel::Gold, vec![], Duration::from_secs(30));
        assert!(tracker.add_escalation_rule("w1", rule));
    }

    #[test]
    fn test_add_escalation_rule_missing_entry() {
        let mut tracker = SlaTracker::new();
        let rule = EscalationRule::new("r1", SlaLevel::Gold, vec![], Duration::from_secs(30));
        assert!(!tracker.add_escalation_rule("nonexistent", rule));
    }

    #[test]
    fn test_breach_report_fields() {
        let report = BreachReport {
            entry_id: "w1".to_string(),
            entry_name: "Workflow 1".to_string(),
            level: SlaLevel::Gold,
            breach_age: Duration::from_secs(10),
        };
        assert_eq!(report.entry_id, "w1");
        assert_eq!(report.level, SlaLevel::Gold);
    }

    #[test]
    fn test_build_breach_report_empty() {
        let tracker = SlaTracker::new();
        let report = build_breach_report(&tracker);
        assert!(report.is_empty());
    }

    #[test]
    fn test_level_accessor() {
        let mut tracker = SlaTracker::new();
        tracker.register("w1", "W1", Duration::from_secs(60), SlaLevel::Platinum);
        assert_eq!(tracker.level("w1"), Some(&SlaLevel::Platinum));
    }

    #[test]
    fn test_remaining_positive() {
        let mut tracker = SlaTracker::new();
        tracker.register("w1", "W1", Duration::from_secs(3600), SlaLevel::Bronze);
        let rem = tracker.remaining("w1").expect("should succeed in test");
        assert!(rem > Duration::from_secs(3590));
    }
}
