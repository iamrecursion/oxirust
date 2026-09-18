//! Alerting rules engine for `OxiMedia` monitoring.
//!
//! Provides a comprehensive rules engine supporting threshold, rate-of-change,
//! absence, and composite alert conditions with cooldown management.
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::SystemTime;

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// Severity level of a fired alert.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum AlertSeverity {
    /// Informational notice, no action required.
    Info,
    /// Warning: performance or capacity degradation observed.
    Warning,
    /// Critical: service impact likely.
    Critical,
    /// Emergency: immediate intervention required.
    Emergency,
}

/// Comparison operator for threshold conditions.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CompareOp {
    /// Greater than.
    Gt,
    /// Less than.
    Lt,
    /// Greater than or equal.
    Gte,
    /// Less than or equal.
    Lte,
    /// Equal (within float epsilon).
    Eq,
    /// Not equal (outside float epsilon).
    Ne,
}

impl CompareOp {
    /// Apply the operator to `lhs` and `rhs`, returning `true` when the condition holds.
    #[must_use]
    pub fn apply(&self, lhs: f64, rhs: f64) -> bool {
        const EPS: f64 = 1e-12;
        match self {
            Self::Gt => lhs > rhs,
            Self::Lt => lhs < rhs,
            Self::Gte => lhs >= rhs,
            Self::Lte => lhs <= rhs,
            Self::Eq => (lhs - rhs).abs() < EPS,
            Self::Ne => (lhs - rhs).abs() >= EPS,
        }
    }
}

/// Logic operator for composite conditions.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LogicOp {
    /// All sub-conditions must be true.
    And,
    /// At least one sub-condition must be true.
    Or,
}

/// An alert condition that determines when an [`AlertRule`] fires.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AlertCondition {
    /// Fire when the named metric satisfies `op` against `value`.
    Threshold {
        /// Metric name to observe.
        metric: String,
        /// Comparison operator.
        op: CompareOp,
        /// Threshold value.
        value: f64,
    },
    /// Fire when the rate of change of the metric over `window_secs` meets or
    /// exceeds `min_change` (absolute value).
    RateOfChange {
        /// Metric name to observe.
        metric: String,
        /// Observation window in seconds.
        window_secs: u64,
        /// Minimum absolute rate of change per second to trigger.
        min_change: f64,
    },
    /// Fire when no snapshot has been received for `metric` within
    /// `max_silence_secs`.
    Absence {
        /// Metric name to observe.
        metric: String,
        /// Maximum allowed silence duration in seconds.
        max_silence_secs: u64,
    },
    /// Combine multiple conditions with AND / OR logic.
    Composite {
        /// Sub-conditions.
        conditions: Vec<AlertCondition>,
        /// How to combine the results.
        logic: LogicOp,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// AlertRule
// ─────────────────────────────────────────────────────────────────────────────

/// A named alert rule that maps a condition to a severity and notification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlertRule {
    /// Unique identifier for this rule.
    pub id: String,
    /// Human-readable rule name.
    pub name: String,
    /// The condition to evaluate on each [`AlertRuleEngine::evaluate`] call.
    pub condition: AlertCondition,
    /// Severity of generated alerts.
    pub severity: AlertSeverity,
    /// Template for alert messages; supports `{value}` and `{threshold}` placeholders.
    pub message_template: String,
    /// Minimum seconds between repeated alerts from this rule.
    pub cooldown_secs: u64,
    /// Whether this rule participates in evaluation.
    pub enabled: bool,
}

impl AlertRule {
    /// Create a new rule with sensible defaults (enabled, 60 s cooldown, empty template).
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        condition: AlertCondition,
        severity: AlertSeverity,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            condition,
            severity,
            message_template: "{metric} value {value} triggered alert".to_string(),
            cooldown_secs: 60,
            enabled: true,
        }
    }

    /// Alert when CPU usage exceeds `threshold_pct` percent.
    #[must_use]
    pub fn high_cpu(threshold_pct: f64) -> Self {
        let mut rule = Self::new(
            "builtin.high_cpu",
            "High CPU Usage",
            AlertCondition::Threshold {
                metric: "cpu.usage_pct".to_string(),
                op: CompareOp::Gt,
                value: threshold_pct,
            },
            AlertSeverity::Warning,
        );
        rule.message_template =
            format!("CPU usage at {{value}}% exceeds threshold {threshold_pct}%");
        rule
    }

    /// Alert when memory usage exceeds `threshold_pct` percent.
    #[must_use]
    pub fn high_memory(threshold_pct: f64) -> Self {
        let mut rule = Self::new(
            "builtin.high_memory",
            "High Memory Usage",
            AlertCondition::Threshold {
                metric: "memory.usage_pct".to_string(),
                op: CompareOp::Gt,
                value: threshold_pct,
            },
            AlertSeverity::Warning,
        );
        rule.message_template =
            format!("Memory usage at {{value}}% exceeds threshold {threshold_pct}%");
        rule
    }

    /// Alert when disk usage exceeds `threshold_pct` percent.
    #[must_use]
    pub fn disk_full(threshold_pct: f64) -> Self {
        let mut rule = Self::new(
            "builtin.disk_full",
            "Disk Full",
            AlertCondition::Threshold {
                metric: "disk.usage_pct".to_string(),
                op: CompareOp::Gt,
                value: threshold_pct,
            },
            AlertSeverity::Critical,
        );
        rule.message_template =
            format!("Disk usage at {{value}}% exceeds threshold {threshold_pct}%");
        rule
    }

    /// Alert when the job queue depth exceeds `max_jobs`.
    #[must_use]
    pub fn queue_depth(max_jobs: f64) -> Self {
        let mut rule = Self::new(
            "builtin.queue_depth",
            "Queue Depth Exceeded",
            AlertCondition::Threshold {
                metric: "queue.depth".to_string(),
                op: CompareOp::Gt,
                value: max_jobs,
            },
            AlertSeverity::Warning,
        );
        rule.message_template = format!("Queue depth {{value}} exceeds maximum {max_jobs}");
        rule
    }

    /// Alert when the transcode failure rate exceeds `max_rate` (0.0–1.0).
    #[must_use]
    pub fn transcode_failure_rate(max_rate: f64) -> Self {
        let mut rule = Self::new(
            "builtin.transcode_failure_rate",
            "Transcode Failure Rate",
            AlertCondition::Threshold {
                metric: "transcode.failure_rate".to_string(),
                op: CompareOp::Gt,
                value: max_rate,
            },
            AlertSeverity::Critical,
        );
        rule.message_template =
            format!("Transcode failure rate {{value}} exceeds maximum {max_rate}");
        rule
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MetricSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A single timestamped metric observation.
#[derive(Debug, Clone)]
pub struct MetricSnapshot {
    /// Name of the metric.
    pub metric: String,
    /// Observed value.
    pub value: f64,
    /// Wall-clock time of the observation.
    pub timestamp: SystemTime,
}

impl MetricSnapshot {
    /// Create a new snapshot stamped to now.
    pub fn now(metric: impl Into<String>, value: f64) -> Self {
        Self {
            metric: metric.into(),
            value,
            timestamp: SystemTime::now(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Alert (fired event)
// ─────────────────────────────────────────────────────────────────────────────

/// A fired alert produced when a rule's condition is satisfied.
#[derive(Debug, Clone)]
pub struct Alert {
    /// ID of the rule that produced this alert.
    pub rule_id: String,
    /// Name of the rule.
    pub rule_name: String,
    /// Severity of this alert.
    pub severity: AlertSeverity,
    /// Rendered alert message.
    pub message: String,
    /// The metric value that triggered the condition (or `f64::NAN` for absence).
    pub metric_value: f64,
    /// When the alert was fired.
    pub fired_at: SystemTime,
}

// ─────────────────────────────────────────────────────────────────────────────
// AlertRuleEngine
// ─────────────────────────────────────────────────────────────────────────────

/// State tracked per rule between evaluation cycles.
#[derive(Debug)]
struct RuleState {
    /// The most recent [`Alert`] that was fired for this rule.
    last_alert: Option<Alert>,
    /// Wall-clock time of the last fire.
    last_fired: Option<SystemTime>,
}

impl RuleState {
    fn new() -> Self {
        Self {
            last_alert: None,
            last_fired: None,
        }
    }
}

/// Stateful alerting rules engine.
///
/// Maintains rules and their cooldown state across successive [`evaluate`] calls.
///
/// [`evaluate`]: AlertRuleEngine::evaluate
pub struct AlertRuleEngine {
    rules: Vec<AlertRule>,
    state: HashMap<String, RuleState>,
}

impl AlertRuleEngine {
    /// Create a new, empty engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            state: HashMap::new(),
        }
    }

    /// Add a rule to the engine. Replaces any existing rule with the same id.
    pub fn add_rule(&mut self, rule: AlertRule) {
        let id = rule.id.clone();
        // Replace existing rule if present.
        if let Some(pos) = self.rules.iter().position(|r| r.id == id) {
            self.rules[pos] = rule;
        } else {
            self.rules.push(rule);
            self.state.entry(id).or_insert_with(RuleState::new);
        }
    }

    /// Remove the rule with the given id. Returns `true` if a rule was removed.
    pub fn remove_rule(&mut self, id: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != id);
        let removed = self.rules.len() < before;
        if removed {
            self.state.remove(id);
        }
        removed
    }

    /// Evaluate all enabled rules against the provided metric snapshots.
    ///
    /// Rules in cooldown are skipped. Active alerts are tracked internally and
    /// can be retrieved via [`active_alerts`].
    ///
    /// [`active_alerts`]: AlertRuleEngine::active_alerts
    pub fn evaluate(&mut self, snapshots: &[MetricSnapshot]) -> Vec<Alert> {
        let now = SystemTime::now();
        let mut fired: Vec<Alert> = Vec::new();

        // Collect rule ids so we can iterate without borrowing self mutably.
        let rule_ids: Vec<String> = self.rules.iter().map(|r| r.id.clone()).collect();

        for rule_id in &rule_ids {
            let rule = match self.rules.iter().find(|r| &r.id == rule_id) {
                Some(r) => r.clone(),
                None => continue,
            };

            if !rule.enabled {
                continue;
            }

            // Check cooldown.
            let state = self
                .state
                .entry(rule.id.clone())
                .or_insert_with(RuleState::new);
            if let Some(last_fired) = state.last_fired {
                let elapsed = now
                    .duration_since(last_fired)
                    .unwrap_or(std::time::Duration::ZERO)
                    .as_secs();
                if elapsed < rule.cooldown_secs {
                    continue;
                }
            }

            // Evaluate condition.
            if let Some(alert) = Self::evaluate_condition(&rule, &rule.condition, snapshots, now) {
                // Record state.
                let state = self
                    .state
                    .entry(rule.id.clone())
                    .or_insert_with(RuleState::new);
                state.last_fired = Some(now);
                state.last_alert = Some(alert.clone());
                fired.push(alert);
            } else {
                // Clear stored alert if condition no longer holds.
                let state = self
                    .state
                    .entry(rule.id.clone())
                    .or_insert_with(RuleState::new);
                state.last_alert = None;
            }
        }

        fired
    }

    /// Return references to all currently active (most-recently-fired) alerts.
    #[must_use]
    pub fn active_alerts(&self) -> Vec<&Alert> {
        self.state
            .values()
            .filter_map(|s| s.last_alert.as_ref())
            .collect()
    }

    /// Clear the stored alert for a specific rule (marks it as resolved).
    pub fn clear_alert(&mut self, rule_id: &str) {
        if let Some(state) = self.state.get_mut(rule_id) {
            state.last_alert = None;
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Evaluate a condition (possibly recursive for composites).
    /// Returns `Some(Alert)` when the condition fires, `None` otherwise.
    fn evaluate_condition(
        rule: &AlertRule,
        condition: &AlertCondition,
        snapshots: &[MetricSnapshot],
        now: SystemTime,
    ) -> Option<Alert> {
        match condition {
            AlertCondition::Threshold { metric, op, value } => {
                // Find the latest snapshot for this metric.
                let latest = Self::latest_for(snapshots, metric)?;
                if op.apply(latest.value, *value) {
                    let msg =
                        render_template(&rule.message_template, latest.value, Some(*value), metric);
                    Some(Alert {
                        rule_id: rule.id.clone(),
                        rule_name: rule.name.clone(),
                        severity: rule.severity.clone(),
                        message: msg,
                        metric_value: latest.value,
                        fired_at: now,
                    })
                } else {
                    None
                }
            }

            AlertCondition::RateOfChange {
                metric,
                window_secs,
                min_change,
            } => {
                let window_dur = std::time::Duration::from_secs(*window_secs);
                let cutoff = now.checked_sub(window_dur)?;

                // Collect snapshots within the window, sorted oldest-first.
                let mut in_window: Vec<&MetricSnapshot> = snapshots
                    .iter()
                    .filter(|s| s.metric == *metric && s.timestamp >= cutoff)
                    .collect();

                if in_window.len() < 2 {
                    return None;
                }

                in_window.sort_by_key(|s| s.timestamp);

                let earliest = in_window.first()?;
                let latest = in_window.last()?;

                let elapsed = latest
                    .timestamp
                    .duration_since(earliest.timestamp)
                    .unwrap_or(std::time::Duration::from_secs(1))
                    .as_secs_f64()
                    .max(1e-9);

                let rate = (latest.value - earliest.value).abs() / elapsed;

                if rate >= *min_change {
                    let msg =
                        render_template(&rule.message_template, rate, Some(*min_change), metric);
                    Some(Alert {
                        rule_id: rule.id.clone(),
                        rule_name: rule.name.clone(),
                        severity: rule.severity.clone(),
                        message: msg,
                        metric_value: rate,
                        fired_at: now,
                    })
                } else {
                    None
                }
            }

            AlertCondition::Absence {
                metric,
                max_silence_secs,
            } => {
                let max_silence = std::time::Duration::from_secs(*max_silence_secs);
                let cutoff = now.checked_sub(max_silence);

                let has_recent = match cutoff {
                    Some(c) => snapshots
                        .iter()
                        .any(|s| s.metric == *metric && s.timestamp >= c),
                    None => !snapshots.iter().any(|s| s.metric == *metric),
                };

                if !has_recent {
                    let msg = render_template(&rule.message_template, f64::NAN, None, metric);
                    Some(Alert {
                        rule_id: rule.id.clone(),
                        rule_name: rule.name.clone(),
                        severity: rule.severity.clone(),
                        message: msg,
                        metric_value: f64::NAN,
                        fired_at: now,
                    })
                } else {
                    None
                }
            }

            AlertCondition::Composite { conditions, logic } => {
                let results: Vec<bool> = conditions
                    .iter()
                    .map(|c| Self::evaluate_condition(rule, c, snapshots, now).is_some())
                    .collect();

                let fires = match logic {
                    LogicOp::And => results.iter().all(|&b| b),
                    LogicOp::Or => results.iter().any(|&b| b),
                };

                if fires {
                    // Use the value from the first matching snapshot if available.
                    let metric_value =
                        Self::first_metric_value(conditions, snapshots).unwrap_or(f64::NAN);
                    let msg =
                        render_template(&rule.message_template, metric_value, None, "composite");
                    Some(Alert {
                        rule_id: rule.id.clone(),
                        rule_name: rule.name.clone(),
                        severity: rule.severity.clone(),
                        message: msg,
                        metric_value,
                        fired_at: now,
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Find the latest [`MetricSnapshot`] for `metric_name`.
    fn latest_for<'a>(
        snapshots: &'a [MetricSnapshot],
        metric_name: &str,
    ) -> Option<&'a MetricSnapshot> {
        snapshots
            .iter()
            .filter(|s| s.metric == metric_name)
            .max_by_key(|s| s.timestamp)
    }

    /// Extract the first observable metric value from a list of conditions for
    /// use in composite alert messages.
    fn first_metric_value(
        conditions: &[AlertCondition],
        snapshots: &[MetricSnapshot],
    ) -> Option<f64> {
        for cond in conditions {
            if let AlertCondition::Threshold { metric, .. } = cond {
                if let Some(snap) = Self::latest_for(snapshots, metric) {
                    return Some(snap.value);
                }
            }
        }
        None
    }
}

impl Default for AlertRuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Template rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Render a message template, substituting known placeholders.
fn render_template(template: &str, value: f64, threshold: Option<f64>, metric: &str) -> String {
    let value_str = if value.is_nan() {
        "N/A".to_string()
    } else {
        format!("{value:.4}")
    };

    let threshold_str = threshold.map_or_else(|| "N/A".to_string(), |t| format!("{t:.4}"));

    template
        .replace("{value}", &value_str)
        .replace("{threshold}", &threshold_str)
        .replace("{metric}", metric)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    // ── Helper ────────────────────────────────────────────────────────────────

    fn snap(metric: &str, value: f64) -> MetricSnapshot {
        MetricSnapshot {
            metric: metric.to_string(),
            value,
            timestamp: SystemTime::now(),
        }
    }

    fn snap_at(metric: &str, value: f64, offset_secs_ago: u64) -> MetricSnapshot {
        MetricSnapshot {
            metric: metric.to_string(),
            value,
            timestamp: SystemTime::now()
                .checked_sub(Duration::from_secs(offset_secs_ago))
                .unwrap_or(SystemTime::UNIX_EPOCH),
        }
    }

    // ── CompareOp ─────────────────────────────────────────────────────────────

    #[test]
    fn test_compare_op_gt() {
        assert!(CompareOp::Gt.apply(5.0, 3.0));
        assert!(!CompareOp::Gt.apply(3.0, 5.0));
        assert!(!CompareOp::Gt.apply(3.0, 3.0));
    }

    #[test]
    fn test_compare_op_lte() {
        assert!(CompareOp::Lte.apply(3.0, 3.0));
        assert!(CompareOp::Lte.apply(2.0, 3.0));
        assert!(!CompareOp::Lte.apply(4.0, 3.0));
    }

    #[test]
    fn test_compare_op_eq_ne() {
        assert!(CompareOp::Eq.apply(1.0, 1.0));
        assert!(!CompareOp::Eq.apply(1.0, 2.0));
        assert!(CompareOp::Ne.apply(1.0, 2.0));
        assert!(!CompareOp::Ne.apply(1.0, 1.0));
    }

    // ── AlertRule factories ───────────────────────────────────────────────────

    #[test]
    fn test_high_cpu_factory() {
        let rule = AlertRule::high_cpu(90.0);
        assert_eq!(rule.id, "builtin.high_cpu");
        assert!(matches!(rule.severity, AlertSeverity::Warning));
        assert!(rule.enabled);
    }

    #[test]
    fn test_disk_full_factory() {
        let rule = AlertRule::disk_full(85.0);
        assert_eq!(rule.id, "builtin.disk_full");
        assert!(matches!(rule.severity, AlertSeverity::Critical));
    }

    #[test]
    fn test_queue_depth_factory() {
        let rule = AlertRule::queue_depth(100.0);
        assert_eq!(rule.id, "builtin.queue_depth");
    }

    #[test]
    fn test_high_memory_factory() {
        let rule = AlertRule::high_memory(80.0);
        assert_eq!(rule.id, "builtin.high_memory");
    }

    #[test]
    fn test_transcode_failure_rate_factory() {
        let rule = AlertRule::transcode_failure_rate(0.05);
        assert_eq!(rule.id, "builtin.transcode_failure_rate");
        assert!(matches!(rule.severity, AlertSeverity::Critical));
    }

    // ── Threshold condition ───────────────────────────────────────────────────

    #[test]
    fn test_threshold_fires_when_exceeded() {
        let mut engine = AlertRuleEngine::new();
        let rule = AlertRule::high_cpu(80.0);
        engine.add_rule(rule);

        let snaps = vec![snap("cpu.usage_pct", 95.0)];
        let alerts = engine.evaluate(&snaps);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].rule_id, "builtin.high_cpu");
    }

    #[test]
    fn test_threshold_does_not_fire_below() {
        let mut engine = AlertRuleEngine::new();
        engine.add_rule(AlertRule::high_cpu(90.0));

        let snaps = vec![snap("cpu.usage_pct", 50.0)];
        let alerts = engine.evaluate(&snaps);
        assert!(alerts.is_empty());
    }

    // ── Cooldown ──────────────────────────────────────────────────────────────

    #[test]
    fn test_cooldown_suppresses_repeat_alert() {
        let mut engine = AlertRuleEngine::new();
        let mut rule = AlertRule::high_cpu(80.0);
        rule.cooldown_secs = 300; // 5 minutes
        engine.add_rule(rule);

        let snaps = vec![snap("cpu.usage_pct", 95.0)];
        let first = engine.evaluate(&snaps);
        assert_eq!(first.len(), 1);

        // Second evaluation within cooldown → suppressed.
        let second = engine.evaluate(&snaps);
        assert!(second.is_empty(), "alert should be suppressed by cooldown");
    }

    #[test]
    fn test_zero_cooldown_fires_repeatedly() {
        let mut engine = AlertRuleEngine::new();
        let mut rule = AlertRule::high_cpu(80.0);
        rule.cooldown_secs = 0;
        engine.add_rule(rule);

        let snaps = vec![snap("cpu.usage_pct", 95.0)];
        let first = engine.evaluate(&snaps);
        let second = engine.evaluate(&snaps);
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
    }

    // ── Active alerts & clear ─────────────────────────────────────────────────

    #[test]
    fn test_active_alerts_returns_fired() {
        let mut engine = AlertRuleEngine::new();
        let mut rule = AlertRule::high_cpu(80.0);
        rule.cooldown_secs = 0;
        engine.add_rule(rule);

        let snaps = vec![snap("cpu.usage_pct", 95.0)];
        engine.evaluate(&snaps);
        assert_eq!(engine.active_alerts().len(), 1);
    }

    #[test]
    fn test_clear_alert_removes_from_active() {
        let mut engine = AlertRuleEngine::new();
        let mut rule = AlertRule::high_cpu(80.0);
        rule.cooldown_secs = 0;
        engine.add_rule(rule);

        let snaps = vec![snap("cpu.usage_pct", 95.0)];
        engine.evaluate(&snaps);
        engine.clear_alert("builtin.high_cpu");
        assert!(engine.active_alerts().is_empty());
    }

    // ── Absence condition ─────────────────────────────────────────────────────

    #[test]
    fn test_absence_fires_when_no_recent_snapshot() {
        let mut engine = AlertRuleEngine::new();
        let rule = AlertRule::new(
            "absence_test",
            "Absence Test",
            AlertCondition::Absence {
                metric: "heartbeat".to_string(),
                max_silence_secs: 30,
            },
            AlertSeverity::Warning,
        );
        engine.add_rule(rule);

        // Snapshot is 60 seconds old → beyond silence window.
        let snaps = vec![snap_at("heartbeat", 1.0, 60)];
        let alerts = engine.evaluate(&snaps);
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn test_absence_does_not_fire_when_recent_snapshot_exists() {
        let mut engine = AlertRuleEngine::new();
        let rule = AlertRule::new(
            "absence_test",
            "Absence Test",
            AlertCondition::Absence {
                metric: "heartbeat".to_string(),
                max_silence_secs: 60,
            },
            AlertSeverity::Warning,
        );
        engine.add_rule(rule);

        // Current snapshot → within silence window.
        let snaps = vec![snap("heartbeat", 1.0)];
        let alerts = engine.evaluate(&snaps);
        assert!(alerts.is_empty());
    }

    // ── Rate of change condition ──────────────────────────────────────────────

    #[test]
    fn test_rate_of_change_fires() {
        let mut engine = AlertRuleEngine::new();
        let rule = AlertRule::new(
            "roc_test",
            "Rate of Change",
            AlertCondition::RateOfChange {
                metric: "queue.depth".to_string(),
                window_secs: 120,
                min_change: 1.0, // 1 unit/sec
            },
            AlertSeverity::Warning,
        );
        engine.add_rule(rule);

        // 100 units change over 10 seconds = 10 units/sec → fires.
        let snaps = vec![snap_at("queue.depth", 0.0, 10), snap("queue.depth", 100.0)];
        let alerts = engine.evaluate(&snaps);
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn test_rate_of_change_does_not_fire_below_threshold() {
        let mut engine = AlertRuleEngine::new();
        let rule = AlertRule::new(
            "roc_test",
            "Rate of Change",
            AlertCondition::RateOfChange {
                metric: "queue.depth".to_string(),
                window_secs: 120,
                min_change: 50.0, // 50 units/sec
            },
            AlertSeverity::Warning,
        );
        engine.add_rule(rule);

        // 1 unit change over 10 seconds = 0.1 units/sec → does not fire.
        let snaps = vec![snap_at("queue.depth", 0.0, 10), snap("queue.depth", 1.0)];
        let alerts = engine.evaluate(&snaps);
        assert!(alerts.is_empty());
    }

    // ── Composite conditions ──────────────────────────────────────────────────

    #[test]
    fn test_composite_and_fires_when_both_true() {
        let mut engine = AlertRuleEngine::new();
        let rule = AlertRule::new(
            "composite_and",
            "Composite AND",
            AlertCondition::Composite {
                conditions: vec![
                    AlertCondition::Threshold {
                        metric: "cpu.usage_pct".to_string(),
                        op: CompareOp::Gt,
                        value: 80.0,
                    },
                    AlertCondition::Threshold {
                        metric: "memory.usage_pct".to_string(),
                        op: CompareOp::Gt,
                        value: 80.0,
                    },
                ],
                logic: LogicOp::And,
            },
            AlertSeverity::Critical,
        );
        engine.add_rule(rule);

        let snaps = vec![snap("cpu.usage_pct", 90.0), snap("memory.usage_pct", 85.0)];
        let alerts = engine.evaluate(&snaps);
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn test_composite_and_does_not_fire_when_one_false() {
        let mut engine = AlertRuleEngine::new();
        let rule = AlertRule::new(
            "composite_and",
            "Composite AND",
            AlertCondition::Composite {
                conditions: vec![
                    AlertCondition::Threshold {
                        metric: "cpu.usage_pct".to_string(),
                        op: CompareOp::Gt,
                        value: 80.0,
                    },
                    AlertCondition::Threshold {
                        metric: "memory.usage_pct".to_string(),
                        op: CompareOp::Gt,
                        value: 80.0,
                    },
                ],
                logic: LogicOp::And,
            },
            AlertSeverity::Critical,
        );
        engine.add_rule(rule);

        // CPU fires but memory does not.
        let snaps = vec![snap("cpu.usage_pct", 90.0), snap("memory.usage_pct", 50.0)];
        let alerts = engine.evaluate(&snaps);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_composite_or_fires_when_one_true() {
        let mut engine = AlertRuleEngine::new();
        let rule = AlertRule::new(
            "composite_or",
            "Composite OR",
            AlertCondition::Composite {
                conditions: vec![
                    AlertCondition::Threshold {
                        metric: "cpu.usage_pct".to_string(),
                        op: CompareOp::Gt,
                        value: 80.0,
                    },
                    AlertCondition::Threshold {
                        metric: "memory.usage_pct".to_string(),
                        op: CompareOp::Gt,
                        value: 80.0,
                    },
                ],
                logic: LogicOp::Or,
            },
            AlertSeverity::Warning,
        );
        engine.add_rule(rule);

        // Only CPU fires.
        let snaps = vec![snap("cpu.usage_pct", 90.0), snap("memory.usage_pct", 50.0)];
        let alerts = engine.evaluate(&snaps);
        assert_eq!(alerts.len(), 1);
    }

    // ── remove_rule ───────────────────────────────────────────────────────────

    #[test]
    fn test_remove_rule() {
        let mut engine = AlertRuleEngine::new();
        engine.add_rule(AlertRule::high_cpu(80.0));
        assert!(engine.remove_rule("builtin.high_cpu"));
        assert!(!engine.remove_rule("builtin.high_cpu")); // already removed

        let snaps = vec![snap("cpu.usage_pct", 95.0)];
        let alerts = engine.evaluate(&snaps);
        assert!(alerts.is_empty());
    }

    // ── Message template rendering ────────────────────────────────────────────

    #[test]
    fn test_message_template_value_substitution() {
        let mut engine = AlertRuleEngine::new();
        let mut rule = AlertRule::high_cpu(80.0);
        rule.message_template = "CPU is at {value}%".to_string();
        rule.cooldown_secs = 0;
        engine.add_rule(rule);

        let snaps = vec![snap("cpu.usage_pct", 95.0)];
        let alerts = engine.evaluate(&snaps);
        assert_eq!(alerts.len(), 1);
        assert!(
            alerts[0].message.contains("95"),
            "message was: {}",
            alerts[0].message
        );
    }
}
