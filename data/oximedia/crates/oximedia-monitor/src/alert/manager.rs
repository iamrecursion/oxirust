//! Alert manager for handling alerts.

use crate::alert::channels::AlertChannel;
use crate::alert::{Alert, AlertRule, AlertState};
use crate::config::AlertConfig;
use crate::error::{MonitorError, MonitorResult};
use chrono::Utc;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Alert manager.
///
/// Owns the full alert pipeline: it holds a set of [`AlertRule`]s plus a set of
/// notification [`AlertChannel`]s (sinks), and [`evaluate_metric`] connects the
/// two — evaluating an incoming metric sample against the matching rules and
/// dispatching notifications to every registered channel.
///
/// [`evaluate_metric`]: AlertManager::evaluate_metric
pub struct AlertManager {
    config: AlertConfig,
    rules: Arc<RwLock<Vec<AlertRule>>>,
    active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
    channels: Arc<RwLock<Vec<Arc<dyn AlertChannel>>>>,
    firing: Arc<RwLock<HashMap<String, Alert>>>,
    running: Arc<tokio::sync::RwLock<bool>>,
}

impl AlertManager {
    /// Create a new alert manager.
    #[must_use]
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            rules: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            channels: Arc::new(RwLock::new(Vec::new())),
            firing: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    /// Start the alert manager.
    pub async fn start(&self) -> crate::error::MonitorResult<()> {
        let mut running = self.running.write().await;
        *running = true;
        Ok(())
    }

    /// Stop the alert manager.
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Add a rule.
    pub fn add_rule(&self, rule: AlertRule) {
        self.rules.write().push(rule);
    }

    /// Fire an alert.
    pub fn fire(&self, alert: Alert) {
        self.active_alerts.write().insert(alert.id.clone(), alert);
    }

    /// Get active alerts.
    #[must_use]
    pub fn active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.read().values().cloned().collect()
    }

    /// Register a notification channel (sink) that dispatched alerts are sent to.
    pub fn add_channel(&self, channel: Arc<dyn AlertChannel>) {
        self.channels.write().push(channel);
    }

    /// Number of registered notification channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.read().len()
    }

    /// Number of rules that are currently in the firing state.
    #[must_use]
    pub fn firing_count(&self) -> usize {
        self.firing.read().len()
    }

    /// Snapshot of the alerts that are currently firing, keyed internally by
    /// rule name.
    #[must_use]
    pub fn firing_alerts(&self) -> Vec<Alert> {
        self.firing.read().values().cloned().collect()
    }

    /// Build an [`Alert`] for `rule` breaching (or recovering) at `value`.
    fn alert_from_rule(rule: &AlertRule, value: f64, state: AlertState) -> Alert {
        let mut alert = Alert::new(
            rule.name.clone(),
            rule.severity,
            rule.message.clone(),
            rule.metric_name.clone(),
            value,
        );
        if let Some(threshold) = rule.condition.threshold() {
            alert = alert.with_threshold(threshold);
        }
        alert.state = state;
        alert
    }

    /// Evaluate a single metric sample against every matching rule and dispatch
    /// notifications through the registered channels.
    ///
    /// This is the full alert pipeline:
    /// *metric value → threshold / rule evaluation → notification dispatch*.
    ///
    /// Behaviour is edge-triggered (hysteresis):
    /// * a rule that newly breaches its threshold fires once — a `Firing`
    ///   [`Alert`] is dispatched to every channel;
    /// * while a rule remains breached no further notifications are sent
    ///   (deduplication / debounce);
    /// * when a firing rule returns below its threshold a single `Resolved`
    ///   (recovery) alert is dispatched and the firing state is cleared, so a
    ///   later breach fires afresh.
    ///
    /// Returns the alerts dispatched during this call (firing and/or recovery).
    ///
    /// # Errors
    ///
    /// Returns the first channel delivery error encountered. Delivery is still
    /// attempted on every channel for every dispatched alert before returning.
    pub async fn evaluate_metric(
        &self,
        metric_name: &str,
        value: f64,
    ) -> MonitorResult<Vec<Alert>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        // Phase 1: evaluate rules and update the firing state under the locks.
        // No `.await` occurs while a lock is held.
        let to_dispatch: Vec<Alert> = {
            let rules = self.rules.read();
            let mut firing = self.firing.write();
            let mut out = Vec::new();

            for rule in rules.iter() {
                if rule.metric_name != metric_name || !rule.enabled {
                    continue;
                }

                let breached = rule.evaluate(value);
                let currently_firing = firing.contains_key(&rule.name);

                match (breached, currently_firing) {
                    // Newly breached: fire and remember.
                    (true, false) => {
                        let alert = Self::alert_from_rule(rule, value, AlertState::Firing);
                        firing.insert(rule.name.clone(), alert.clone());
                        out.push(alert);
                    }
                    // Still below / still above: nothing to dispatch.
                    (true, true) | (false, false) => {}
                    // Recovered: emit a resolved (clear) notification.
                    (false, true) => {
                        if let Some(mut alert) = firing.remove(&rule.name) {
                            alert.state = AlertState::Resolved;
                            alert.metric_value = value;
                            alert.timestamp = Utc::now();
                            out.push(alert);
                        }
                    }
                }
            }
            out
        };

        // Phase 2: dispatch to channels without holding any lock.
        let channels: Vec<Arc<dyn AlertChannel>> = self.channels.read().clone();
        let mut first_error: Option<MonitorError> = None;
        for alert in &to_dispatch {
            for channel in &channels {
                if let Err(err) = channel.send(alert).await {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        match first_error {
            Some(err) => Err(err),
            None => Ok(to_dispatch),
        }
    }
}

/// Alert deduplicator.
pub struct AlertDeduplicator {
    seen_alerts: Arc<RwLock<HashMap<String, chrono::DateTime<chrono::Utc>>>>,
}

impl AlertDeduplicator {
    /// Create a new deduplicator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            seen_alerts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if an alert should be deduplicated.
    #[must_use]
    pub fn should_deduplicate(&self, alert_id: &str, window_secs: i64) -> bool {
        let seen = self.seen_alerts.read();
        if let Some(last_seen) = seen.get(alert_id) {
            let now = chrono::Utc::now();
            let elapsed = now.signed_duration_since(*last_seen);
            elapsed.num_seconds() < window_secs
        } else {
            false
        }
    }

    /// Mark an alert as seen.
    pub fn mark_seen(&self, alert_id: String) {
        self.seen_alerts
            .write()
            .insert(alert_id, chrono::Utc::now());
    }
}

impl Default for AlertDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alert::conditions::{AlertCondition, ComparisonOperator, ThresholdCondition};
    use crate::alert::AlertSeverity;
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// In-memory [`AlertChannel`] that records every alert it receives.
    struct CapturingSink {
        captured: Arc<Mutex<Vec<Alert>>>,
    }

    #[async_trait]
    impl AlertChannel for CapturingSink {
        async fn send(&self, alert: &Alert) -> MonitorResult<()> {
            self.captured
                .lock()
                .expect("capture mutex poisoned")
                .push(alert.clone());
            Ok(())
        }
    }

    fn cpu_rule() -> AlertRule {
        AlertRule::new(
            "cpu_high",
            "cpu.usage",
            AlertCondition::Threshold(ThresholdCondition {
                operator: ComparisonOperator::GreaterThan,
                value: 90.0,
                duration_secs: 0,
            }),
            AlertSeverity::Critical,
            "CPU usage exceeded 90%",
        )
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let config = AlertConfig::default();
        let manager = AlertManager::new(config);

        manager.start().await.expect("await should be valid");

        let alert = Alert::new(
            "test",
            AlertSeverity::Warning,
            "Test alert",
            "test.metric",
            100.0,
        );

        manager.fire(alert);

        assert_eq!(manager.active_alerts().len(), 1);

        manager.stop().await;
    }

    #[tokio::test]
    async fn test_evaluate_metric_dispatches_on_breach() {
        let manager = AlertManager::new(AlertConfig::default());
        manager.add_rule(cpu_rule());
        let captured = Arc::new(Mutex::new(Vec::new()));
        manager.add_channel(Arc::new(CapturingSink {
            captured: captured.clone(),
        }));

        // Below threshold: nothing dispatched.
        let below = manager
            .evaluate_metric("cpu.usage", 50.0)
            .await
            .expect("evaluate below threshold");
        assert!(below.is_empty());
        assert!(captured.lock().expect("lock").is_empty());

        // Crosses threshold: one firing alert dispatched.
        let above = manager
            .evaluate_metric("cpu.usage", 95.0)
            .await
            .expect("evaluate above threshold");
        assert_eq!(above.len(), 1);

        let dispatched = captured.lock().expect("lock").clone();
        assert_eq!(dispatched.len(), 1);
        assert_eq!(dispatched[0].rule_name, "cpu_high");
        assert_eq!(dispatched[0].metric_name, "cpu.usage");
        assert_eq!(dispatched[0].severity, AlertSeverity::Critical);
        assert_eq!(dispatched[0].threshold, Some(90.0));
        assert_eq!(dispatched[0].state, AlertState::Firing);
        assert_eq!(manager.firing_count(), 1);
    }

    #[tokio::test]
    async fn test_evaluate_metric_recovery_and_dedup() {
        let manager = AlertManager::new(AlertConfig::default());
        manager.add_rule(cpu_rule());
        let captured = Arc::new(Mutex::new(Vec::new()));
        manager.add_channel(Arc::new(CapturingSink {
            captured: captured.clone(),
        }));

        // Fire.
        let _ = manager
            .evaluate_metric("cpu.usage", 95.0)
            .await
            .expect("fire");
        // Still breached: deduplicated (no second notification).
        let repeat = manager
            .evaluate_metric("cpu.usage", 97.0)
            .await
            .expect("repeat");
        assert!(repeat.is_empty());
        assert_eq!(captured.lock().expect("lock").len(), 1);

        // Drops below threshold: recovery notification.
        let recovery = manager
            .evaluate_metric("cpu.usage", 10.0)
            .await
            .expect("recovery");
        assert_eq!(recovery.len(), 1);
        assert_eq!(recovery[0].state, AlertState::Resolved);
        assert_eq!(manager.firing_count(), 0);
        assert_eq!(captured.lock().expect("lock").len(), 2);
    }
}
