//! End-to-end integration test for the full alert pipeline:
//! *metric threshold → rule evaluation → notification dispatch*.
//!
//! These tests exercise the real crate APIs with no mocking of the crate's own
//! logic. The only test-supplied component is a capturing notification sink —
//! a genuine implementation of the crate's [`AlertChannel`] trait that records
//! every alert it is handed — so we can observe exactly what the pipeline
//! dispatches.
//!
//! Pipeline under test: [`AlertManager::evaluate_metric`] takes a single metric
//! sample, evaluates it against every matching [`AlertRule`] (whose
//! [`ThresholdCondition`] does the actual threshold comparison) and dispatches
//! the resulting [`Alert`] to every registered [`AlertChannel`]. The pipeline is
//! edge-triggered, so it also covers deduplication (no repeat while firing) and
//! recovery (a `Resolved` alert when the metric returns below threshold).

#![allow(clippy::float_cmp)]

use async_trait::async_trait;
use oximedia_monitor::alert::conditions::{AlertCondition, ComparisonOperator, ThresholdCondition};
use oximedia_monitor::alert::{AlertChannel, AlertManager, AlertState};
use oximedia_monitor::{Alert, AlertRule, AlertSeverity, MonitorResult};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Capturing notification sink
// ---------------------------------------------------------------------------

/// A deterministic, in-memory [`AlertChannel`] that captures every dispatched
/// alert into a shared vector for later inspection.
struct CapturingChannel {
    captured: Arc<Mutex<Vec<Alert>>>,
}

impl CapturingChannel {
    fn new() -> Self {
        Self {
            captured: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A clonable handle to the captured alerts.
    fn handle(&self) -> Arc<Mutex<Vec<Alert>>> {
        self.captured.clone()
    }
}

#[async_trait]
impl AlertChannel for CapturingChannel {
    async fn send(&self, alert: &Alert) -> MonitorResult<()> {
        self.captured
            .lock()
            .expect("capture mutex poisoned")
            .push(alert.clone());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A `cpu.usage > 90` critical rule.
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

/// Build a manager wired with the cpu rule and a single capturing sink.
/// Returns the manager and a handle onto the captured alerts.
fn manager_with_capture() -> (AlertManager, Arc<Mutex<Vec<Alert>>>) {
    let manager = AlertManager::new(oximedia_monitor::AlertConfig::default());
    manager.add_rule(cpu_rule());
    let sink = CapturingChannel::new();
    let captured = sink.handle();
    manager.add_channel(Arc::new(sink));
    (manager, captured)
}

fn snapshot(captured: &Arc<Mutex<Vec<Alert>>>) -> Vec<Alert> {
    captured.lock().expect("capture mutex poisoned").clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A metric value below the threshold must not produce any notification.
#[tokio::test]
async fn test_metric_below_threshold_dispatches_nothing() {
    let (manager, captured) = manager_with_capture();

    let dispatched = manager
        .evaluate_metric("cpu.usage", 42.0)
        .await
        .expect("evaluate below threshold");

    assert!(
        dispatched.is_empty(),
        "no alert should be dispatched below threshold, got {dispatched:?}"
    );
    assert!(
        snapshot(&captured).is_empty(),
        "capturing sink must have received nothing"
    );
    assert_eq!(manager.firing_count(), 0);
}

/// Crossing the threshold fires the rule and dispatches a notification carrying
/// the correct severity, message and metric identity.
#[tokio::test]
async fn test_threshold_crossing_dispatches_notification_with_full_identity() {
    let (manager, captured) = manager_with_capture();

    // Start below threshold so we observe an actual crossing.
    let _ = manager
        .evaluate_metric("cpu.usage", 10.0)
        .await
        .expect("evaluate below");
    assert!(snapshot(&captured).is_empty());

    let dispatched = manager
        .evaluate_metric("cpu.usage", 95.0)
        .await
        .expect("evaluate above");

    assert_eq!(dispatched.len(), 1, "exactly one alert should fire");

    let captured_alerts = snapshot(&captured);
    assert_eq!(
        captured_alerts.len(),
        1,
        "the capturing sink must have received exactly the fired alert"
    );

    let alert = &captured_alerts[0];
    assert_eq!(alert.rule_name, "cpu_high");
    assert_eq!(alert.metric_name, "cpu.usage");
    assert_eq!(alert.severity, AlertSeverity::Critical);
    assert_eq!(alert.message, "CPU usage exceeded 90%");
    assert_eq!(alert.threshold, Some(90.0));
    assert_eq!(alert.metric_value, 95.0);
    assert_eq!(alert.state, AlertState::Firing);

    // The dispatched value returned to the caller matches what the sink saw.
    assert_eq!(dispatched[0].id, alert.id);
    assert_eq!(manager.firing_count(), 1);
}

/// While a rule stays breached, repeated evaluations must not re-dispatch
/// (deduplication / debounce).
#[tokio::test]
async fn test_repeated_breach_is_deduplicated() {
    let (manager, captured) = manager_with_capture();

    let first = manager
        .evaluate_metric("cpu.usage", 95.0)
        .await
        .expect("first breach");
    assert_eq!(first.len(), 1, "first breach fires");

    // Two further over-threshold samples while still firing.
    let second = manager
        .evaluate_metric("cpu.usage", 99.0)
        .await
        .expect("second breach");
    let third = manager
        .evaluate_metric("cpu.usage", 100.0)
        .await
        .expect("third breach");

    assert!(second.is_empty(), "repeat breach must not re-dispatch");
    assert!(third.is_empty(), "repeat breach must not re-dispatch");

    assert_eq!(
        snapshot(&captured).len(),
        1,
        "sink should have received exactly one notification across repeats"
    );
    assert_eq!(manager.firing_count(), 1);
}

/// When a firing metric returns below threshold, a single `Resolved` recovery
/// notification is dispatched and the firing state clears.
#[tokio::test]
async fn test_recovery_below_threshold_emits_resolved_notification() {
    let (manager, captured) = manager_with_capture();

    let _ = manager
        .evaluate_metric("cpu.usage", 95.0)
        .await
        .expect("fire");
    assert_eq!(manager.firing_count(), 1);

    let recovery = manager
        .evaluate_metric("cpu.usage", 30.0)
        .await
        .expect("recovery");

    assert_eq!(recovery.len(), 1, "recovery dispatches one resolved alert");
    assert_eq!(recovery[0].state, AlertState::Resolved);
    assert_eq!(recovery[0].rule_name, "cpu_high");
    assert_eq!(recovery[0].metric_value, 30.0);
    assert_eq!(
        manager.firing_count(),
        0,
        "firing state cleared on recovery"
    );

    let captured_alerts = snapshot(&captured);
    assert_eq!(
        captured_alerts.len(),
        2,
        "fire + recovery were both dispatched"
    );
    assert_eq!(captured_alerts[0].state, AlertState::Firing);
    assert_eq!(captured_alerts[1].state, AlertState::Resolved);
}

/// Full lifecycle in one flow: below → cross → repeat (deduped) → recover →
/// re-fire. Verifies the complete edge-triggered pipeline end to end.
#[tokio::test]
async fn test_full_lifecycle_below_cross_repeat_recover_refire() {
    let (manager, captured) = manager_with_capture();

    // 1. Below threshold — silent.
    assert!(manager
        .evaluate_metric("cpu.usage", 12.0)
        .await
        .expect("below")
        .is_empty());

    // 2. Cross — fires.
    assert_eq!(
        manager
            .evaluate_metric("cpu.usage", 91.0)
            .await
            .expect("cross")
            .len(),
        1
    );

    // 3. Repeat while firing — deduplicated.
    assert!(manager
        .evaluate_metric("cpu.usage", 99.0)
        .await
        .expect("repeat")
        .is_empty());

    // 4. Recover — resolved.
    let recovery = manager
        .evaluate_metric("cpu.usage", 5.0)
        .await
        .expect("recover");
    assert_eq!(recovery.len(), 1);
    assert_eq!(recovery[0].state, AlertState::Resolved);

    // 5. Breach again after recovery — fires afresh.
    let refire = manager
        .evaluate_metric("cpu.usage", 93.0)
        .await
        .expect("refire");
    assert_eq!(refire.len(), 1);
    assert_eq!(refire[0].state, AlertState::Firing);

    // Sink saw: fire, resolve, fire (3 notifications total).
    let states: Vec<AlertState> = snapshot(&captured).iter().map(|a| a.state).collect();
    assert_eq!(
        states,
        vec![AlertState::Firing, AlertState::Resolved, AlertState::Firing]
    );
    assert_eq!(manager.firing_count(), 1);
}

/// A sample for a metric no rule watches must be ignored entirely.
#[tokio::test]
async fn test_unrelated_metric_name_is_ignored() {
    let (manager, captured) = manager_with_capture();

    let dispatched = manager
        .evaluate_metric("memory.usage", 999.0)
        .await
        .expect("unrelated metric");

    assert!(dispatched.is_empty());
    assert!(snapshot(&captured).is_empty());
    assert_eq!(manager.firing_count(), 0);
}

/// A disabled rule must never fire, even when its threshold is breached.
#[tokio::test]
async fn test_disabled_rule_never_dispatches() {
    let manager = AlertManager::new(oximedia_monitor::AlertConfig::default());
    let mut rule = cpu_rule();
    rule.enabled = false;
    manager.add_rule(rule);

    let sink = CapturingChannel::new();
    let captured = sink.handle();
    manager.add_channel(Arc::new(sink));

    let dispatched = manager
        .evaluate_metric("cpu.usage", 100.0)
        .await
        .expect("disabled rule");

    assert!(dispatched.is_empty(), "disabled rule must not fire");
    assert!(snapshot(&captured).is_empty());
    assert_eq!(manager.firing_count(), 0);
}

/// Every registered channel must receive each dispatched alert (fan-out).
#[tokio::test]
async fn test_multiple_channels_all_receive_dispatch() {
    let manager = AlertManager::new(oximedia_monitor::AlertConfig::default());
    manager.add_rule(cpu_rule());

    let sink_a = CapturingChannel::new();
    let sink_b = CapturingChannel::new();
    let captured_a = sink_a.handle();
    let captured_b = sink_b.handle();
    manager.add_channel(Arc::new(sink_a));
    manager.add_channel(Arc::new(sink_b));
    assert_eq!(manager.channel_count(), 2);

    let dispatched = manager
        .evaluate_metric("cpu.usage", 96.0)
        .await
        .expect("fan-out fire");
    assert_eq!(dispatched.len(), 1);

    let a = snapshot(&captured_a);
    let b = snapshot(&captured_b);
    assert_eq!(a.len(), 1, "channel A must receive the alert");
    assert_eq!(b.len(), 1, "channel B must receive the alert");
    assert_eq!(a[0].rule_name, "cpu_high");
    assert_eq!(b[0].rule_name, "cpu_high");
    assert_eq!(a[0].severity, AlertSeverity::Critical);
    assert_eq!(b[0].severity, AlertSeverity::Critical);
}
