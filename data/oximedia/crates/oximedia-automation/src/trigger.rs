//! Automation triggers: time-based, event-based, and condition-based.
//!
//! This module provides the trigger infrastructure for the automation system,
//! allowing actions to fire based on schedules, events, or logical conditions.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Unique identifier for a trigger.
pub type TriggerId = u64;

/// The current evaluation state of a trigger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerState {
    /// Trigger is active and will fire when its condition is met.
    Active,
    /// Trigger has fired and is waiting to be reset.
    Fired,
    /// Trigger is disabled and will not fire.
    Disabled,
    /// Trigger has been consumed (one-shot) and will not fire again.
    Consumed,
}

/// How often a trigger may fire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerRepeat {
    /// Fire once then transition to `Consumed`.
    Once,
    /// Fire every time the condition becomes true.
    Always,
    /// Fire at most `n` times.
    Count(u32),
}

/// A time-based trigger that fires at a fixed wall-clock offset from its
/// creation instant.
#[derive(Debug, Clone)]
pub struct TimeTrigger {
    pub id: TriggerId,
    pub delay: Duration,
    pub repeat_interval: Option<Duration>,
    pub repeat: TriggerRepeat,
    pub state: TriggerState,
    created_at: Instant,
    last_fired_at: Option<Instant>,
    fire_count: u32,
}

impl TimeTrigger {
    /// Create a new one-shot time trigger.
    pub fn new_oneshot(id: TriggerId, delay: Duration) -> Self {
        Self {
            id,
            delay,
            repeat_interval: None,
            repeat: TriggerRepeat::Once,
            state: TriggerState::Active,
            created_at: Instant::now(),
            last_fired_at: None,
            fire_count: 0,
        }
    }

    /// Create a repeating time trigger.
    pub fn new_repeating(id: TriggerId, delay: Duration, interval: Duration) -> Self {
        Self {
            id,
            delay,
            repeat_interval: Some(interval),
            repeat: TriggerRepeat::Always,
            state: TriggerState::Active,
            created_at: Instant::now(),
            last_fired_at: None,
            fire_count: 0,
        }
    }

    /// Evaluate the trigger against the current instant.
    /// Returns `true` if the trigger fires.
    pub fn evaluate(&mut self, now: Instant) -> bool {
        if self.state != TriggerState::Active {
            return false;
        }
        let elapsed = now.duration_since(self.created_at);
        if elapsed < self.delay {
            return false;
        }
        // Check repeat interval
        if let Some(interval) = self.repeat_interval {
            if let Some(last) = self.last_fired_at {
                if now.duration_since(last) < interval {
                    return false;
                }
            }
        }
        // Fire
        self.fire_count += 1;
        self.last_fired_at = Some(now);
        match &self.repeat {
            TriggerRepeat::Once => self.state = TriggerState::Consumed,
            TriggerRepeat::Always => {}
            TriggerRepeat::Count(max) => {
                if self.fire_count >= *max {
                    self.state = TriggerState::Consumed;
                }
            }
        }
        true
    }

    /// How many times this trigger has fired.
    pub fn fire_count(&self) -> u32 {
        self.fire_count
    }

    /// Disable this trigger.
    pub fn disable(&mut self) {
        self.state = TriggerState::Disabled;
    }

    /// Re-enable a disabled trigger.
    pub fn enable(&mut self) {
        if self.state == TriggerState::Disabled {
            self.state = TriggerState::Active;
        }
    }
}

// ---------------------------------------------------------------------------
// Event-based trigger
// ---------------------------------------------------------------------------

/// An event tag used by the event bus.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventTag(pub String);

impl EventTag {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// An event-based trigger fires when a matching event is emitted.
#[derive(Debug, Clone)]
pub struct EventTrigger {
    pub id: TriggerId,
    pub event_tag: EventTag,
    pub repeat: TriggerRepeat,
    pub state: TriggerState,
    fire_count: u32,
}

impl EventTrigger {
    pub fn new(id: TriggerId, event_tag: EventTag) -> Self {
        Self {
            id,
            event_tag,
            repeat: TriggerRepeat::Always,
            state: TriggerState::Active,
            fire_count: 0,
        }
    }

    pub fn new_oneshot(id: TriggerId, event_tag: EventTag) -> Self {
        Self {
            id,
            event_tag,
            repeat: TriggerRepeat::Once,
            state: TriggerState::Active,
            fire_count: 0,
        }
    }

    /// Called when an event arrives. Returns `true` if this trigger fires.
    pub fn on_event(&mut self, tag: &EventTag) -> bool {
        if self.state != TriggerState::Active {
            return false;
        }
        if &self.event_tag != tag {
            return false;
        }
        self.fire_count += 1;
        if self.repeat == TriggerRepeat::Once {
            self.state = TriggerState::Consumed;
        }
        true
    }

    pub fn fire_count(&self) -> u32 {
        self.fire_count
    }
}

// ---------------------------------------------------------------------------
// Condition-based trigger
// ---------------------------------------------------------------------------

/// Logical operator for combining conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionOp {
    And,
    Or,
    Not,
}

/// A named signal that can be set to a boolean value.
#[derive(Debug, Clone)]
pub struct Signal {
    pub name: String,
    pub value: bool,
}

impl Signal {
    pub fn new(name: impl Into<String>, value: bool) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// A condition-based trigger evaluates a logical expression over named signals.
#[derive(Debug, Clone)]
pub struct ConditionTrigger {
    pub id: TriggerId,
    pub signal_name: String,
    pub expected_value: bool,
    pub op: ConditionOp,
    pub state: TriggerState,
    fire_count: u32,
}

impl ConditionTrigger {
    pub fn new(id: TriggerId, signal_name: impl Into<String>, expected: bool) -> Self {
        Self {
            id,
            signal_name: signal_name.into(),
            expected_value: expected,
            op: ConditionOp::And,
            state: TriggerState::Active,
            fire_count: 0,
        }
    }

    /// Evaluate against a slice of current signals.
    pub fn evaluate(&mut self, signals: &[Signal]) -> bool {
        if self.state != TriggerState::Active {
            return false;
        }
        let matched = signals
            .iter()
            .find(|s| s.name == self.signal_name)
            .is_some_and(|s| s.value == self.expected_value);
        if matched {
            self.fire_count += 1;
        }
        matched
    }

    pub fn fire_count(&self) -> u32 {
        self.fire_count
    }

    pub fn disable(&mut self) {
        self.state = TriggerState::Disabled;
    }
}

// ---------------------------------------------------------------------------
// Trigger registry
// ---------------------------------------------------------------------------

/// A simple registry holding all trigger types.
#[derive(Debug, Default)]
pub struct TriggerRegistry {
    time_triggers: Vec<TimeTrigger>,
    event_triggers: Vec<EventTrigger>,
    condition_triggers: Vec<ConditionTrigger>,
    next_id: TriggerId,
}

impl TriggerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_id(&mut self) -> TriggerId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn add_time(&mut self, t: TimeTrigger) {
        self.time_triggers.push(t);
    }

    pub fn add_event(&mut self, t: EventTrigger) {
        self.event_triggers.push(t);
    }

    pub fn add_condition(&mut self, t: ConditionTrigger) {
        self.condition_triggers.push(t);
    }

    /// Evaluate all time triggers, returning IDs of those that fired.
    pub fn tick_time(&mut self, now: Instant) -> Vec<TriggerId> {
        let mut fired = Vec::new();
        for t in &mut self.time_triggers {
            if t.evaluate(now) {
                fired.push(t.id);
            }
        }
        fired
    }

    /// Dispatch an event, returning IDs of event triggers that fired.
    pub fn dispatch_event(&mut self, tag: &EventTag) -> Vec<TriggerId> {
        let mut fired = Vec::new();
        for t in &mut self.event_triggers {
            if t.on_event(tag) {
                fired.push(t.id);
            }
        }
        fired
    }

    /// Evaluate condition triggers against the current signal set.
    pub fn evaluate_conditions(&mut self, signals: &[Signal]) -> Vec<TriggerId> {
        let mut fired = Vec::new();
        for t in &mut self.condition_triggers {
            if t.evaluate(signals) {
                fired.push(t.id);
            }
        }
        fired
    }

    pub fn time_trigger_count(&self) -> usize {
        self.time_triggers.len()
    }

    pub fn event_trigger_count(&self) -> usize {
        self.event_triggers.len()
    }

    pub fn condition_trigger_count(&self) -> usize {
        self.condition_triggers.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_time_trigger_oneshot_fires_after_delay() {
        let mut t = TimeTrigger::new_oneshot(1, Duration::from_millis(0));
        // zero delay → fires immediately
        assert!(t.evaluate(Instant::now()));
        assert_eq!(t.state, TriggerState::Consumed);
    }

    #[test]
    fn test_time_trigger_does_not_fire_before_delay() {
        let mut t = TimeTrigger::new_oneshot(2, Duration::from_secs(9999));
        assert!(!t.evaluate(Instant::now()));
        assert_eq!(t.state, TriggerState::Active);
    }

    #[test]
    fn test_time_trigger_consumed_does_not_fire_again() {
        let mut t = TimeTrigger::new_oneshot(3, Duration::from_millis(0));
        assert!(t.evaluate(Instant::now()));
        assert!(!t.evaluate(Instant::now()));
    }

    #[test]
    fn test_time_trigger_disabled_does_not_fire() {
        let mut t = TimeTrigger::new_oneshot(4, Duration::from_millis(0));
        t.disable();
        assert!(!t.evaluate(Instant::now()));
    }

    #[test]
    fn test_time_trigger_enable_after_disable() {
        let mut t = TimeTrigger::new_oneshot(5, Duration::from_millis(0));
        t.disable();
        t.enable();
        assert!(t.evaluate(Instant::now()));
    }

    #[test]
    fn test_time_trigger_repeating_fires_multiple_times() {
        let mut t =
            TimeTrigger::new_repeating(6, Duration::from_millis(0), Duration::from_millis(0));
        let now = Instant::now();
        assert!(t.evaluate(now));
        // With zero interval, fires again
        let later = now + Duration::from_millis(1);
        assert!(t.evaluate(later));
        assert_eq!(t.fire_count(), 2);
    }

    #[test]
    fn test_event_trigger_fires_on_matching_tag() {
        let tag = EventTag::new("clip.end");
        let mut t = EventTrigger::new(10, tag.clone());
        assert!(t.on_event(&tag));
        assert_eq!(t.fire_count(), 1);
    }

    #[test]
    fn test_event_trigger_ignores_non_matching_tag() {
        let tag = EventTag::new("clip.end");
        let other = EventTag::new("clip.start");
        let mut t = EventTrigger::new(11, tag);
        assert!(!t.on_event(&other));
        assert_eq!(t.fire_count(), 0);
    }

    #[test]
    fn test_event_trigger_oneshot_consumed_after_one_fire() {
        let tag = EventTag::new("test");
        let mut t = EventTrigger::new_oneshot(12, tag.clone());
        assert!(t.on_event(&tag));
        assert!(!t.on_event(&tag));
        assert_eq!(t.fire_count(), 1);
    }

    #[test]
    fn test_condition_trigger_fires_when_signal_matches() {
        let mut t = ConditionTrigger::new(20, "air", true);
        let signals = vec![Signal::new("air", true)];
        assert!(t.evaluate(&signals));
        assert_eq!(t.fire_count(), 1);
    }

    #[test]
    fn test_condition_trigger_does_not_fire_when_signal_mismatches() {
        let mut t = ConditionTrigger::new(21, "air", true);
        let signals = vec![Signal::new("air", false)];
        assert!(!t.evaluate(&signals));
    }

    #[test]
    fn test_condition_trigger_disabled() {
        let mut t = ConditionTrigger::new(22, "air", true);
        t.disable();
        let signals = vec![Signal::new("air", true)];
        assert!(!t.evaluate(&signals));
    }

    #[test]
    fn test_registry_time_trigger_tick() {
        let mut reg = TriggerRegistry::new();
        let id = reg.next_id();
        reg.add_time(TimeTrigger::new_oneshot(id, Duration::from_millis(0)));
        let fired = reg.tick_time(Instant::now());
        assert_eq!(fired, vec![id]);
        assert_eq!(reg.time_trigger_count(), 1);
    }

    #[test]
    fn test_registry_event_dispatch() {
        let mut reg = TriggerRegistry::new();
        let id = reg.next_id();
        let tag = EventTag::new("evt");
        reg.add_event(EventTrigger::new(id, tag.clone()));
        let fired = reg.dispatch_event(&tag);
        assert_eq!(fired, vec![id]);
    }

    #[test]
    fn test_registry_condition_evaluate() {
        let mut reg = TriggerRegistry::new();
        let id = reg.next_id();
        reg.add_condition(ConditionTrigger::new(id, "ready", true));
        let signals = vec![Signal::new("ready", true)];
        let fired = reg.evaluate_conditions(&signals);
        assert_eq!(fired, vec![id]);
    }
}
