//! Workflow trigger system for broadcast automation.
//!
//! Provides event-driven trigger evaluation and registry for
//! associating broadcast events with automation rules.

#![allow(dead_code)]

use std::collections::HashMap;

/// A trigger event that can fire in the automation system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerEvent {
    /// Timecode-based event
    Timecode(String),
    /// GPI input fired
    GpiInput(u8),
    /// Playlist item started
    PlaylistItemStart(String),
    /// Playlist item ended
    PlaylistItemEnd(String),
    /// System alert
    SystemAlert(String),
    /// Custom named event
    Custom(String),
}

impl TriggerEvent {
    /// Returns the name/label of this event.
    #[must_use]
    pub fn event_name(&self) -> &str {
        match self {
            Self::Timecode(_) => "Timecode",
            Self::GpiInput(_) => "GpiInput",
            Self::PlaylistItemStart(_) => "PlaylistItemStart",
            Self::PlaylistItemEnd(_) => "PlaylistItemEnd",
            Self::SystemAlert(_) => "SystemAlert",
            Self::Custom(name) => name.as_str(),
        }
    }

    /// Returns `true` if this is a system-generated (non-custom) event.
    #[must_use]
    pub fn is_system_event(&self) -> bool {
        !matches!(self, Self::Custom(_))
    }
}

/// Condition that must be satisfied for a trigger to fire.
#[derive(Debug, Clone)]
pub struct TriggerCondition {
    /// Required event name prefix (empty means any event matches)
    pub event_prefix: String,
    /// Minimum numeric severity level (0 = always matches)
    pub min_level: u32,
    /// Optional label that must be present in the event payload
    pub required_label: Option<String>,
}

impl TriggerCondition {
    /// Creates a condition that matches any event.
    #[must_use]
    pub fn any() -> Self {
        Self {
            event_prefix: String::new(),
            min_level: 0,
            required_label: None,
        }
    }

    /// Creates a condition that matches a specific event name prefix.
    #[must_use]
    pub fn for_prefix(prefix: &str) -> Self {
        Self {
            event_prefix: prefix.to_owned(),
            min_level: 0,
            required_label: None,
        }
    }

    /// Evaluates whether this condition is satisfied by the given event.
    #[must_use]
    pub fn evaluate(&self, event: &TriggerEvent, level: u32) -> bool {
        if level < self.min_level {
            return false;
        }
        let name = event.event_name();
        if !self.event_prefix.is_empty() && !name.starts_with(&self.event_prefix) {
            return false;
        }
        if let Some(label) = &self.required_label {
            let payload = match event {
                TriggerEvent::Timecode(v)
                | TriggerEvent::PlaylistItemStart(v)
                | TriggerEvent::PlaylistItemEnd(v)
                | TriggerEvent::SystemAlert(v)
                | TriggerEvent::Custom(v) => v.as_str(),
                TriggerEvent::GpiInput(_) => "",
            };
            if !payload.contains(label.as_str()) {
                return false;
            }
        }
        true
    }
}

/// A named trigger rule binding a condition to an action label.
#[derive(Debug, Clone)]
pub struct TriggerRule {
    /// Unique rule identifier
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Condition that must be met
    pub condition: TriggerCondition,
    /// Label of the action to invoke when the rule fires
    pub action: String,
    /// Whether this rule is currently enabled
    pub enabled: bool,
}

impl TriggerRule {
    /// Creates a new enabled trigger rule.
    #[must_use]
    pub fn new(id: &str, description: &str, condition: TriggerCondition, action: &str) -> Self {
        Self {
            id: id.to_owned(),
            description: description.to_owned(),
            condition,
            action: action.to_owned(),
            enabled: true,
        }
    }

    /// Returns `true` if this rule matches the given event at the given level.
    #[must_use]
    pub fn matches_event(&self, event: &TriggerEvent, level: u32) -> bool {
        self.enabled && self.condition.evaluate(event, level)
    }
}

/// Registry that stores trigger rules and fires matching ones.
#[derive(Debug, Default)]
pub struct TriggerRegistry {
    rules: HashMap<String, TriggerRule>,
}

impl TriggerRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a trigger rule.
    /// If a rule with the same id already exists it is replaced.
    pub fn register(&mut self, rule: TriggerRule) {
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Removes a rule by id.  Returns `true` if a rule was removed.
    pub fn unregister(&mut self, id: &str) -> bool {
        self.rules.remove(id).is_some()
    }

    /// Fires an event with a given severity level and returns the action labels
    /// of every matching rule.
    #[must_use]
    pub fn fire(&self, event: &TriggerEvent, level: u32) -> Vec<String> {
        let mut actions: Vec<String> = self
            .rules
            .values()
            .filter(|r| r.matches_event(event, level))
            .map(|r| r.action.clone())
            .collect();
        actions.sort(); // deterministic order for tests
        actions
    }

    /// Returns the number of registered rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn gpi_event() -> TriggerEvent {
        TriggerEvent::GpiInput(3)
    }

    fn custom_event(name: &str) -> TriggerEvent {
        TriggerEvent::Custom(name.to_owned())
    }

    #[test]
    fn test_event_name_timecode() {
        let e = TriggerEvent::Timecode("10:00:00:00".to_owned());
        assert_eq!(e.event_name(), "Timecode");
    }

    #[test]
    fn test_event_name_gpi() {
        assert_eq!(gpi_event().event_name(), "GpiInput");
    }

    #[test]
    fn test_event_name_custom() {
        assert_eq!(custom_event("MyEvent").event_name(), "MyEvent");
    }

    #[test]
    fn test_is_system_event_true() {
        assert!(TriggerEvent::SystemAlert("x".to_owned()).is_system_event());
    }

    #[test]
    fn test_is_system_event_false_for_custom() {
        assert!(!custom_event("X").is_system_event());
    }

    #[test]
    fn test_condition_any_matches_all() {
        let cond = TriggerCondition::any();
        assert!(cond.evaluate(&gpi_event(), 0));
        assert!(cond.evaluate(&custom_event("Z"), 99));
    }

    #[test]
    fn test_condition_prefix_match() {
        let cond = TriggerCondition::for_prefix("Gpi");
        assert!(cond.evaluate(&gpi_event(), 0));
        assert!(!cond.evaluate(&custom_event("Other"), 0));
    }

    #[test]
    fn test_condition_min_level() {
        let mut cond = TriggerCondition::any();
        cond.min_level = 5;
        assert!(!cond.evaluate(&gpi_event(), 4));
        assert!(cond.evaluate(&gpi_event(), 5));
    }

    #[test]
    fn test_condition_required_label() {
        let mut cond = TriggerCondition::any();
        cond.required_label = Some("important".to_owned());
        let e = TriggerEvent::SystemAlert("this is important".to_owned());
        assert!(cond.evaluate(&e, 0));
        let e2 = TriggerEvent::SystemAlert("routine".to_owned());
        assert!(!cond.evaluate(&e2, 0));
    }

    #[test]
    fn test_trigger_rule_matches_when_enabled() {
        let rule = TriggerRule::new("r1", "desc", TriggerCondition::any(), "do_something");
        assert!(rule.matches_event(&gpi_event(), 0));
    }

    #[test]
    fn test_trigger_rule_disabled() {
        let mut rule = TriggerRule::new("r1", "desc", TriggerCondition::any(), "act");
        rule.enabled = false;
        assert!(!rule.matches_event(&gpi_event(), 0));
    }

    #[test]
    fn test_registry_register_and_count() {
        let mut reg = TriggerRegistry::new();
        reg.register(TriggerRule::new("r1", "", TriggerCondition::any(), "a"));
        assert_eq!(reg.rule_count(), 1);
    }

    #[test]
    fn test_registry_fire_returns_matching_actions() {
        let mut reg = TriggerRegistry::new();
        reg.register(TriggerRule::new(
            "r1",
            "",
            TriggerCondition::for_prefix("Gpi"),
            "gpi_action",
        ));
        reg.register(TriggerRule::new(
            "r2",
            "",
            TriggerCondition::any(),
            "any_action",
        ));
        let actions = reg.fire(&gpi_event(), 0);
        assert!(actions.contains(&"gpi_action".to_owned()));
        assert!(actions.contains(&"any_action".to_owned()));
    }

    #[test]
    fn test_registry_fire_no_match() {
        let mut reg = TriggerRegistry::new();
        reg.register(TriggerRule::new(
            "r1",
            "",
            TriggerCondition::for_prefix("Timecode"),
            "tc_action",
        ));
        let actions = reg.fire(&gpi_event(), 0);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = TriggerRegistry::new();
        reg.register(TriggerRule::new("r1", "", TriggerCondition::any(), "a"));
        assert!(reg.unregister("r1"));
        assert_eq!(reg.rule_count(), 0);
        assert!(!reg.unregister("r1")); // second call returns false
    }
}
