// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Finite state machine for workflow execution state tracking.
//!
//! Provides a simple but complete FSM implementation with named states,
//! guarded transitions, and terminal-state detection.

/// A single state in the FSM.
#[derive(Debug, Clone)]
pub struct WorkflowState {
    /// Unique identifier for this state.
    pub id: String,
    /// Human-readable label.
    pub name: String,
    /// If `true` the FSM stops accepting transitions once in this state.
    pub is_terminal: bool,
    /// Optional action identifier to invoke on entry.
    pub on_enter: Option<String>,
    /// Optional action identifier to invoke on exit.
    pub on_exit: Option<String>,
}

impl WorkflowState {
    /// Create a new state.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            is_terminal: false,
            on_enter: None,
            on_exit: None,
        }
    }

    /// Mark this state as terminal (no outgoing transitions accepted).
    #[must_use]
    pub fn terminal(mut self) -> Self {
        self.is_terminal = true;
        self
    }

    /// Set the on-enter action identifier.
    #[must_use]
    pub fn on_enter(mut self, action: impl Into<String>) -> Self {
        self.on_enter = Some(action.into());
        self
    }

    /// Set the on-exit action identifier.
    #[must_use]
    pub fn on_exit(mut self, action: impl Into<String>) -> Self {
        self.on_exit = Some(action.into());
        self
    }

    /// Returns `true` when the state is a terminal state.
    #[must_use]
    pub fn is_final(&self) -> bool {
        self.is_terminal
    }
}

/// A directed transition between two states.
#[derive(Debug, Clone)]
pub struct WorkflowTransition {
    /// The source state ID.
    pub from: String,
    /// The destination state ID.
    pub to: String,
    /// Event name that triggers this transition.
    pub trigger: String,
    /// Optional guard expression (informational only in this implementation).
    pub guard: Option<String>,
}

impl WorkflowTransition {
    /// Create a new transition.
    #[must_use]
    pub fn new(from: impl Into<String>, to: impl Into<String>, trigger: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            trigger: trigger.into(),
            guard: None,
        }
    }

    /// Attach a guard expression to this transition.
    #[must_use]
    pub fn with_guard(mut self, guard: impl Into<String>) -> Self {
        self.guard = Some(guard.into());
        self
    }

    /// Returns `true` when `t` matches this transition's trigger.
    #[must_use]
    pub fn matches_trigger(&self, t: &str) -> bool {
        self.trigger == t
    }
}

/// A finite state machine composed of [`WorkflowState`]s and [`WorkflowTransition`]s.
#[derive(Debug, Clone)]
pub struct StateMachine {
    /// All registered states.
    pub states: Vec<WorkflowState>,
    /// All registered transitions.
    pub transitions: Vec<WorkflowTransition>,
    /// ID of the currently active state.
    pub current_state: String,
}

impl StateMachine {
    /// Create a new FSM with `initial_state_id` as the starting state.
    ///
    /// The initial state is NOT required to exist in `states` at construction
    /// time; it is added implicitly when the first state is added.
    #[must_use]
    pub fn new(initial_state_id: impl Into<String>) -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            current_state: initial_state_id.into(),
        }
    }

    /// Register a state.
    pub fn add_state(&mut self, state: WorkflowState) {
        self.states.push(state);
    }

    /// Register a transition.
    pub fn add_transition(&mut self, t: WorkflowTransition) {
        self.transitions.push(t);
    }

    /// Fire `event`.  Returns `true` if a transition was found and applied.
    ///
    /// A transition is eligible when:
    /// 1. Its `from` field matches `current_state`.
    /// 2. Its `trigger` matches `event`.
    /// 3. The current state is not terminal.
    pub fn trigger(&mut self, event: &str) -> bool {
        if self.is_terminal() {
            return false;
        }

        let transition = self
            .transitions
            .iter()
            .find(|t| t.from == self.current_state && t.matches_trigger(event));

        if let Some(t) = transition {
            let next = t.to.clone();
            self.current_state = next;
            true
        } else {
            false
        }
    }

    /// Return the ID of the current state.
    #[must_use]
    pub fn current(&self) -> &str {
        &self.current_state
    }

    /// Returns `true` when the current state is marked terminal.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.states
            .iter()
            .find(|s| s.id == self.current_state)
            .is_some_and(|s| s.is_terminal)
    }

    /// Return the event names of all transitions originating from the current state.
    #[must_use]
    pub fn valid_triggers(&self) -> Vec<&str> {
        self.transitions
            .iter()
            .filter(|t| t.from == self.current_state)
            .map(|t| t.trigger.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_fsm() -> StateMachine {
        let mut fsm = StateMachine::new("idle");
        fsm.add_state(WorkflowState::new("idle", "Idle"));
        fsm.add_state(WorkflowState::new("running", "Running"));
        fsm.add_state(WorkflowState::new("done", "Done").terminal());
        fsm.add_state(WorkflowState::new("failed", "Failed").terminal());

        fsm.add_transition(WorkflowTransition::new("idle", "running", "start"));
        fsm.add_transition(WorkflowTransition::new("running", "done", "complete"));
        fsm.add_transition(WorkflowTransition::new("running", "failed", "error"));
        fsm
    }

    #[test]
    fn test_initial_state() {
        let fsm = build_simple_fsm();
        assert_eq!(fsm.current(), "idle");
    }

    #[test]
    fn test_trigger_valid_event() {
        let mut fsm = build_simple_fsm();
        assert!(fsm.trigger("start"));
        assert_eq!(fsm.current(), "running");
    }

    #[test]
    fn test_trigger_unknown_event_returns_false() {
        let mut fsm = build_simple_fsm();
        assert!(!fsm.trigger("unknown"));
        assert_eq!(fsm.current(), "idle");
    }

    #[test]
    fn test_multi_step_transition() {
        let mut fsm = build_simple_fsm();
        fsm.trigger("start");
        fsm.trigger("complete");
        assert_eq!(fsm.current(), "done");
    }

    #[test]
    fn test_terminal_state_blocks_trigger() {
        let mut fsm = build_simple_fsm();
        fsm.trigger("start");
        fsm.trigger("complete");
        assert!(fsm.is_terminal());
        // further triggers should be ignored
        assert!(!fsm.trigger("start"));
        assert_eq!(fsm.current(), "done");
    }

    #[test]
    fn test_error_path() {
        let mut fsm = build_simple_fsm();
        fsm.trigger("start");
        assert!(fsm.trigger("error"));
        assert_eq!(fsm.current(), "failed");
        assert!(fsm.is_terminal());
    }

    #[test]
    fn test_is_terminal_false_on_nonterminal() {
        let fsm = build_simple_fsm();
        assert!(!fsm.is_terminal());
    }

    #[test]
    fn test_valid_triggers_from_idle() {
        let fsm = build_simple_fsm();
        let triggers = fsm.valid_triggers();
        assert_eq!(triggers, vec!["start"]);
    }

    #[test]
    fn test_valid_triggers_from_running() {
        let mut fsm = build_simple_fsm();
        fsm.trigger("start");
        let mut triggers = fsm.valid_triggers();
        triggers.sort_unstable();
        assert_eq!(triggers, vec!["complete", "error"]);
    }

    #[test]
    fn test_workflow_state_is_final() {
        let s = WorkflowState::new("end", "End").terminal();
        assert!(s.is_final());
        let s2 = WorkflowState::new("mid", "Mid");
        assert!(!s2.is_final());
    }

    #[test]
    fn test_workflow_state_on_enter_exit() {
        let s = WorkflowState::new("s", "S")
            .on_enter("log_entry")
            .on_exit("log_exit");
        assert_eq!(s.on_enter.as_deref(), Some("log_entry"));
        assert_eq!(s.on_exit.as_deref(), Some("log_exit"));
    }

    #[test]
    fn test_transition_matches_trigger() {
        let t = WorkflowTransition::new("a", "b", "go");
        assert!(t.matches_trigger("go"));
        assert!(!t.matches_trigger("stop"));
    }

    #[test]
    fn test_transition_with_guard() {
        let t = WorkflowTransition::new("a", "b", "go").with_guard("x > 0");
        assert_eq!(t.guard.as_deref(), Some("x > 0"));
    }

    #[test]
    fn test_valid_triggers_empty_when_terminal() {
        let mut fsm = build_simple_fsm();
        fsm.trigger("start");
        fsm.trigger("complete");
        // terminal state — no transitions defined from "done"
        assert!(fsm.valid_triggers().is_empty());
    }
}
