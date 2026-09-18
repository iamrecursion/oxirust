//! High-level macro operations: action capture, playback, and parameter substitution.
//!
//! This module provides a higher-level macro system built on top of the
//! lower-level `script::macro_recorder` primitives.  It adds parameter
//! substitution (named variables in action payloads) and a structured
//! playback engine.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Parameter substitution
// ---------------------------------------------------------------------------

/// A parameter store used during macro playback.
#[derive(Debug, Clone, Default)]
pub struct ParamStore {
    values: HashMap<String, String>,
}

impl ParamStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.values.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    /// Substitute `{key}` placeholders in `template` with stored values.
    pub fn substitute(&self, template: &str) -> String {
        let mut result = template.to_owned();
        for (k, v) in &self.values {
            result = result.replace(&format!("{{{k}}}"), v);
        }
        result
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Macro action
// ---------------------------------------------------------------------------

/// The payload of a single macro step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionPayload {
    /// Set a named parameter to a value (may contain `{placeholders}`).
    SetParam { key: String, value: String },
    /// Execute a named command with optional argument.
    Command { name: String, arg: Option<String> },
    /// Pause playback for a given duration.
    Wait { millis: u64 },
    /// Jump to a named label within the macro.
    Jump { label: String },
    /// Mark a position in the macro for jumps.
    Label { name: String },
}

/// A single step in a recorded macro.
#[derive(Debug, Clone)]
pub struct MacroStep {
    /// Human-readable description.
    pub description: String,
    /// The action to perform.
    pub payload: ActionPayload,
}

impl MacroStep {
    pub fn set_param(key: impl Into<String>, value: impl Into<String>) -> Self {
        let key_str: String = key.into();
        let value_str: String = value.into();
        Self {
            description: format!("set {key_str}"),
            payload: ActionPayload::SetParam {
                key: key_str,
                value: value_str,
            },
        }
    }

    pub fn command(name: impl Into<String>, arg: Option<String>) -> Self {
        let n: String = name.into();
        Self {
            description: format!("cmd:{n}"),
            payload: ActionPayload::Command { name: n, arg },
        }
    }

    pub fn wait(millis: u64) -> Self {
        Self {
            description: format!("wait {millis}ms"),
            payload: ActionPayload::Wait { millis },
        }
    }

    pub fn label(name: impl Into<String>) -> Self {
        let n: String = name.into();
        Self {
            description: format!("label:{n}"),
            payload: ActionPayload::Label { name: n },
        }
    }

    pub fn jump(label: impl Into<String>) -> Self {
        let l: String = label.into();
        Self {
            description: format!("jump:{l}"),
            payload: ActionPayload::Jump { label: l },
        }
    }
}

// ---------------------------------------------------------------------------
// Macro definition
// ---------------------------------------------------------------------------

/// A recorded macro: an ordered list of steps.
#[derive(Debug, Clone, Default)]
pub struct AutoMacro {
    pub name: String,
    pub steps: Vec<MacroStep>,
}

impl AutoMacro {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
        }
    }

    pub fn push(&mut self, step: MacroStep) {
        self.steps.push(step);
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Macro recorder
// ---------------------------------------------------------------------------

/// Records operator actions into an `AutoMacro`.
#[derive(Debug, Default)]
pub struct MacroCapture {
    pub current: AutoMacro,
    pub recording: bool,
}

impl MacroCapture {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            current: AutoMacro::new(name),
            recording: false,
        }
    }

    pub fn start(&mut self) {
        self.recording = true;
    }

    pub fn stop(&mut self) -> AutoMacro {
        self.recording = false;
        std::mem::replace(&mut self.current, AutoMacro::new("untitled"))
    }

    pub fn record(&mut self, step: MacroStep) -> bool {
        if !self.recording {
            return false;
        }
        self.current.push(step);
        true
    }

    pub fn step_count(&self) -> usize {
        self.current.len()
    }
}

// ---------------------------------------------------------------------------
// Macro playback engine
// ---------------------------------------------------------------------------

/// Result of executing a single macro step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepResult {
    /// Step completed; advance to the next.
    Continue,
    /// Jump to the given label index.
    JumpTo(String),
    /// Pause execution for the given duration.
    Pause(Duration),
    /// Macro playback is complete.
    Done,
}

/// Synchronous macro playback engine.
#[derive(Debug)]
pub struct MacroPlayer {
    pub params: ParamStore,
    pub executed_commands: Vec<String>,
}

impl MacroPlayer {
    pub fn new() -> Self {
        Self {
            params: ParamStore::new(),
            executed_commands: Vec::new(),
        }
    }

    /// Execute a single step, returning what the playback loop should do next.
    pub fn execute_step(&mut self, step: &MacroStep) -> StepResult {
        match &step.payload {
            ActionPayload::SetParam { key, value } => {
                let resolved = self.params.substitute(value);
                self.params.set(key.clone(), resolved);
                StepResult::Continue
            }
            ActionPayload::Command { name, arg } => {
                let resolved_name = self.params.substitute(name);
                let resolved_arg = arg.as_deref().map(|a| self.params.substitute(a));
                let cmd = if let Some(a) = resolved_arg {
                    format!("{resolved_name}({a})")
                } else {
                    resolved_name
                };
                self.executed_commands.push(cmd);
                StepResult::Continue
            }
            ActionPayload::Wait { millis } => StepResult::Pause(Duration::from_millis(*millis)),
            ActionPayload::Jump { label } => StepResult::JumpTo(label.clone()),
            ActionPayload::Label { .. } => StepResult::Continue,
        }
    }

    /// Run an entire macro synchronously (skipping actual waits).
    pub fn run_dry(&mut self, m: &AutoMacro) -> usize {
        let mut pc = 0usize;
        let mut iterations = 0usize;
        let max_iterations = m.steps.len() * 100 + 1000;
        while pc < m.steps.len() && iterations < max_iterations {
            let result = self.execute_step(&m.steps[pc]);
            iterations += 1;
            match result {
                StepResult::Continue | StepResult::Pause(_) => pc += 1,
                StepResult::Done => break,
                StepResult::JumpTo(label) => {
                    // Find the label step
                    if let Some(idx) = m.steps.iter().position(
                        |s| matches!(&s.payload, ActionPayload::Label { name } if name == &label),
                    ) {
                        pc = idx + 1;
                    } else {
                        pc += 1;
                    }
                }
            }
        }
        self.executed_commands.len()
    }
}

impl Default for MacroPlayer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_store_set_and_get() {
        let mut store = ParamStore::new();
        store.set("channel", "1");
        assert_eq!(store.get("channel"), Some("1"));
    }

    #[test]
    fn test_param_store_substitute_single() {
        let mut store = ParamStore::new();
        store.set("ch", "CH1");
        let out = store.substitute("Switch to {ch}");
        assert_eq!(out, "Switch to CH1");
    }

    #[test]
    fn test_param_store_substitute_multiple() {
        let mut store = ParamStore::new();
        store.set("src", "CAM1");
        store.set("dst", "OUT2");
        let out = store.substitute("{src} → {dst}");
        assert_eq!(out, "CAM1 → OUT2");
    }

    #[test]
    fn test_param_store_missing_key_unchanged() {
        let store = ParamStore::new();
        let out = store.substitute("{missing}");
        assert_eq!(out, "{missing}");
    }

    #[test]
    fn test_param_store_len() {
        let mut store = ParamStore::new();
        assert!(store.is_empty());
        store.set("k", "v");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_macro_step_set_param() {
        let step = MacroStep::set_param("gain", "0.8");
        assert!(matches!(&step.payload, ActionPayload::SetParam { key, .. } if key == "gain"));
    }

    #[test]
    fn test_macro_step_command_with_arg() {
        let step = MacroStep::command("play", Some("clip1".to_string()));
        assert!(
            matches!(&step.payload, ActionPayload::Command { name, arg: Some(_) } if name == "play")
        );
    }

    #[test]
    fn test_macro_step_wait() {
        let step = MacroStep::wait(500);
        assert_eq!(step.payload, ActionPayload::Wait { millis: 500 });
    }

    #[test]
    fn test_macro_capture_records_only_when_started() {
        let mut cap = MacroCapture::new("test");
        let recorded = cap.record(MacroStep::wait(10));
        assert!(!recorded);
        cap.start();
        let recorded = cap.record(MacroStep::wait(10));
        assert!(recorded);
        assert_eq!(cap.step_count(), 1);
    }

    #[test]
    fn test_macro_capture_stop_returns_macro() {
        let mut cap = MacroCapture::new("my_macro");
        cap.start();
        cap.record(MacroStep::command("cut", None));
        let m = cap.stop();
        assert_eq!(m.name, "my_macro");
        assert_eq!(m.len(), 1);
        assert!(!cap.recording);
    }

    #[test]
    fn test_player_execute_set_param() {
        let mut player = MacroPlayer::new();
        let step = MacroStep::set_param("x", "42");
        let r = player.execute_step(&step);
        assert_eq!(r, StepResult::Continue);
        assert_eq!(player.params.get("x"), Some("42"));
    }

    #[test]
    fn test_player_execute_command() {
        let mut player = MacroPlayer::new();
        let step = MacroStep::command("fade", Some("1s".to_string()));
        let r = player.execute_step(&step);
        assert_eq!(r, StepResult::Continue);
        assert_eq!(player.executed_commands.len(), 1);
        assert!(player.executed_commands[0].contains("fade"));
    }

    #[test]
    fn test_player_execute_wait() {
        let mut player = MacroPlayer::new();
        let step = MacroStep::wait(100);
        let r = player.execute_step(&step);
        assert_eq!(r, StepResult::Pause(Duration::from_millis(100)));
    }

    #[test]
    fn test_player_run_dry_simple_macro() {
        let mut m = AutoMacro::new("simple");
        m.push(MacroStep::command("cut", None));
        m.push(MacroStep::command("fade", None));
        let mut player = MacroPlayer::new();
        let count = player.run_dry(&m);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_auto_macro_is_empty() {
        let m = AutoMacro::new("empty");
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }
}
