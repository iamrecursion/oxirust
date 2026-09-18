//! Macro recording and playback for broadcast automation.
//!
//! This module provides the ability to record sequences of operator actions
//! and play them back later, optionally at a different speed.

#![allow(dead_code)]

/// Types of actions that can be recorded in a macro.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroActionType {
    /// Simulate a button click
    Click,
    /// Set a control to a specific value
    SetValue,
    /// Wait for a fixed duration (specified in `delay_ms`)
    Wait,
    /// Play a media clip
    PlayClip,
    /// Stop a media clip
    StopClip,
    /// Switch the active input on a router / switcher
    SwitchInput,
    /// Load a graphics scene
    LoadScene,
}

/// A single recorded operator action.
#[derive(Debug, Clone)]
pub struct MacroAction {
    /// Type of action
    pub action_type: MacroActionType,
    /// Target element identifier (button ID, control path, etc.)
    pub target: String,
    /// Associated value (clip name, scene path, input label, etc.)
    pub value: String,
    /// Delay in milliseconds *before* executing this action
    pub delay_ms: u32,
}

impl MacroAction {
    /// Create a new `MacroAction`.
    pub fn new(
        action_type: MacroActionType,
        target: impl Into<String>,
        value: impl Into<String>,
        delay_ms: u32,
    ) -> Self {
        Self {
            action_type,
            target: target.into(),
            value: value.into(),
            delay_ms,
        }
    }
}

/// A named, immutable sequence of recorded actions.
#[derive(Debug, Clone)]
pub struct Macro {
    /// Human-readable name
    pub name: String,
    /// Ordered list of actions
    pub actions: Vec<MacroAction>,
    /// Sum of all `delay_ms` values (total macro duration in ms)
    pub total_duration_ms: u64,
}

impl Macro {
    /// Create a macro directly from a list of actions.
    pub fn from_actions(name: impl Into<String>, actions: Vec<MacroAction>) -> Self {
        let total_duration_ms = actions.iter().map(|a| u64::from(a.delay_ms)).sum();
        Self {
            name: name.into(),
            actions,
            total_duration_ms,
        }
    }
}

/// Stateful recorder that captures operator actions in real time.
#[derive(Debug, Default)]
pub struct MacroRecorder {
    recording: bool,
    pending_actions: Vec<MacroAction>,
}

impl MacroRecorder {
    /// Create a new recorder (initially not recording).
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new recording session.
    ///
    /// Any previously pending (unfinished) actions are discarded.
    pub fn start_recording(&mut self) {
        self.recording = true;
        self.pending_actions.clear();
    }

    /// Stop recording and return the captured [`Macro`].
    ///
    /// If recording was not active the returned macro will have no actions.
    pub fn stop_recording(&mut self, macro_name: impl Into<String>) -> Macro {
        self.recording = false;
        let actions = std::mem::take(&mut self.pending_actions);
        Macro::from_actions(macro_name, actions)
    }

    /// Record a single action.
    ///
    /// If the recorder is not currently active the action is silently dropped.
    pub fn record_action(&mut self, action: MacroAction) {
        if self.recording {
            self.pending_actions.push(action);
        }
    }

    /// Return `true` if a recording session is currently active.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Return the number of actions captured so far in the current session.
    pub fn pending_count(&self) -> usize {
        self.pending_actions.len()
    }
}

/// An action paired with its absolute playback timestamp.
#[derive(Debug, Clone)]
pub struct TimedAction {
    /// The action to execute
    pub action: MacroAction,
    /// Absolute time in milliseconds (from the start of the macro) at which
    /// to execute this action
    pub execute_at_ms: u64,
}

/// Playback engine for recorded macros.
pub struct MacroPlayer;

impl MacroPlayer {
    /// Expand a [`Macro`] into a timeline of [`TimedAction`]s at the given speed.
    ///
    /// `speed` is a positive multiplier:
    /// - `1.0` → original speed
    /// - `2.0` → twice as fast (all delays halved)
    /// - `0.5` → half speed (all delays doubled)
    ///
    /// The `execute_at_ms` for each action is the cumulative scaled delay up to
    /// (and including) that action.
    ///
    /// # Panics
    /// Panics if `speed` is not positive (≤ 0.0).
    #[must_use]
    pub fn play(m: &Macro, speed: f32) -> Vec<TimedAction> {
        assert!(speed > 0.0, "speed must be positive");

        let mut cursor_ms: u64 = 0;
        let mut result = Vec::with_capacity(m.actions.len());

        for action in &m.actions {
            let scaled_delay = (f64::from(action.delay_ms) / f64::from(speed)) as u64;
            cursor_ms += scaled_delay;
            result.push(TimedAction {
                action: action.clone(),
                execute_at_ms: cursor_ms,
            });
        }

        result
    }
}

/// Named library of reusable macros.
#[derive(Debug, Default)]
pub struct MacroLibrary {
    macros: Vec<Macro>,
}

impl MacroLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a macro to the library.
    pub fn add(&mut self, m: Macro) {
        self.macros.push(m);
    }

    /// Remove the first macro matching `name`; returns `true` if removed.
    pub fn remove(&mut self, name: &str) -> bool {
        if let Some(pos) = self.macros.iter().position(|m| m.name == name) {
            self.macros.remove(pos);
            true
        } else {
            false
        }
    }

    /// Find the first macro with the given name.
    pub fn find_by_name(&self, name: &str) -> Option<&Macro> {
        self.macros.iter().find(|m| m.name == name)
    }

    /// Return the number of macros in the library.
    pub fn count(&self) -> usize {
        self.macros.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_macro() -> Macro {
        let actions = vec![
            MacroAction::new(MacroActionType::Click, "btn_play", "", 0),
            MacroAction::new(MacroActionType::Wait, "", "", 500),
            MacroAction::new(MacroActionType::SetValue, "fader", "0.8", 100),
        ];
        Macro::from_actions("Test Macro", actions)
    }

    #[test]
    fn test_macro_from_actions_duration() {
        let m = make_macro();
        assert_eq!(m.total_duration_ms, 600); // 0 + 500 + 100
    }

    #[test]
    fn test_macro_recorder_start_stop() {
        let mut rec = MacroRecorder::new();
        assert!(!rec.is_recording());
        rec.start_recording();
        assert!(rec.is_recording());
        let m = rec.stop_recording("My Macro");
        assert!(!rec.is_recording());
        assert!(m.actions.is_empty());
    }

    #[test]
    fn test_macro_recorder_captures_actions() {
        let mut rec = MacroRecorder::new();
        rec.start_recording();
        rec.record_action(MacroAction::new(MacroActionType::Click, "btn", "", 0));
        rec.record_action(MacroAction::new(MacroActionType::Wait, "", "", 200));
        assert_eq!(rec.pending_count(), 2);
        let m = rec.stop_recording("Capture Test");
        assert_eq!(m.actions.len(), 2);
    }

    #[test]
    fn test_macro_recorder_drops_when_not_recording() {
        let mut rec = MacroRecorder::new();
        // Not started
        rec.record_action(MacroAction::new(MacroActionType::Click, "x", "", 0));
        assert_eq!(rec.pending_count(), 0);
    }

    #[test]
    fn test_macro_recorder_clears_on_restart() {
        let mut rec = MacroRecorder::new();
        rec.start_recording();
        rec.record_action(MacroAction::new(MacroActionType::Click, "x", "", 0));
        // Restart without stopping
        rec.start_recording();
        assert_eq!(rec.pending_count(), 0);
    }

    #[test]
    fn test_macro_player_1x_speed() {
        let m = make_macro();
        let timed = MacroPlayer::play(&m, 1.0);
        assert_eq!(timed.len(), 3);
        assert_eq!(timed[0].execute_at_ms, 0); // delay_ms = 0
        assert_eq!(timed[1].execute_at_ms, 500); // +500
        assert_eq!(timed[2].execute_at_ms, 600); // +100
    }

    #[test]
    fn test_macro_player_2x_speed() {
        let m = make_macro();
        let timed = MacroPlayer::play(&m, 2.0);
        assert_eq!(timed[1].execute_at_ms, 250); // 500 / 2
        assert_eq!(timed[2].execute_at_ms, 300); // 250 + 50
    }

    #[test]
    fn test_macro_player_half_speed() {
        let m = make_macro();
        let timed = MacroPlayer::play(&m, 0.5);
        assert_eq!(timed[1].execute_at_ms, 1000); // 500 / 0.5
        assert_eq!(timed[2].execute_at_ms, 1200); // 1000 + 200
    }

    #[test]
    fn test_macro_library_add_find() {
        let mut lib = MacroLibrary::new();
        lib.add(make_macro());
        assert_eq!(lib.count(), 1);
        assert!(lib.find_by_name("Test Macro").is_some());
        assert!(lib.find_by_name("Unknown").is_none());
    }

    #[test]
    fn test_macro_library_remove() {
        let mut lib = MacroLibrary::new();
        lib.add(make_macro());
        assert!(lib.remove("Test Macro"));
        assert_eq!(lib.count(), 0);
    }

    #[test]
    fn test_macro_library_remove_nonexistent() {
        let mut lib = MacroLibrary::new();
        assert!(!lib.remove("Ghost"));
    }
}
