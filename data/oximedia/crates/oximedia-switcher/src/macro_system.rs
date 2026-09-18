//! Macro/preset recall system for production switcher.
//!
//! Provides structured macro management with banks of macros, each containing
//! ordered steps with optional delays between them.

#![allow(dead_code)]

/// A single step within a macro.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroStep {
    /// The action to execute (e.g. "set_program", "cut", "dissolve")
    pub action: String,
    /// Parameter key for the action
    pub param_key: String,
    /// Parameter value for the action
    pub param_value: String,
    /// Delay in milliseconds after this step executes (0 = no delay)
    pub delay_ms: u32,
}

impl MacroStep {
    /// Create a new macro step.
    pub fn new(
        action: impl Into<String>,
        param_key: impl Into<String>,
        param_value: impl Into<String>,
        delay_ms: u32,
    ) -> Self {
        Self {
            action: action.into(),
            param_key: param_key.into(),
            param_value: param_value.into(),
            delay_ms,
        }
    }

    /// Returns true if this step has a non-zero delay.
    pub fn has_delay(&self) -> bool {
        self.delay_ms > 0
    }
}

/// A macro consisting of one or more ordered steps.
#[derive(Debug, Clone)]
pub struct Macro {
    /// Unique identifier for this macro
    pub id: u32,
    /// Human-readable name
    pub name: String,
    /// Ordered list of steps
    pub steps: Vec<MacroStep>,
}

impl Macro {
    /// Create a new empty macro.
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            steps: Vec::new(),
        }
    }

    /// Add a step to this macro.
    pub fn add_step(&mut self, step: MacroStep) {
        self.steps.push(step);
    }

    /// Return the number of steps in this macro.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return the total playback duration in milliseconds (sum of all step delays).
    pub fn total_duration_ms(&self) -> u32 {
        self.steps.iter().map(|s| s.delay_ms).sum()
    }

    /// Returns true if this macro has no delays (executes all steps in a single frame).
    pub fn is_instant(&self) -> bool {
        self.total_duration_ms() == 0
    }
}

/// A bank of macros available for recall.
#[derive(Debug, Clone, Default)]
pub struct MacroBank {
    /// The macros stored in this bank
    pub macros: Vec<Macro>,
}

impl MacroBank {
    /// Create an empty macro bank.
    pub fn new() -> Self {
        Self { macros: Vec::new() }
    }

    /// Add a macro to this bank.
    pub fn add(&mut self, m: Macro) {
        self.macros.push(m);
    }

    /// Find a macro by its numeric ID.
    pub fn find_by_id(&self, id: u32) -> Option<&Macro> {
        self.macros.iter().find(|m| m.id == id)
    }

    /// Find a macro by its name (first match).
    pub fn find_by_name(&self, name: &str) -> Option<&Macro> {
        self.macros.iter().find(|m| m.name == name)
    }

    /// Return the number of macros in this bank.
    pub fn macro_count(&self) -> usize {
        self.macros.len()
    }
}

// ---------------------------------------------------------------------------
// New types: MacroAction, SwitcherMacro, MacroPlayer, MacroLibrary
// ---------------------------------------------------------------------------

/// A single atomic action that can be stored in a `SwitcherMacro`.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum MacroAction {
    /// Perform an instant cut to the given input number.
    CutToInput(u32),
    /// Execute a named/stored transition by its ID.
    RunTransition(u32),
    /// Set a keyer on or off.
    SetKey(u32, bool),
    /// Wait for the given number of frames before proceeding.
    WaitFrames(u32),
    /// Trigger a GPI (general-purpose interface) output.
    SendGpi(u32),
}

impl MacroAction {
    /// Returns the estimated duration in frames for this action.
    ///
    /// Most actions are instantaneous (0 frames); `WaitFrames` returns its
    /// payload; `RunTransition` is assumed to take 1 frame to start.
    #[must_use]
    pub fn estimated_duration_frames(&self) -> u32 {
        match self {
            Self::WaitFrames(n) => *n,
            Self::RunTransition(_) => 1,
            _ => 0,
        }
    }
}

/// A named sequence of `MacroAction`s for the production switcher.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SwitcherMacro {
    /// Unique macro identifier.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Ordered list of actions.
    pub actions: Vec<MacroAction>,
}

impl SwitcherMacro {
    /// Create a new empty `SwitcherMacro`.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            actions: Vec::new(),
        }
    }

    /// Total duration of the macro in frames (sum of all action durations).
    #[must_use]
    pub fn total_duration_frames(&self) -> u32 {
        self.actions
            .iter()
            .map(MacroAction::estimated_duration_frames)
            .sum()
    }

    /// Number of actions in the macro.
    #[must_use]
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }
}

/// A player that steps through a loaded `SwitcherMacro` one frame at a time.
///
/// Internally tracks the current action index and counts down any `WaitFrames`
/// delay before advancing to the next action.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct MacroPlayer {
    /// The macro currently loaded (if any).
    pub current_macro: Option<SwitcherMacro>,
    /// Index of the next action to execute.
    pub action_idx: usize,
    /// Remaining frame delay before the next action may fire.
    pub frame_delay: u32,
}

impl MacroPlayer {
    /// Create a new idle `MacroPlayer`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load a macro, resetting the player to the beginning.
    pub fn load(&mut self, m: SwitcherMacro) {
        self.action_idx = 0;
        self.frame_delay = 0;
        self.current_macro = Some(m);
    }

    /// Advance by one frame.  Returns `Some(&MacroAction)` if an action fires
    /// this frame, or `None` if still waiting or if no macro is loaded.
    #[must_use]
    pub fn step(&mut self) -> Option<&MacroAction> {
        let m = self.current_macro.as_ref()?;
        if self.action_idx >= m.actions.len() {
            return None;
        }

        // Count down any active delay first.
        if self.frame_delay > 0 {
            self.frame_delay -= 1;
            return None;
        }

        let action = &m.actions[self.action_idx];

        // If this action introduces a delay, set it and stay on the same action.
        if let MacroAction::WaitFrames(n) = action {
            if *n > 0 {
                self.frame_delay = n - 1; // consume this frame as one tick
                self.action_idx += 1;
                // Return the WaitFrames action so the caller sees it was issued.
                return self
                    .current_macro
                    .as_ref()
                    .and_then(|m| m.actions.get(self.action_idx - 1));
            }
        }

        self.action_idx += 1;
        self.current_macro
            .as_ref()
            .and_then(|m| m.actions.get(self.action_idx - 1))
    }

    /// Returns `true` if no macro is loaded or all actions have been executed.
    #[must_use]
    pub fn is_done(&self) -> bool {
        match &self.current_macro {
            None => true,
            Some(m) => self.action_idx >= m.actions.len() && self.frame_delay == 0,
        }
    }
}

/// A collection of named `SwitcherMacro`s with auto-incrementing IDs.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct MacroLibrary {
    /// Stored macros.
    pub macros: Vec<SwitcherMacro>,
    next_id: u32,
}

impl MacroLibrary {
    /// Create an empty `MacroLibrary`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a new macro and return its assigned ID.
    pub fn store(&mut self, name: impl Into<String>, actions: Vec<MacroAction>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.macros.push(SwitcherMacro {
            id,
            name: name.into(),
            actions,
        });
        id
    }

    /// Find a macro by ID.
    #[must_use]
    pub fn find(&self, id: u32) -> Option<&SwitcherMacro> {
        self.macros.iter().find(|m| m.id == id)
    }

    /// Return the number of stored macros.
    #[must_use]
    pub fn macro_count(&self) -> usize {
        self.macros.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_step(delay_ms: u32) -> MacroStep {
        MacroStep::new("set_program", "input", "1", delay_ms)
    }

    fn make_macro_with_steps(id: u32, name: &str, delays: &[u32]) -> Macro {
        let mut m = Macro::new(id, name);
        for &d in delays {
            m.add_step(make_step(d));
        }
        m
    }

    // --- MacroStep tests ---

    #[test]
    fn test_step_has_delay_nonzero() {
        let step = MacroStep::new("cut", "me", "0", 500);
        assert!(step.has_delay());
    }

    #[test]
    fn test_step_has_delay_zero() {
        let step = MacroStep::new("cut", "me", "0", 0);
        assert!(!step.has_delay());
    }

    #[test]
    fn test_step_fields_stored_correctly() {
        let step = MacroStep::new("dissolve", "rate", "25", 40);
        assert_eq!(step.action, "dissolve");
        assert_eq!(step.param_key, "rate");
        assert_eq!(step.param_value, "25");
        assert_eq!(step.delay_ms, 40);
    }

    // --- Macro tests ---

    #[test]
    fn test_macro_step_count_empty() {
        let m = Macro::new(1, "Empty");
        assert_eq!(m.step_count(), 0);
    }

    #[test]
    fn test_macro_step_count_after_add() {
        let m = make_macro_with_steps(1, "Test", &[0, 100, 200]);
        assert_eq!(m.step_count(), 3);
    }

    #[test]
    fn test_macro_total_duration_ms() {
        let m = make_macro_with_steps(2, "Timed", &[100, 200, 300]);
        assert_eq!(m.total_duration_ms(), 600);
    }

    #[test]
    fn test_macro_total_duration_ms_zero() {
        let m = make_macro_with_steps(3, "Instant", &[0, 0]);
        assert_eq!(m.total_duration_ms(), 0);
    }

    #[test]
    fn test_macro_is_instant_true() {
        let m = make_macro_with_steps(4, "Instant", &[0, 0, 0]);
        assert!(m.is_instant());
    }

    #[test]
    fn test_macro_is_instant_false() {
        let m = make_macro_with_steps(5, "Delayed", &[0, 50]);
        assert!(!m.is_instant());
    }

    #[test]
    fn test_macro_empty_is_instant() {
        let m = Macro::new(6, "Empty");
        assert!(m.is_instant());
    }

    // --- MacroBank tests ---

    #[test]
    fn test_bank_macro_count_empty() {
        let bank = MacroBank::new();
        assert_eq!(bank.macro_count(), 0);
    }

    #[test]
    fn test_bank_add_increases_count() {
        let mut bank = MacroBank::new();
        bank.add(Macro::new(1, "A"));
        bank.add(Macro::new(2, "B"));
        assert_eq!(bank.macro_count(), 2);
    }

    #[test]
    fn test_bank_find_by_id_found() {
        let mut bank = MacroBank::new();
        bank.add(Macro::new(42, "Answer"));
        let found = bank.find_by_id(42);
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").name, "Answer");
    }

    #[test]
    fn test_bank_find_by_id_not_found() {
        let bank = MacroBank::new();
        assert!(bank.find_by_id(99).is_none());
    }

    #[test]
    fn test_bank_find_by_name_found() {
        let mut bank = MacroBank::new();
        bank.add(Macro::new(1, "OpeningSequence"));
        let found = bank.find_by_name("OpeningSequence");
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").id, 1);
    }

    #[test]
    fn test_bank_find_by_name_not_found() {
        let mut bank = MacroBank::new();
        bank.add(Macro::new(1, "Alpha"));
        assert!(bank.find_by_name("Beta").is_none());
    }

    // --- MacroAction tests ---

    #[test]
    fn test_macro_action_cut_to_input_duration_zero() {
        assert_eq!(MacroAction::CutToInput(1).estimated_duration_frames(), 0);
    }

    #[test]
    fn test_macro_action_run_transition_duration_one() {
        assert_eq!(MacroAction::RunTransition(5).estimated_duration_frames(), 1);
    }

    #[test]
    fn test_macro_action_wait_frames_duration() {
        assert_eq!(MacroAction::WaitFrames(25).estimated_duration_frames(), 25);
    }

    #[test]
    fn test_macro_action_set_key_duration_zero() {
        assert_eq!(MacroAction::SetKey(0, true).estimated_duration_frames(), 0);
    }

    #[test]
    fn test_macro_action_send_gpi_duration_zero() {
        assert_eq!(MacroAction::SendGpi(2).estimated_duration_frames(), 0);
    }

    // --- SwitcherMacro tests ---

    #[test]
    fn test_switcher_macro_total_duration_empty() {
        let m = SwitcherMacro::new(0, "Empty");
        assert_eq!(m.total_duration_frames(), 0);
    }

    #[test]
    fn test_switcher_macro_total_duration_with_wait() {
        let mut m = SwitcherMacro::new(0, "M");
        m.actions.push(MacroAction::CutToInput(1));
        m.actions.push(MacroAction::WaitFrames(10));
        m.actions.push(MacroAction::CutToInput(2));
        assert_eq!(m.total_duration_frames(), 10);
    }

    #[test]
    fn test_switcher_macro_action_count() {
        let mut m = SwitcherMacro::new(0, "M");
        m.actions.push(MacroAction::CutToInput(1));
        m.actions.push(MacroAction::SendGpi(0));
        assert_eq!(m.action_count(), 2);
    }

    // --- MacroPlayer tests ---

    #[test]
    fn test_macro_player_is_done_when_empty() {
        let player = MacroPlayer::new();
        assert!(player.is_done());
    }

    #[test]
    fn test_macro_player_step_empty_macro_returns_none() {
        let mut player = MacroPlayer::new();
        player.load(SwitcherMacro::new(0, "Empty"));
        assert!(player.step().is_none());
    }

    #[test]
    fn test_macro_player_step_fires_action() {
        let mut m = SwitcherMacro::new(0, "M");
        m.actions.push(MacroAction::CutToInput(3));
        let mut player = MacroPlayer::new();
        player.load(m);
        let action = player.step();
        assert!(action.is_some());
        assert_eq!(
            *action.expect("should succeed in test"),
            MacroAction::CutToInput(3)
        );
    }

    #[test]
    fn test_macro_player_is_done_after_all_actions() {
        let mut m = SwitcherMacro::new(0, "M");
        m.actions.push(MacroAction::CutToInput(1));
        let mut player = MacroPlayer::new();
        player.load(m);
        let _ = player.step();
        assert!(player.is_done());
    }

    // --- MacroLibrary tests ---

    #[test]
    fn test_macro_library_empty() {
        let lib = MacroLibrary::new();
        assert_eq!(lib.macro_count(), 0);
    }

    #[test]
    fn test_macro_library_store_and_find() {
        let mut lib = MacroLibrary::new();
        let id = lib.store("Opening", vec![MacroAction::CutToInput(1)]);
        assert_eq!(lib.macro_count(), 1);
        let found = lib.find(id);
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").name, "Opening");
    }

    #[test]
    fn test_macro_library_find_not_found() {
        let lib = MacroLibrary::new();
        assert!(lib.find(99).is_none());
    }

    #[test]
    fn test_macro_library_ids_auto_increment() {
        let mut lib = MacroLibrary::new();
        let id0 = lib.store("A", vec![]);
        let id1 = lib.store("B", vec![]);
        assert_eq!(id1, id0 + 1);
    }
}
