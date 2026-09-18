//! Switcher macro execution engine.
//!
//! Provides macro banks, button assignments, conditional step execution,
//! and macro scheduling for professional live production switchers.

#![allow(dead_code)]

use std::collections::HashMap;

/// Error types for macro execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroExecError {
    /// Bank index is out of range.
    InvalidBank(usize),
    /// Button index is out of range.
    InvalidButton(usize),
    /// No macro assigned to this bank/button combination.
    NotAssigned { bank: usize, button: usize },
    /// Macro is already executing.
    AlreadyRunning,
    /// Condition evaluation failed.
    ConditionError(String),
}

impl std::fmt::Display for MacroExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBank(b) => write!(f, "invalid bank: {b}"),
            Self::InvalidButton(b) => write!(f, "invalid button: {b}"),
            Self::NotAssigned { bank, button } => {
                write!(f, "no macro at bank {bank} button {button}")
            }
            Self::AlreadyRunning => write!(f, "macro is already running"),
            Self::ConditionError(e) => write!(f, "condition error: {e}"),
        }
    }
}

/// A primitive switcher command in a macro step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepCommand {
    /// Set the program source on a given M/E row.
    SetProgram { me_row: usize, source: usize },
    /// Set the preview source on a given M/E row.
    SetPreview { me_row: usize, source: usize },
    /// Perform a cut on a given M/E row.
    Cut { me_row: usize },
    /// Trigger an auto-transition on a given M/E row.
    Auto { me_row: usize },
    /// Set an aux output to a source.
    SetAux { aux: usize, source: usize },
    /// Enable or disable a keyer.
    SetKeyer { keyer: usize, on_air: bool },
    /// Wait for a number of frames before continuing.
    WaitFrames { frames: u32 },
    /// Pause macro execution until an external resume call.
    Pause,
    /// Jump to a step by index.
    Jump { step: usize },
}

/// A condition under which a step is executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepCondition {
    /// Step always executes.
    Always,
    /// Step executes only if the named variable equals the given value.
    VarEquals { var: String, value: String },
    /// Step executes only if the named variable does not equal the given value.
    VarNotEquals { var: String, value: String },
}

impl StepCondition {
    /// Evaluate the condition against a variable store.
    #[must_use]
    pub fn evaluate(&self, vars: &HashMap<String, String>) -> bool {
        match self {
            Self::Always => true,
            Self::VarEquals { var, value } => vars.get(var) == Some(value),
            Self::VarNotEquals { var, value } => vars.get(var) != Some(value),
        }
    }
}

/// A single step within a macro.
#[derive(Debug, Clone)]
pub struct MacroStep {
    /// Index of this step within the macro.
    pub index: usize,
    /// The command to execute.
    pub command: StepCommand,
    /// Condition that must be met to execute this step.
    pub condition: StepCondition,
    /// Human-readable description.
    pub description: String,
}

impl MacroStep {
    /// Create a new unconditional step.
    #[must_use]
    pub fn new(index: usize, command: StepCommand) -> Self {
        Self {
            index,
            command,
            condition: StepCondition::Always,
            description: String::new(),
        }
    }

    /// Create a conditional step.
    #[must_use]
    pub fn conditional(index: usize, command: StepCommand, condition: StepCondition) -> Self {
        Self {
            index,
            command,
            condition,
            description: String::new(),
        }
    }

    /// Check whether this step should execute.
    #[must_use]
    pub fn should_execute(&self, vars: &HashMap<String, String>) -> bool {
        self.condition.evaluate(vars)
    }
}

/// A named macro consisting of an ordered list of steps.
#[derive(Debug, Clone)]
pub struct SwitcherMacro {
    /// Macro identifier.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Ordered steps.
    pub steps: Vec<MacroStep>,
    /// Whether this macro is currently looping.
    pub looping: bool,
}

impl SwitcherMacro {
    /// Create a new macro.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            steps: Vec::new(),
            looping: false,
        }
    }

    /// Add a step.
    pub fn add_step(&mut self, step: MacroStep) {
        self.steps.push(step);
    }

    /// Total number of steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// Assignment of a macro to a bank/button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ButtonAssignment {
    /// Bank index (0-based).
    pub bank: usize,
    /// Button index within the bank (0-based).
    pub button: usize,
}

impl ButtonAssignment {
    /// Create a new button assignment.
    #[must_use]
    pub fn new(bank: usize, button: usize) -> Self {
        Self { bank, button }
    }
}

/// State of the macro executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecState {
    /// Not running.
    Idle,
    /// Running a macro.
    Running,
    /// Paused mid-macro.
    Paused,
}

/// Macro bank and execution engine.
#[derive(Debug)]
pub struct MacroBankExec {
    /// Number of banks.
    pub num_banks: usize,
    /// Number of buttons per bank.
    pub buttons_per_bank: usize,
    /// Macros stored in the engine.
    macros: HashMap<u32, SwitcherMacro>,
    /// Button-to-macro assignment map.
    assignments: HashMap<ButtonAssignment, u32>,
    /// Current execution state.
    pub state: ExecState,
    /// Currently executing macro ID.
    pub running_macro: Option<u32>,
    /// Current step index within the executing macro.
    pub current_step: usize,
    /// Runtime variables for conditional execution.
    pub variables: HashMap<String, String>,
}

impl MacroBankExec {
    /// Create a new macro bank executor.
    #[must_use]
    pub fn new(num_banks: usize, buttons_per_bank: usize) -> Self {
        Self {
            num_banks,
            buttons_per_bank,
            macros: HashMap::new(),
            assignments: HashMap::new(),
            state: ExecState::Idle,
            running_macro: None,
            current_step: 0,
            variables: HashMap::new(),
        }
    }

    /// Store a macro.
    pub fn store_macro(&mut self, m: SwitcherMacro) {
        self.macros.insert(m.id, m);
    }

    /// Assign a macro to a bank/button.
    pub fn assign(
        &mut self,
        bank: usize,
        button: usize,
        macro_id: u32,
    ) -> Result<(), MacroExecError> {
        if bank >= self.num_banks {
            return Err(MacroExecError::InvalidBank(bank));
        }
        if button >= self.buttons_per_bank {
            return Err(MacroExecError::InvalidButton(button));
        }
        self.assignments
            .insert(ButtonAssignment::new(bank, button), macro_id);
        Ok(())
    }

    /// Get the macro ID assigned to a bank/button.
    #[must_use]
    pub fn assigned_macro_id(&self, bank: usize, button: usize) -> Option<u32> {
        self.assignments
            .get(&ButtonAssignment::new(bank, button))
            .copied()
    }

    /// Trigger execution of the macro at a bank/button.
    pub fn trigger(&mut self, bank: usize, button: usize) -> Result<u32, MacroExecError> {
        if self.state == ExecState::Running {
            return Err(MacroExecError::AlreadyRunning);
        }
        let macro_id = self
            .assignments
            .get(&ButtonAssignment::new(bank, button))
            .copied()
            .ok_or(MacroExecError::NotAssigned { bank, button })?;
        self.state = ExecState::Running;
        self.running_macro = Some(macro_id);
        self.current_step = 0;
        Ok(macro_id)
    }

    /// Advance one step in the currently running macro.
    /// Returns the executed command, or `None` if the macro finished.
    pub fn advance(&mut self) -> Option<StepCommand> {
        if self.state != ExecState::Running {
            return None;
        }
        let macro_id = self.running_macro?;
        let mac = self.macros.get(&macro_id)?;

        while self.current_step < mac.steps.len() {
            let step = &mac.steps[self.current_step];
            let execute = step.should_execute(&self.variables);
            let cmd = step.command.clone();
            self.current_step += 1;

            if execute {
                if cmd == StepCommand::Pause {
                    self.state = ExecState::Paused;
                    return None;
                }
                return Some(cmd);
            }
        }

        // Macro finished
        self.state = ExecState::Idle;
        self.running_macro = None;
        None
    }

    /// Resume a paused macro.
    pub fn resume(&mut self) {
        if self.state == ExecState::Paused {
            self.state = ExecState::Running;
        }
    }

    /// Stop the current macro.
    pub fn stop(&mut self) {
        self.state = ExecState::Idle;
        self.running_macro = None;
        self.current_step = 0;
    }

    /// Set a runtime variable.
    pub fn set_var(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.variables.insert(key.into(), value.into());
    }

    /// Total number of stored macros.
    #[must_use]
    pub fn macro_count(&self) -> usize {
        self.macros.len()
    }

    /// Total number of button assignments.
    #[must_use]
    pub fn assignment_count(&self) -> usize {
        self.assignments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exec() -> MacroBankExec {
        MacroBankExec::new(4, 10)
    }

    fn make_macro_with_steps(id: u32) -> SwitcherMacro {
        let mut m = SwitcherMacro::new(id, format!("Macro {id}"));
        m.add_step(MacroStep::new(
            0,
            StepCommand::SetProgram {
                me_row: 0,
                source: 1,
            },
        ));
        m.add_step(MacroStep::new(1, StepCommand::Cut { me_row: 0 }));
        m
    }

    #[test]
    fn test_store_and_count_macros() {
        let mut exec = make_exec();
        exec.store_macro(make_macro_with_steps(1));
        exec.store_macro(make_macro_with_steps(2));
        assert_eq!(exec.macro_count(), 2);
    }

    #[test]
    fn test_assign_macro_to_button() {
        let mut exec = make_exec();
        exec.store_macro(make_macro_with_steps(1));
        assert!(exec.assign(0, 0, 1).is_ok());
        assert_eq!(exec.assigned_macro_id(0, 0), Some(1));
        assert_eq!(exec.assignment_count(), 1);
    }

    #[test]
    fn test_assign_invalid_bank_errors() {
        let mut exec = make_exec();
        assert_eq!(exec.assign(99, 0, 1), Err(MacroExecError::InvalidBank(99)));
    }

    #[test]
    fn test_assign_invalid_button_errors() {
        let mut exec = make_exec();
        assert_eq!(
            exec.assign(0, 99, 1),
            Err(MacroExecError::InvalidButton(99))
        );
    }

    #[test]
    fn test_trigger_executes_macro() {
        let mut exec = make_exec();
        exec.store_macro(make_macro_with_steps(1));
        exec.assign(0, 0, 1).expect("should succeed in test");
        let id = exec.trigger(0, 0).expect("should succeed in test");
        assert_eq!(id, 1);
        assert_eq!(exec.state, ExecState::Running);
    }

    #[test]
    fn test_trigger_not_assigned_errors() {
        let mut exec = make_exec();
        let err = exec.trigger(0, 0);
        assert_eq!(err, Err(MacroExecError::NotAssigned { bank: 0, button: 0 }));
    }

    #[test]
    fn test_advance_returns_commands_then_none() {
        let mut exec = make_exec();
        exec.store_macro(make_macro_with_steps(1));
        exec.assign(0, 0, 1).expect("should succeed in test");
        exec.trigger(0, 0).expect("should succeed in test");
        let cmd1 = exec.advance();
        assert!(cmd1.is_some());
        let cmd2 = exec.advance();
        assert!(cmd2.is_some());
        let cmd3 = exec.advance();
        assert!(cmd3.is_none());
        assert_eq!(exec.state, ExecState::Idle);
    }

    #[test]
    fn test_conditional_step_skipped() {
        let mut exec = make_exec();
        let mut m = SwitcherMacro::new(5, "Cond Macro");
        m.add_step(MacroStep::conditional(
            0,
            StepCommand::Cut { me_row: 0 },
            StepCondition::VarEquals {
                var: "mode".into(),
                value: "live".into(),
            },
        ));
        exec.store_macro(m);
        exec.assign(0, 1, 5).expect("should succeed in test");
        exec.trigger(0, 1).expect("should succeed in test");
        // Variable not set -> condition fails -> step skipped -> macro done
        let cmd = exec.advance();
        assert!(cmd.is_none());
        assert_eq!(exec.state, ExecState::Idle);
    }

    #[test]
    fn test_conditional_step_executed_when_var_matches() {
        let mut exec = make_exec();
        let mut m = SwitcherMacro::new(6, "Cond Macro 2");
        m.add_step(MacroStep::conditional(
            0,
            StepCommand::Auto { me_row: 0 },
            StepCondition::VarEquals {
                var: "mode".into(),
                value: "live".into(),
            },
        ));
        exec.store_macro(m);
        exec.assign(1, 0, 6).expect("should succeed in test");
        exec.set_var("mode", "live");
        exec.trigger(1, 0).expect("should succeed in test");
        let cmd = exec.advance();
        assert_eq!(cmd, Some(StepCommand::Auto { me_row: 0 }));
    }

    #[test]
    fn test_stop_clears_state() {
        let mut exec = make_exec();
        exec.store_macro(make_macro_with_steps(1));
        exec.assign(0, 0, 1).expect("should succeed in test");
        exec.trigger(0, 0).expect("should succeed in test");
        exec.stop();
        assert_eq!(exec.state, ExecState::Idle);
        assert!(exec.running_macro.is_none());
    }

    #[test]
    fn test_pause_and_resume() {
        let mut exec = make_exec();
        let mut m = SwitcherMacro::new(7, "Pause Macro");
        m.add_step(MacroStep::new(0, StepCommand::Pause));
        m.add_step(MacroStep::new(1, StepCommand::Cut { me_row: 0 }));
        exec.store_macro(m);
        exec.assign(0, 2, 7).expect("should succeed in test");
        exec.trigger(0, 2).expect("should succeed in test");
        let cmd = exec.advance(); // hits Pause step
        assert!(cmd.is_none());
        assert_eq!(exec.state, ExecState::Paused);
        exec.resume();
        assert_eq!(exec.state, ExecState::Running);
        let cmd2 = exec.advance();
        assert_eq!(cmd2, Some(StepCommand::Cut { me_row: 0 }));
    }

    #[test]
    fn test_double_trigger_errors() {
        let mut exec = make_exec();
        exec.store_macro(make_macro_with_steps(1));
        exec.assign(0, 0, 1).expect("should succeed in test");
        exec.trigger(0, 0).expect("should succeed in test");
        let err = exec.trigger(0, 0);
        assert_eq!(err, Err(MacroExecError::AlreadyRunning));
    }

    #[test]
    fn test_set_var_and_retrieve() {
        let mut exec = make_exec();
        exec.set_var("env", "production");
        assert_eq!(
            exec.variables.get("env").map(String::as_str),
            Some("production")
        );
    }
}
