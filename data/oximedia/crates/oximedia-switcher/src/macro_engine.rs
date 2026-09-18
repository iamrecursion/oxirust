//! Macro recording and playback engine for video switchers.
//!
//! Allows recording sequences of switcher operations and playing them back.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur with macro operations.
#[derive(Error, Debug, Clone)]
pub enum MacroError {
    #[error("Macro {0} not found")]
    MacroNotFound(usize),

    #[error("Invalid macro ID: {0}")]
    InvalidMacroId(usize),

    #[error("Macro is empty")]
    EmptyMacro,

    #[error("Macro already recording")]
    AlreadyRecording,

    #[error("No recording in progress")]
    NotRecording,

    #[error("Macro already running")]
    AlreadyRunning,

    #[error("Playback error: {0}")]
    PlaybackError(String),
}

/// Macro command types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MacroCommand {
    /// Select input on program bus
    SelectProgram { me_row: usize, input: usize },
    /// Select input on preview bus
    SelectPreview { me_row: usize, input: usize },
    /// Perform cut transition
    Cut { me_row: usize },
    /// Perform auto transition
    Auto { me_row: usize },
    /// Set transition type
    SetTransition {
        me_row: usize,
        transition_type: String,
    },
    /// Enable/disable upstream keyer
    SetKeyerOnAir { keyer_id: usize, on_air: bool },
    /// Enable/disable downstream keyer
    SetDskOnAir { dsk_id: usize, on_air: bool },
    /// Select aux output
    SelectAux { aux_id: usize, input: usize },
    /// Load media pool slot
    LoadMediaPool { slot_id: usize },
    /// Wait for duration
    Wait { duration_ms: u64 },
    /// Run another macro
    RunMacro { macro_id: usize },
}

impl MacroCommand {
    /// Get a description of the command.
    pub fn description(&self) -> String {
        match self {
            MacroCommand::SelectProgram { me_row, input } => {
                format!("Select Input {input} on Program M/E {me_row}")
            }
            MacroCommand::SelectPreview { me_row, input } => {
                format!("Select Input {input} on Preview M/E {me_row}")
            }
            MacroCommand::Cut { me_row } => {
                format!("Cut on M/E {me_row}")
            }
            MacroCommand::Auto { me_row } => {
                format!("Auto Transition on M/E {me_row}")
            }
            MacroCommand::SetTransition {
                me_row,
                transition_type,
            } => {
                format!("Set Transition to {transition_type} on M/E {me_row}")
            }
            MacroCommand::SetKeyerOnAir { keyer_id, on_air } => {
                format!("USK {} {}", keyer_id, if *on_air { "ON" } else { "OFF" })
            }
            MacroCommand::SetDskOnAir { dsk_id, on_air } => {
                format!("DSK {} {}", dsk_id, if *on_air { "ON" } else { "OFF" })
            }
            MacroCommand::SelectAux { aux_id, input } => {
                format!("Aux {aux_id} to Input {input}")
            }
            MacroCommand::LoadMediaPool { slot_id } => {
                format!("Load Media Pool Slot {slot_id}")
            }
            MacroCommand::Wait { duration_ms } => {
                format!("Wait {duration_ms} ms")
            }
            MacroCommand::RunMacro { macro_id } => {
                format!("Run Macro {macro_id}")
            }
        }
    }
}

/// Macro definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Macro {
    /// Macro ID
    pub id: usize,
    /// Macro name
    pub name: String,
    /// Description
    pub description: String,
    /// Commands in the macro
    pub commands: Vec<MacroCommand>,
    /// Loop count (0 = run once)
    pub loop_count: usize,
}

impl Macro {
    /// Create a new empty macro.
    pub fn new(id: usize, name: String) -> Self {
        Self {
            id,
            name,
            description: String::new(),
            commands: Vec::new(),
            loop_count: 0,
        }
    }

    /// Add a command to the macro.
    pub fn add_command(&mut self, command: MacroCommand) {
        self.commands.push(command);
    }

    /// Clear all commands.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// Get the number of commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Check if the macro is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Set loop count.
    pub fn set_loop_count(&mut self, count: usize) {
        self.loop_count = count;
    }

    /// Get total duration estimate.
    pub fn estimated_duration(&self) -> Duration {
        let mut total_ms = 0u64;

        for command in &self.commands {
            if let MacroCommand::Wait { duration_ms } = command {
                total_ms += duration_ms;
            } else {
                // Assume 50ms per command for switcher operations
                total_ms += 50;
            }
        }

        Duration::from_millis(total_ms)
    }
}

/// Macro playback state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MacroPlaybackState {
    /// Not running
    Idle,
    /// Currently running
    Running,
    /// Paused
    Paused,
}

/// Macro recorder.
pub struct MacroRecorder {
    recording: bool,
    current_macro: Option<Macro>,
}

impl MacroRecorder {
    /// Create a new macro recorder.
    pub fn new() -> Self {
        Self {
            recording: false,
            current_macro: None,
        }
    }

    /// Start recording a new macro.
    pub fn start_recording(&mut self, id: usize, name: String) -> Result<(), MacroError> {
        if self.recording {
            return Err(MacroError::AlreadyRecording);
        }

        self.current_macro = Some(Macro::new(id, name));
        self.recording = true;
        Ok(())
    }

    /// Stop recording and return the macro.
    pub fn stop_recording(&mut self) -> Result<Macro, MacroError> {
        if !self.recording {
            return Err(MacroError::NotRecording);
        }

        self.recording = false;
        self.current_macro.take().ok_or(MacroError::NotRecording)
    }

    /// Record a command.
    pub fn record_command(&mut self, command: MacroCommand) -> Result<(), MacroError> {
        if !self.recording {
            return Err(MacroError::NotRecording);
        }

        if let Some(macro_) = &mut self.current_macro {
            macro_.add_command(command);
            Ok(())
        } else {
            Err(MacroError::NotRecording)
        }
    }

    /// Check if recording.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Cancel recording.
    pub fn cancel_recording(&mut self) {
        self.recording = false;
        self.current_macro = None;
    }
}

impl Default for MacroRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro player.
pub struct MacroPlayer {
    state: MacroPlaybackState,
    current_macro: Option<Macro>,
    current_command_index: usize,
    current_loop: usize,
}

impl MacroPlayer {
    /// Create a new macro player.
    pub fn new() -> Self {
        Self {
            state: MacroPlaybackState::Idle,
            current_macro: None,
            current_command_index: 0,
            current_loop: 0,
        }
    }

    /// Start playing a macro.
    pub fn play(&mut self, macro_: Macro) -> Result<(), MacroError> {
        if self.state == MacroPlaybackState::Running {
            return Err(MacroError::AlreadyRunning);
        }

        if macro_.is_empty() {
            return Err(MacroError::EmptyMacro);
        }

        self.current_macro = Some(macro_);
        self.current_command_index = 0;
        self.current_loop = 0;
        self.state = MacroPlaybackState::Running;
        Ok(())
    }

    /// Get the next command to execute.
    pub fn next_command(&mut self) -> Option<MacroCommand> {
        if self.state != MacroPlaybackState::Running {
            return None;
        }

        let macro_ = self.current_macro.as_ref()?;

        if self.current_command_index >= macro_.commands.len() {
            // End of commands
            self.current_loop += 1;

            if macro_.loop_count == 0 || self.current_loop >= macro_.loop_count {
                // Done
                self.stop();
                return None;
            }
            // Loop again
            self.current_command_index = 0;
        }

        let command = macro_.commands.get(self.current_command_index)?;
        self.current_command_index += 1;

        Some(command.clone())
    }

    /// Pause playback.
    pub fn pause(&mut self) -> Result<(), MacroError> {
        if self.state != MacroPlaybackState::Running {
            return Err(MacroError::PlaybackError("Not running".to_string()));
        }

        self.state = MacroPlaybackState::Paused;
        Ok(())
    }

    /// Resume playback.
    pub fn resume(&mut self) -> Result<(), MacroError> {
        if self.state != MacroPlaybackState::Paused {
            return Err(MacroError::PlaybackError("Not paused".to_string()));
        }

        self.state = MacroPlaybackState::Running;
        Ok(())
    }

    /// Stop playback.
    pub fn stop(&mut self) {
        self.state = MacroPlaybackState::Idle;
        self.current_macro = None;
        self.current_command_index = 0;
        self.current_loop = 0;
    }

    /// Get the playback state.
    pub fn state(&self) -> MacroPlaybackState {
        self.state
    }

    /// Check if playing.
    pub fn is_playing(&self) -> bool {
        self.state == MacroPlaybackState::Running
    }

    /// Get current progress (0.0 - 1.0).
    pub fn progress(&self) -> f32 {
        if let Some(macro_) = &self.current_macro {
            if macro_.commands.is_empty() {
                return 0.0;
            }
            self.current_command_index as f32 / macro_.commands.len() as f32
        } else {
            0.0
        }
    }
}

impl Default for MacroPlayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro engine manages macro storage and execution.
pub struct MacroEngine {
    macros: HashMap<usize, Macro>,
    recorder: MacroRecorder,
    player: MacroPlayer,
    max_macros: usize,
}

impl MacroEngine {
    /// Create a new macro engine.
    pub fn new(max_macros: usize) -> Self {
        Self {
            macros: HashMap::new(),
            recorder: MacroRecorder::new(),
            player: MacroPlayer::new(),
            max_macros,
        }
    }

    /// Store a macro.
    pub fn store_macro(&mut self, macro_: Macro) -> Result<(), MacroError> {
        if self.macros.len() >= self.max_macros && !self.macros.contains_key(&macro_.id) {
            return Err(MacroError::PlaybackError(format!(
                "Maximum number of macros ({}) reached",
                self.max_macros
            )));
        }

        self.macros.insert(macro_.id, macro_);
        Ok(())
    }

    /// Get a macro.
    pub fn get_macro(&self, id: usize) -> Result<&Macro, MacroError> {
        self.macros.get(&id).ok_or(MacroError::MacroNotFound(id))
    }

    /// Delete a macro.
    pub fn delete_macro(&mut self, id: usize) -> Result<(), MacroError> {
        self.macros
            .remove(&id)
            .ok_or(MacroError::MacroNotFound(id))?;
        Ok(())
    }

    /// Get all macro IDs.
    pub fn macro_ids(&self) -> Vec<usize> {
        self.macros.keys().copied().collect()
    }

    /// Get the recorder.
    pub fn recorder(&self) -> &MacroRecorder {
        &self.recorder
    }

    /// Get mutable recorder.
    pub fn recorder_mut(&mut self) -> &mut MacroRecorder {
        &mut self.recorder
    }

    /// Get the player.
    pub fn player(&self) -> &MacroPlayer {
        &self.player
    }

    /// Get mutable player.
    pub fn player_mut(&mut self) -> &mut MacroPlayer {
        &mut self.player
    }

    /// Run a macro by ID.
    pub fn run_macro(&mut self, id: usize) -> Result<(), MacroError> {
        let macro_ = self.get_macro(id)?.clone();
        self.player.play(macro_)
    }

    /// Get the number of stored macros.
    pub fn macro_count(&self) -> usize {
        self.macros.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_command_description() {
        let cmd = MacroCommand::SelectProgram {
            me_row: 0,
            input: 1,
        };
        assert!(cmd.description().contains("Input 1"));

        let cmd = MacroCommand::Wait { duration_ms: 1000 };
        assert!(cmd.description().contains("1000"));
    }

    #[test]
    fn test_macro_creation() {
        let macro_ = Macro::new(0, "Test Macro".to_string());
        assert_eq!(macro_.id, 0);
        assert_eq!(macro_.name, "Test Macro");
        assert!(macro_.is_empty());
        assert_eq!(macro_.command_count(), 0);
    }

    #[test]
    fn test_macro_add_commands() {
        let mut macro_ = Macro::new(0, "Test".to_string());

        macro_.add_command(MacroCommand::Cut { me_row: 0 });
        macro_.add_command(MacroCommand::Wait { duration_ms: 1000 });

        assert_eq!(macro_.command_count(), 2);
        assert!(!macro_.is_empty());
    }

    #[test]
    fn test_macro_clear() {
        let mut macro_ = Macro::new(0, "Test".to_string());
        macro_.add_command(MacroCommand::Cut { me_row: 0 });

        assert_eq!(macro_.command_count(), 1);

        macro_.clear();
        assert_eq!(macro_.command_count(), 0);
        assert!(macro_.is_empty());
    }

    #[test]
    fn test_macro_estimated_duration() {
        let mut macro_ = Macro::new(0, "Test".to_string());

        macro_.add_command(MacroCommand::Wait { duration_ms: 1000 });
        macro_.add_command(MacroCommand::Cut { me_row: 0 });
        macro_.add_command(MacroCommand::Wait { duration_ms: 500 });

        let duration = macro_.estimated_duration();
        // 1000 + 50 + 500 = 1550 ms
        assert_eq!(duration.as_millis(), 1550);
    }

    #[test]
    fn test_macro_recorder() {
        let mut recorder = MacroRecorder::new();
        assert!(!recorder.is_recording());

        recorder
            .start_recording(0, "Test".to_string())
            .expect("should succeed in test");
        assert!(recorder.is_recording());

        recorder
            .record_command(MacroCommand::Cut { me_row: 0 })
            .expect("should succeed in test");

        let macro_ = recorder.stop_recording().expect("should succeed in test");
        assert_eq!(macro_.command_count(), 1);
        assert!(!recorder.is_recording());
    }

    #[test]
    fn test_macro_recorder_already_recording() {
        let mut recorder = MacroRecorder::new();
        recorder
            .start_recording(0, "Test 1".to_string())
            .expect("should succeed in test");

        assert!(recorder.start_recording(1, "Test 2".to_string()).is_err());
    }

    #[test]
    fn test_macro_recorder_not_recording() {
        let mut recorder = MacroRecorder::new();

        assert!(recorder
            .record_command(MacroCommand::Cut { me_row: 0 })
            .is_err());
        assert!(recorder.stop_recording().is_err());
    }

    #[test]
    fn test_macro_recorder_cancel() {
        let mut recorder = MacroRecorder::new();
        recorder
            .start_recording(0, "Test".to_string())
            .expect("should succeed in test");
        recorder
            .record_command(MacroCommand::Cut { me_row: 0 })
            .expect("should succeed in test");

        recorder.cancel_recording();
        assert!(!recorder.is_recording());
    }

    #[test]
    fn test_macro_player() {
        let mut player = MacroPlayer::new();
        assert!(!player.is_playing());

        let mut macro_ = Macro::new(0, "Test".to_string());
        macro_.add_command(MacroCommand::Cut { me_row: 0 });
        macro_.add_command(MacroCommand::Auto { me_row: 0 });

        player.play(macro_).expect("should succeed in test");
        assert!(player.is_playing());

        let cmd1 = player.next_command();
        assert!(cmd1.is_some());

        let cmd2 = player.next_command();
        assert!(cmd2.is_some());

        let cmd3 = player.next_command();
        assert!(cmd3.is_none());
        assert!(!player.is_playing());
    }

    #[test]
    fn test_macro_player_empty() {
        let mut player = MacroPlayer::new();
        let macro_ = Macro::new(0, "Empty".to_string());

        assert!(player.play(macro_).is_err());
    }

    #[test]
    fn test_macro_player_pause_resume() {
        let mut player = MacroPlayer::new();
        let mut macro_ = Macro::new(0, "Test".to_string());
        macro_.add_command(MacroCommand::Cut { me_row: 0 });

        player.play(macro_).expect("should succeed in test");
        assert!(player.is_playing());

        player.pause().expect("should succeed in test");
        assert_eq!(player.state(), MacroPlaybackState::Paused);

        player.resume().expect("should succeed in test");
        assert!(player.is_playing());
    }

    #[test]
    fn test_macro_player_progress() {
        let mut player = MacroPlayer::new();
        let mut macro_ = Macro::new(0, "Test".to_string());
        macro_.add_command(MacroCommand::Cut { me_row: 0 });
        macro_.add_command(MacroCommand::Auto { me_row: 0 });

        player.play(macro_).expect("should succeed in test");
        assert_eq!(player.progress(), 0.0);

        player.next_command();
        assert_eq!(player.progress(), 0.5);

        player.next_command();
        assert_eq!(player.progress(), 1.0);
    }

    #[test]
    fn test_macro_engine() {
        let mut engine = MacroEngine::new(100);

        let macro_ = Macro::new(0, "Test".to_string());
        engine.store_macro(macro_).expect("should succeed in test");

        assert_eq!(engine.macro_count(), 1);
        assert!(engine.get_macro(0).is_ok());
    }

    #[test]
    fn test_macro_engine_delete() {
        let mut engine = MacroEngine::new(100);

        let macro_ = Macro::new(0, "Test".to_string());
        engine.store_macro(macro_).expect("should succeed in test");

        assert_eq!(engine.macro_count(), 1);

        engine.delete_macro(0).expect("should succeed in test");
        assert_eq!(engine.macro_count(), 0);
        assert!(engine.get_macro(0).is_err());
    }

    #[test]
    fn test_macro_engine_run() {
        let mut engine = MacroEngine::new(100);

        let mut macro_ = Macro::new(0, "Test".to_string());
        macro_.add_command(MacroCommand::Cut { me_row: 0 });
        engine.store_macro(macro_).expect("should succeed in test");

        engine.run_macro(0).expect("should succeed in test");
        assert!(engine.player().is_playing());
    }

    #[test]
    fn test_macro_loop() {
        let mut macro_ = Macro::new(0, "Loop Test".to_string());
        macro_.add_command(MacroCommand::Cut { me_row: 0 });
        macro_.set_loop_count(3);

        assert_eq!(macro_.loop_count, 3);
    }
}
