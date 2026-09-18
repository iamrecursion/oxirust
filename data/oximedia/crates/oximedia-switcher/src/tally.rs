//! Tally system for video switchers.
//!
//! Manages tally states (on-air, preview, etc.) for camera operators and production crew.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur with tally operations.
#[derive(Error, Debug, Clone)]
pub enum TallyError {
    #[error("Invalid input ID: {0}")]
    InvalidInputId(usize),

    #[error("Tally output {0} not found")]
    OutputNotFound(usize),

    #[error("Tally configuration error: {0}")]
    ConfigError(String),
}

/// Tally state for an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TallyState {
    /// Not in use
    Idle,
    /// On program (on-air)
    Program,
    /// On preview
    Preview,
    /// On both program and preview
    ProgramPreview,
}

impl TallyState {
    /// Check if the input is on program.
    pub fn is_program(&self) -> bool {
        matches!(self, TallyState::Program | TallyState::ProgramPreview)
    }

    /// Check if the input is on preview.
    pub fn is_preview(&self) -> bool {
        matches!(self, TallyState::Preview | TallyState::ProgramPreview)
    }

    /// Check if the input is idle.
    pub fn is_idle(&self) -> bool {
        matches!(self, TallyState::Idle)
    }

    /// Combine with another tally state.
    pub fn combine(&self, other: TallyState) -> TallyState {
        match (self, other) {
            (TallyState::Idle, other_state) => other_state,
            (self_state, TallyState::Idle) => *self_state,
            (TallyState::Program, TallyState::Preview)
            | (TallyState::Preview, TallyState::Program)
            | (TallyState::ProgramPreview, _)
            | (_, TallyState::ProgramPreview) => TallyState::ProgramPreview,
            (TallyState::Program, TallyState::Program) => TallyState::Program,
            (TallyState::Preview, TallyState::Preview) => TallyState::Preview,
        }
    }
}

/// Tally output type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TallyOutput {
    /// GPIO output (pin number)
    Gpio { pin: usize },
    /// Network tally (IP address)
    Network { address: String, port: u16 },
    /// Serial tally
    Serial { port: String },
    /// TSL UMD protocol
    TslUmd { address: String, port: u16 },
}

/// Tally configuration for an input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TallyConfig {
    /// Input ID this tally is for
    pub input_id: usize,
    /// Enabled state
    pub enabled: bool,
    /// Output configurations
    pub outputs: Vec<TallyOutput>,
}

impl TallyConfig {
    /// Create a new tally configuration.
    pub fn new(input_id: usize) -> Self {
        Self {
            input_id,
            enabled: true,
            outputs: Vec::new(),
        }
    }

    /// Add an output.
    pub fn add_output(&mut self, output: TallyOutput) {
        self.outputs.push(output);
    }
}

/// Tally information for an input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TallyInfo {
    /// Input ID
    pub input_id: usize,
    /// Current tally state
    pub state: TallyState,
    /// Configuration
    pub config: TallyConfig,
}

impl TallyInfo {
    /// Create a new tally info.
    pub fn new(input_id: usize) -> Self {
        Self {
            input_id,
            state: TallyState::Idle,
            config: TallyConfig::new(input_id),
        }
    }

    /// Update the tally state.
    pub fn set_state(&mut self, state: TallyState) {
        self.state = state;
    }

    /// Get the tally state.
    pub fn state(&self) -> TallyState {
        self.state
    }
}

/// Tally manager tracks tally states for all inputs.
pub struct TallyManager {
    /// Tally information per input
    tallies: HashMap<usize, TallyInfo>,
    /// Whether tally system is enabled
    enabled: bool,
}

impl TallyManager {
    /// Create a new tally manager.
    pub fn new() -> Self {
        Self {
            tallies: HashMap::new(),
            enabled: true,
        }
    }

    /// Enable or disable the tally system.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the tally system is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Add or update an input's tally.
    pub fn set_tally(&mut self, input_id: usize, state: TallyState) {
        self.tallies
            .entry(input_id)
            .or_insert_with(|| TallyInfo::new(input_id))
            .set_state(state);
    }

    /// Get an input's tally state.
    pub fn get_tally(&self, input_id: usize) -> TallyState {
        self.tallies
            .get(&input_id)
            .map_or(TallyState::Idle, |t| t.state)
    }

    /// Get tally information for an input.
    pub fn get_tally_info(&self, input_id: usize) -> Option<&TallyInfo> {
        self.tallies.get(&input_id)
    }

    /// Get mutable tally information for an input.
    pub fn get_tally_info_mut(&mut self, input_id: usize) -> Option<&mut TallyInfo> {
        self.tallies.get_mut(&input_id)
    }

    /// Clear all tally states.
    pub fn clear_all(&mut self) {
        for tally in self.tallies.values_mut() {
            tally.set_state(TallyState::Idle);
        }
    }

    /// Update tallies based on program and preview selections.
    pub fn update_from_buses(&mut self, program_inputs: &[usize], preview_inputs: &[usize]) {
        // Clear all first
        self.clear_all();

        // Set program tallies
        for &input_id in program_inputs {
            let current = self.get_tally(input_id);
            self.set_tally(input_id, current.combine(TallyState::Program));
        }

        // Set preview tallies
        for &input_id in preview_inputs {
            let current = self.get_tally(input_id);
            self.set_tally(input_id, current.combine(TallyState::Preview));
        }
    }

    /// Get all inputs on program.
    pub fn get_program_inputs(&self) -> Vec<usize> {
        self.tallies
            .iter()
            .filter(|(_, t)| t.state.is_program())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all inputs on preview.
    pub fn get_preview_inputs(&self) -> Vec<usize> {
        self.tallies
            .iter()
            .filter(|(_, t)| t.state.is_preview())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all idle inputs.
    pub fn get_idle_inputs(&self) -> Vec<usize> {
        self.tallies
            .iter()
            .filter(|(_, t)| t.state.is_idle())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get the number of inputs with tally info.
    pub fn count(&self) -> usize {
        self.tallies.len()
    }

    /// Configure tally output for an input.
    pub fn configure_output(
        &mut self,
        input_id: usize,
        output: TallyOutput,
    ) -> Result<(), TallyError> {
        let tally = self
            .tallies
            .entry(input_id)
            .or_insert_with(|| TallyInfo::new(input_id));

        tally.config.add_output(output);
        Ok(())
    }

    /// Get all tally states as a map.
    pub fn get_all_states(&self) -> HashMap<usize, TallyState> {
        self.tallies
            .iter()
            .map(|(&id, info)| (id, info.state))
            .collect()
    }
}

impl Default for TallyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tally_state_checks() {
        assert!(TallyState::Program.is_program());
        assert!(!TallyState::Program.is_preview());
        assert!(!TallyState::Program.is_idle());

        assert!(!TallyState::Preview.is_program());
        assert!(TallyState::Preview.is_preview());
        assert!(!TallyState::Preview.is_idle());

        assert!(TallyState::ProgramPreview.is_program());
        assert!(TallyState::ProgramPreview.is_preview());
        assert!(!TallyState::ProgramPreview.is_idle());

        assert!(!TallyState::Idle.is_program());
        assert!(!TallyState::Idle.is_preview());
        assert!(TallyState::Idle.is_idle());
    }

    #[test]
    fn test_tally_state_combine() {
        assert_eq!(
            TallyState::Idle.combine(TallyState::Program),
            TallyState::Program
        );

        assert_eq!(
            TallyState::Program.combine(TallyState::Preview),
            TallyState::ProgramPreview
        );

        assert_eq!(
            TallyState::Program.combine(TallyState::Program),
            TallyState::Program
        );

        assert_eq!(
            TallyState::ProgramPreview.combine(TallyState::Program),
            TallyState::ProgramPreview
        );
    }

    #[test]
    fn test_tally_manager_creation() {
        let manager = TallyManager::new();
        assert!(manager.is_enabled());
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_set_get_tally() {
        let mut manager = TallyManager::new();

        assert_eq!(manager.get_tally(1), TallyState::Idle);

        manager.set_tally(1, TallyState::Program);
        assert_eq!(manager.get_tally(1), TallyState::Program);

        manager.set_tally(1, TallyState::Preview);
        assert_eq!(manager.get_tally(1), TallyState::Preview);
    }

    #[test]
    fn test_clear_all() {
        let mut manager = TallyManager::new();

        manager.set_tally(1, TallyState::Program);
        manager.set_tally(2, TallyState::Preview);
        manager.set_tally(3, TallyState::ProgramPreview);

        assert_eq!(manager.count(), 3);

        manager.clear_all();

        assert_eq!(manager.get_tally(1), TallyState::Idle);
        assert_eq!(manager.get_tally(2), TallyState::Idle);
        assert_eq!(manager.get_tally(3), TallyState::Idle);
    }

    #[test]
    fn test_update_from_buses() {
        let mut manager = TallyManager::new();

        let program = vec![1, 2];
        let preview = vec![2, 3];

        manager.update_from_buses(&program, &preview);

        assert_eq!(manager.get_tally(1), TallyState::Program);
        assert_eq!(manager.get_tally(2), TallyState::ProgramPreview);
        assert_eq!(manager.get_tally(3), TallyState::Preview);
        assert_eq!(manager.get_tally(4), TallyState::Idle);
    }

    #[test]
    fn test_get_program_inputs() {
        let mut manager = TallyManager::new();

        manager.set_tally(1, TallyState::Program);
        manager.set_tally(2, TallyState::Preview);
        manager.set_tally(3, TallyState::ProgramPreview);

        let program = manager.get_program_inputs();
        assert_eq!(program.len(), 2);
        assert!(program.contains(&1));
        assert!(program.contains(&3));
    }

    #[test]
    fn test_get_preview_inputs() {
        let mut manager = TallyManager::new();

        manager.set_tally(1, TallyState::Program);
        manager.set_tally(2, TallyState::Preview);
        manager.set_tally(3, TallyState::ProgramPreview);

        let preview = manager.get_preview_inputs();
        assert_eq!(preview.len(), 2);
        assert!(preview.contains(&2));
        assert!(preview.contains(&3));
    }

    #[test]
    fn test_tally_config() {
        let mut config = TallyConfig::new(1);
        assert_eq!(config.input_id, 1);
        assert!(config.enabled);
        assert_eq!(config.outputs.len(), 0);

        config.add_output(TallyOutput::Gpio { pin: 5 });
        assert_eq!(config.outputs.len(), 1);
    }

    #[test]
    fn test_tally_output_types() {
        let gpio = TallyOutput::Gpio { pin: 1 };
        let network = TallyOutput::Network {
            address: "192.168.1.100".to_string(),
            port: 8080,
        };
        let serial = TallyOutput::Serial {
            port: "/dev/ttyUSB0".to_string(),
        };
        let tsl = TallyOutput::TslUmd {
            address: "192.168.1.101".to_string(),
            port: 5727,
        };

        assert!(matches!(gpio, TallyOutput::Gpio { .. }));
        assert!(matches!(network, TallyOutput::Network { .. }));
        assert!(matches!(serial, TallyOutput::Serial { .. }));
        assert!(matches!(tsl, TallyOutput::TslUmd { .. }));
    }

    #[test]
    fn test_configure_output() {
        let mut manager = TallyManager::new();

        let output = TallyOutput::Network {
            address: "192.168.1.100".to_string(),
            port: 8080,
        };

        manager
            .configure_output(1, output)
            .expect("should succeed in test");

        let info = manager.get_tally_info(1).expect("should succeed in test");
        assert_eq!(info.config.outputs.len(), 1);
    }

    #[test]
    fn test_enable_disable() {
        let mut manager = TallyManager::new();
        assert!(manager.is_enabled());

        manager.set_enabled(false);
        assert!(!manager.is_enabled());

        manager.set_enabled(true);
        assert!(manager.is_enabled());
    }
}
