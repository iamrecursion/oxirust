//! Program and Preview bus management for video switchers.
//!
//! Implements the traditional switcher architecture with Program (on-air) and Preview buses.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur with bus operations.
#[derive(Error, Debug, Clone)]
pub enum BusError {
    #[error("Invalid bus ID: {0}")]
    InvalidBusId(usize),

    #[error("Invalid input ID: {0}")]
    InvalidInputId(usize),

    #[error("Bus {0} not found")]
    BusNotFound(usize),

    #[error("Cannot perform operation: transition in progress")]
    TransitionInProgress,

    #[error("Bus configuration error: {0}")]
    ConfigError(String),
}

/// Bus type in the switcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BusType {
    /// Program bus (on-air output)
    Program,
    /// Preview bus (off-air monitoring)
    Preview,
    /// Aux bus (auxiliary output)
    Aux(usize),
    /// Clean feed (no graphics)
    CleanFeed,
}

/// Bus assignment - which input is selected on a bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusAssignment {
    /// Bus type
    pub bus_type: BusType,
    /// Selected input ID
    pub input_id: usize,
    /// Mix/Effect row (for multi-M/E switchers)
    pub me_row: usize,
}

impl BusAssignment {
    /// Create a new bus assignment.
    pub fn new(bus_type: BusType, input_id: usize, me_row: usize) -> Self {
        Self {
            bus_type,
            input_id,
            me_row,
        }
    }
}

/// Program/Preview bus manager.
pub struct BusManager {
    /// Number of M/E rows
    me_rows: usize,
    /// Program bus assignments (one per M/E row)
    program: Vec<usize>,
    /// Preview bus assignments (one per M/E row)
    preview: Vec<usize>,
    /// Aux bus assignments
    aux: HashMap<usize, usize>,
    /// Clean feed assignment
    clean_feed: Option<usize>,
    /// Whether a transition is in progress
    transition_active: Vec<bool>,
}

impl BusManager {
    /// Create a new bus manager.
    pub fn new(me_rows: usize, num_aux: usize) -> Self {
        let mut aux = HashMap::new();
        for i in 0..num_aux {
            aux.insert(i, 0); // Default to input 0 (black)
        }

        Self {
            me_rows,
            program: vec![0; me_rows],
            preview: vec![0; me_rows],
            aux,
            clean_feed: Some(0),
            transition_active: vec![false; me_rows],
        }
    }

    /// Get the number of M/E rows.
    pub fn me_rows(&self) -> usize {
        self.me_rows
    }

    /// Set the program bus source.
    pub fn set_program(&mut self, me_row: usize, input_id: usize) -> Result<(), BusError> {
        if me_row >= self.me_rows {
            return Err(BusError::InvalidBusId(me_row));
        }

        self.program[me_row] = input_id;
        Ok(())
    }

    /// Get the program bus source.
    pub fn get_program(&self, me_row: usize) -> Result<usize, BusError> {
        if me_row >= self.me_rows {
            return Err(BusError::InvalidBusId(me_row));
        }

        Ok(self.program[me_row])
    }

    /// Set the preview bus source.
    pub fn set_preview(&mut self, me_row: usize, input_id: usize) -> Result<(), BusError> {
        if me_row >= self.me_rows {
            return Err(BusError::InvalidBusId(me_row));
        }

        self.preview[me_row] = input_id;
        Ok(())
    }

    /// Get the preview bus source.
    pub fn get_preview(&self, me_row: usize) -> Result<usize, BusError> {
        if me_row >= self.me_rows {
            return Err(BusError::InvalidBusId(me_row));
        }

        Ok(self.preview[me_row])
    }

    /// Set an aux bus source.
    pub fn set_aux(&mut self, aux_id: usize, input_id: usize) -> Result<(), BusError> {
        if !self.aux.contains_key(&aux_id) {
            return Err(BusError::InvalidBusId(aux_id));
        }

        self.aux.insert(aux_id, input_id);
        Ok(())
    }

    /// Get an aux bus source.
    pub fn get_aux(&self, aux_id: usize) -> Result<usize, BusError> {
        self.aux
            .get(&aux_id)
            .copied()
            .ok_or(BusError::BusNotFound(aux_id))
    }

    /// Set the clean feed source.
    pub fn set_clean_feed(&mut self, input_id: Option<usize>) {
        self.clean_feed = input_id;
    }

    /// Get the clean feed source.
    pub fn get_clean_feed(&self) -> Option<usize> {
        self.clean_feed
    }

    /// Cut - swap program and preview (instant transition).
    pub fn cut(&mut self, me_row: usize) -> Result<(), BusError> {
        if me_row >= self.me_rows {
            return Err(BusError::InvalidBusId(me_row));
        }

        if self.transition_active[me_row] {
            return Err(BusError::TransitionInProgress);
        }

        // Swap program and preview
        std::mem::swap(&mut self.program[me_row], &mut self.preview[me_row]);

        Ok(())
    }

    /// Take - move preview to program (for use with transitions).
    pub fn take(&mut self, me_row: usize) -> Result<(), BusError> {
        if me_row >= self.me_rows {
            return Err(BusError::InvalidBusId(me_row));
        }

        self.program[me_row] = self.preview[me_row];
        Ok(())
    }

    /// Mark a transition as active.
    pub fn set_transition_active(&mut self, me_row: usize, active: bool) -> Result<(), BusError> {
        if me_row >= self.me_rows {
            return Err(BusError::InvalidBusId(me_row));
        }

        self.transition_active[me_row] = active;
        Ok(())
    }

    /// Check if a transition is active.
    pub fn is_transition_active(&self, me_row: usize) -> bool {
        self.transition_active.get(me_row).copied().unwrap_or(false)
    }

    /// Get all bus assignments.
    pub fn get_all_assignments(&self) -> Vec<BusAssignment> {
        let mut assignments = Vec::new();

        // Program buses
        for (me_row, &input_id) in self.program.iter().enumerate() {
            assignments.push(BusAssignment::new(BusType::Program, input_id, me_row));
        }

        // Preview buses
        for (me_row, &input_id) in self.preview.iter().enumerate() {
            assignments.push(BusAssignment::new(BusType::Preview, input_id, me_row));
        }

        // Aux buses
        for (&aux_id, &input_id) in &self.aux {
            assignments.push(BusAssignment::new(BusType::Aux(aux_id), input_id, 0));
        }

        // Clean feed
        if let Some(input_id) = self.clean_feed {
            assignments.push(BusAssignment::new(BusType::CleanFeed, input_id, 0));
        }

        assignments
    }

    /// Get the input for a specific bus.
    pub fn get_bus_input(&self, bus_type: BusType, me_row: usize) -> Result<usize, BusError> {
        match bus_type {
            BusType::Program => self.get_program(me_row),
            BusType::Preview => self.get_preview(me_row),
            BusType::Aux(aux_id) => self.get_aux(aux_id),
            BusType::CleanFeed => self.get_clean_feed().ok_or(BusError::ConfigError(
                "Clean feed not configured".to_string(),
            )),
        }
    }

    /// Set the input for a specific bus.
    pub fn set_bus_input(
        &mut self,
        bus_type: BusType,
        me_row: usize,
        input_id: usize,
    ) -> Result<(), BusError> {
        match bus_type {
            BusType::Program => self.set_program(me_row, input_id),
            BusType::Preview => self.set_preview(me_row, input_id),
            BusType::Aux(aux_id) => self.set_aux(aux_id, input_id),
            BusType::CleanFeed => {
                self.set_clean_feed(Some(input_id));
                Ok(())
            }
        }
    }

    /// Get all program inputs.
    pub fn get_all_program(&self) -> &[usize] {
        &self.program
    }

    /// Get all preview inputs.
    pub fn get_all_preview(&self) -> &[usize] {
        &self.preview
    }

    /// Get the number of aux buses.
    pub fn aux_count(&self) -> usize {
        self.aux.len()
    }
}

/// Background bus for fill and key sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundBus {
    /// Fill source
    pub fill: usize,
    /// Key source (optional)
    pub key: Option<usize>,
}

impl BackgroundBus {
    /// Create a new background bus.
    pub fn new(fill: usize) -> Self {
        Self { fill, key: None }
    }

    /// Set the fill source.
    pub fn set_fill(&mut self, fill: usize) {
        self.fill = fill;
    }

    /// Set the key source.
    pub fn set_key(&mut self, key: Option<usize>) {
        self.key = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_manager_creation() {
        let manager = BusManager::new(2, 4);
        assert_eq!(manager.me_rows(), 2);
        assert_eq!(manager.aux_count(), 4);
    }

    #[test]
    fn test_program_bus() {
        let mut manager = BusManager::new(2, 4);

        // Default should be input 0
        assert_eq!(manager.get_program(0).expect("should succeed in test"), 0);

        // Set program to input 1
        manager.set_program(0, 1).expect("should succeed in test");
        assert_eq!(manager.get_program(0).expect("should succeed in test"), 1);

        // Invalid M/E row should error
        assert!(manager.set_program(5, 1).is_err());
        assert!(manager.get_program(5).is_err());
    }

    #[test]
    fn test_preview_bus() {
        let mut manager = BusManager::new(2, 4);

        assert_eq!(manager.get_preview(0).expect("should succeed in test"), 0);

        manager.set_preview(0, 2).expect("should succeed in test");
        assert_eq!(manager.get_preview(0).expect("should succeed in test"), 2);
    }

    #[test]
    fn test_aux_bus() {
        let mut manager = BusManager::new(2, 4);

        assert_eq!(manager.get_aux(0).expect("should succeed in test"), 0);

        manager.set_aux(0, 3).expect("should succeed in test");
        assert_eq!(manager.get_aux(0).expect("should succeed in test"), 3);

        // Invalid aux should error
        assert!(manager.get_aux(10).is_err());
    }

    #[test]
    fn test_clean_feed() {
        let mut manager = BusManager::new(2, 4);

        assert_eq!(manager.get_clean_feed(), Some(0));

        manager.set_clean_feed(Some(5));
        assert_eq!(manager.get_clean_feed(), Some(5));

        manager.set_clean_feed(None);
        assert_eq!(manager.get_clean_feed(), None);
    }

    #[test]
    fn test_cut_transition() {
        let mut manager = BusManager::new(2, 4);

        manager.set_program(0, 1).expect("should succeed in test");
        manager.set_preview(0, 2).expect("should succeed in test");

        // Perform cut
        manager.cut(0).expect("should succeed in test");

        // Program and preview should be swapped
        assert_eq!(manager.get_program(0).expect("should succeed in test"), 2);
        assert_eq!(manager.get_preview(0).expect("should succeed in test"), 1);
    }

    #[test]
    fn test_take_transition() {
        let mut manager = BusManager::new(2, 4);

        manager.set_program(0, 1).expect("should succeed in test");
        manager.set_preview(0, 2).expect("should succeed in test");

        // Perform take
        manager.take(0).expect("should succeed in test");

        // Program should match preview
        assert_eq!(manager.get_program(0).expect("should succeed in test"), 2);
        // Preview unchanged
        assert_eq!(manager.get_preview(0).expect("should succeed in test"), 2);
    }

    #[test]
    fn test_transition_active() {
        let mut manager = BusManager::new(2, 4);

        assert!(!manager.is_transition_active(0));

        manager
            .set_transition_active(0, true)
            .expect("should succeed in test");
        assert!(manager.is_transition_active(0));

        // Cannot cut during transition
        assert!(manager.cut(0).is_err());

        manager
            .set_transition_active(0, false)
            .expect("should succeed in test");
        assert!(manager.cut(0).is_ok());
    }

    #[test]
    fn test_bus_assignments() {
        let mut manager = BusManager::new(2, 2);

        manager.set_program(0, 1).expect("should succeed in test");
        manager.set_preview(0, 2).expect("should succeed in test");
        manager.set_aux(0, 3).expect("should succeed in test");

        let assignments = manager.get_all_assignments();
        assert!(assignments.len() >= 5); // 2 program, 2 preview, 2 aux, 1 clean feed
    }

    #[test]
    fn test_background_bus() {
        let mut bg = BackgroundBus::new(1);
        assert_eq!(bg.fill, 1);
        assert_eq!(bg.key, None);

        bg.set_key(Some(2));
        assert_eq!(bg.key, Some(2));

        bg.set_fill(3);
        assert_eq!(bg.fill, 3);
    }

    #[test]
    fn test_bus_type_variants() {
        let program = BusType::Program;
        let preview = BusType::Preview;
        let aux = BusType::Aux(0);
        let clean = BusType::CleanFeed;

        assert_eq!(program, BusType::Program);
        assert_eq!(preview, BusType::Preview);
        assert_eq!(aux, BusType::Aux(0));
        assert_eq!(clean, BusType::CleanFeed);
    }

    #[test]
    fn test_get_set_bus_input() {
        let mut manager = BusManager::new(2, 2);

        // Test program bus
        manager
            .set_bus_input(BusType::Program, 0, 5)
            .expect("should succeed in test");
        assert_eq!(
            manager
                .get_bus_input(BusType::Program, 0)
                .expect("should succeed in test"),
            5
        );

        // Test preview bus
        manager
            .set_bus_input(BusType::Preview, 0, 6)
            .expect("should succeed in test");
        assert_eq!(
            manager
                .get_bus_input(BusType::Preview, 0)
                .expect("should succeed in test"),
            6
        );

        // Test aux bus
        manager
            .set_bus_input(BusType::Aux(0), 0, 7)
            .expect("should succeed in test");
        assert_eq!(
            manager
                .get_bus_input(BusType::Aux(0), 0)
                .expect("should succeed in test"),
            7
        );

        // Test clean feed
        manager
            .set_bus_input(BusType::CleanFeed, 0, 8)
            .expect("should succeed in test");
        assert_eq!(
            manager
                .get_bus_input(BusType::CleanFeed, 0)
                .expect("should succeed in test"),
            8
        );
    }
}
