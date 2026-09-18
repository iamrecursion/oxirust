//! Input management for virtual patch bay.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of input source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceType {
    /// Microphone input
    Microphone,
    /// Line level input
    Line,
    /// Instrument input
    Instrument,
    /// Digital input (AES3, SPDIF, etc.)
    Digital,
    /// Network audio input (Dante, AES67, etc.)
    Network,
    /// Playback from file or recorder
    Playback,
    /// Virtual/internal bus
    Virtual,
}

/// Input channel in the patch bay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchInput {
    /// Unique identifier
    pub id: InputId,
    /// Human-readable label
    pub label: String,
    /// Source type
    pub source_type: SourceType,
    /// Physical or logical location
    pub location: String,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u8,
    /// Number of channels
    pub channel_count: u8,
    /// Whether this input is currently active
    pub active: bool,
    /// Phantom power enabled (for microphones)
    pub phantom_power: bool,
    /// Pad attenuation in dB
    pub pad_db: f32,
    /// Input gain/trim in dB
    pub gain_db: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Unique identifier for an input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InputId(u64);

impl InputId {
    /// Create a new input ID
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the inner ID value
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl PatchInput {
    /// Create a new patch input
    #[must_use]
    pub fn new(id: InputId, label: String, source_type: SourceType) -> Self {
        Self {
            id,
            label,
            source_type,
            location: String::new(),
            sample_rate: 48000,
            bit_depth: 24,
            channel_count: 1,
            active: true,
            phantom_power: false,
            pad_db: 0.0,
            gain_db: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the sample rate
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the bit depth
    #[must_use]
    pub fn with_bit_depth(mut self, bit_depth: u8) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Set the channel count
    #[must_use]
    pub fn with_channel_count(mut self, channel_count: u8) -> Self {
        self.channel_count = channel_count;
        self
    }

    /// Set the location
    #[must_use]
    pub fn with_location(mut self, location: String) -> Self {
        self.location = location;
        self
    }

    /// Enable or disable phantom power
    pub fn set_phantom_power(&mut self, enabled: bool) {
        self.phantom_power = enabled;
    }

    /// Set pad attenuation
    pub fn set_pad(&mut self, pad_db: f32) {
        self.pad_db = pad_db;
    }

    /// Set input gain
    pub fn set_gain(&mut self, gain_db: f32) {
        self.gain_db = gain_db;
    }

    /// Check if this is a stereo input
    #[must_use]
    pub const fn is_stereo(&self) -> bool {
        self.channel_count == 2
    }

    /// Check if this is a multi-channel input
    #[must_use]
    pub const fn is_multichannel(&self) -> bool {
        self.channel_count > 2
    }
}

/// Manager for all patch inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputManager {
    /// All inputs indexed by ID
    inputs: HashMap<InputId, PatchInput>,
    /// Next input ID to assign
    next_id: u64,
    /// Index by source type
    type_index: HashMap<SourceType, Vec<InputId>>,
}

impl Default for InputManager {
    fn default() -> Self {
        Self::new()
    }
}

impl InputManager {
    /// Create a new input manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            next_id: 1,
            type_index: HashMap::new(),
        }
    }

    /// Add a new input
    pub fn add_input(&mut self, label: String, source_type: SourceType) -> InputId {
        let id = InputId::new(self.next_id);
        self.next_id += 1;

        let input = PatchInput::new(id, label, source_type);

        self.type_index.entry(source_type).or_default().push(id);

        self.inputs.insert(id, input);
        id
    }

    /// Remove an input
    pub fn remove_input(&mut self, id: InputId) -> Option<PatchInput> {
        if let Some(input) = self.inputs.remove(&id) {
            if let Some(type_inputs) = self.type_index.get_mut(&input.source_type) {
                type_inputs.retain(|&input_id| input_id != id);
            }
            Some(input)
        } else {
            None
        }
    }

    /// Get an input by ID
    #[must_use]
    pub fn get_input(&self, id: InputId) -> Option<&PatchInput> {
        self.inputs.get(&id)
    }

    /// Get a mutable reference to an input
    pub fn get_input_mut(&mut self, id: InputId) -> Option<&mut PatchInput> {
        self.inputs.get_mut(&id)
    }

    /// Get all inputs of a specific type
    #[must_use]
    pub fn get_inputs_by_type(&self, source_type: SourceType) -> Vec<&PatchInput> {
        self.type_index
            .get(&source_type)
            .map(|ids| ids.iter().filter_map(|id| self.inputs.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all active inputs
    #[must_use]
    pub fn get_active_inputs(&self) -> Vec<&PatchInput> {
        self.inputs.values().filter(|input| input.active).collect()
    }

    /// Get total number of inputs
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Find input by label
    #[must_use]
    pub fn find_by_label(&self, label: &str) -> Option<&PatchInput> {
        self.inputs.values().find(|input| input.label == label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_creation() {
        let input = PatchInput::new(InputId::new(1), "Mic 1".to_string(), SourceType::Microphone);
        assert_eq!(input.label, "Mic 1");
        assert_eq!(input.source_type, SourceType::Microphone);
        assert!(input.active);
    }

    #[test]
    fn test_input_builder() {
        let input = PatchInput::new(InputId::new(1), "Stereo Line".to_string(), SourceType::Line)
            .with_sample_rate(96000)
            .with_bit_depth(24)
            .with_channel_count(2);

        assert_eq!(input.sample_rate, 96000);
        assert_eq!(input.bit_depth, 24);
        assert!(input.is_stereo());
    }

    #[test]
    fn test_input_manager() {
        let mut manager = InputManager::new();

        let id1 = manager.add_input("Mic 1".to_string(), SourceType::Microphone);
        let _id2 = manager.add_input("Line 1".to_string(), SourceType::Line);
        let _id3 = manager.add_input("Mic 2".to_string(), SourceType::Microphone);

        assert_eq!(manager.input_count(), 3);

        let mics = manager.get_inputs_by_type(SourceType::Microphone);
        assert_eq!(mics.len(), 2);

        manager.remove_input(id1);
        assert_eq!(manager.input_count(), 2);
    }

    #[test]
    fn test_phantom_power() {
        let mut input = PatchInput::new(
            InputId::new(1),
            "Condenser Mic".to_string(),
            SourceType::Microphone,
        );

        assert!(!input.phantom_power);
        input.set_phantom_power(true);
        assert!(input.phantom_power);
    }

    #[test]
    fn test_find_by_label() {
        let mut manager = InputManager::new();
        manager.add_input("Vocal Mic".to_string(), SourceType::Microphone);
        manager.add_input("Guitar DI".to_string(), SourceType::Instrument);

        let found = manager.find_by_label("Vocal Mic");
        assert!(found.is_some());
        assert_eq!(
            found.expect("should succeed in test").source_type,
            SourceType::Microphone
        );

        let not_found = manager.find_by_label("Bass DI");
        assert!(not_found.is_none());
    }
}
