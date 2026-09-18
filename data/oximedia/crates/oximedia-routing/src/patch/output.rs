//! Output management for virtual patch bay.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of output destination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DestinationType {
    /// Monitor/speaker output
    Monitor,
    /// Headphone output
    Headphone,
    /// Line level output
    Line,
    /// Digital output (AES3, SPDIF, etc.)
    Digital,
    /// Network audio output (Dante, AES67, etc.)
    Network,
    /// Recording destination
    Recording,
    /// Broadcast/transmission output
    Broadcast,
    /// Virtual/internal bus
    Virtual,
}

/// Output channel in the patch bay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchOutput {
    /// Unique identifier
    pub id: OutputId,
    /// Human-readable label
    pub label: String,
    /// Destination type
    pub destination_type: DestinationType,
    /// Physical or logical location
    pub location: String,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u8,
    /// Number of channels
    pub channel_count: u8,
    /// Whether this output is currently active
    pub active: bool,
    /// Output level in dB
    pub level_db: f32,
    /// Mute state
    pub muted: bool,
    /// Dim attenuation when engaged (in dB)
    pub dim_db: f32,
    /// Dim engaged
    pub dim_enabled: bool,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Unique identifier for an output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutputId(u64);

impl OutputId {
    /// Create a new output ID
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

impl PatchOutput {
    /// Create a new patch output
    #[must_use]
    pub fn new(id: OutputId, label: String, destination_type: DestinationType) -> Self {
        Self {
            id,
            label,
            destination_type,
            location: String::new(),
            sample_rate: 48000,
            bit_depth: 24,
            channel_count: 2,
            active: true,
            level_db: 0.0,
            muted: false,
            dim_db: -20.0,
            dim_enabled: false,
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

    /// Set output level
    pub fn set_level(&mut self, level_db: f32) {
        self.level_db = level_db;
    }

    /// Mute or unmute the output
    pub fn set_mute(&mut self, muted: bool) {
        self.muted = muted;
    }

    /// Enable or disable dim
    pub fn set_dim(&mut self, enabled: bool) {
        self.dim_enabled = enabled;
    }

    /// Set dim attenuation
    pub fn set_dim_level(&mut self, dim_db: f32) {
        self.dim_db = dim_db;
    }

    /// Get effective output level (considering mute and dim)
    #[must_use]
    pub fn effective_level_db(&self) -> f32 {
        if self.muted {
            f32::NEG_INFINITY
        } else if self.dim_enabled {
            self.level_db + self.dim_db
        } else {
            self.level_db
        }
    }

    /// Check if this is a stereo output
    #[must_use]
    pub const fn is_stereo(&self) -> bool {
        self.channel_count == 2
    }

    /// Check if this is a surround output
    #[must_use]
    pub const fn is_surround(&self) -> bool {
        self.channel_count >= 6
    }
}

/// Manager for all patch outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputManager {
    /// All outputs indexed by ID
    outputs: HashMap<OutputId, PatchOutput>,
    /// Next output ID to assign
    next_id: u64,
    /// Index by destination type
    type_index: HashMap<DestinationType, Vec<OutputId>>,
}

impl Default for OutputManager {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputManager {
    /// Create a new output manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            outputs: HashMap::new(),
            next_id: 1,
            type_index: HashMap::new(),
        }
    }

    /// Add a new output
    pub fn add_output(&mut self, label: String, destination_type: DestinationType) -> OutputId {
        let id = OutputId::new(self.next_id);
        self.next_id += 1;

        let output = PatchOutput::new(id, label, destination_type);

        self.type_index
            .entry(destination_type)
            .or_default()
            .push(id);

        self.outputs.insert(id, output);
        id
    }

    /// Remove an output
    pub fn remove_output(&mut self, id: OutputId) -> Option<PatchOutput> {
        if let Some(output) = self.outputs.remove(&id) {
            if let Some(type_outputs) = self.type_index.get_mut(&output.destination_type) {
                type_outputs.retain(|&output_id| output_id != id);
            }
            Some(output)
        } else {
            None
        }
    }

    /// Get an output by ID
    #[must_use]
    pub fn get_output(&self, id: OutputId) -> Option<&PatchOutput> {
        self.outputs.get(&id)
    }

    /// Get a mutable reference to an output
    pub fn get_output_mut(&mut self, id: OutputId) -> Option<&mut PatchOutput> {
        self.outputs.get_mut(&id)
    }

    /// Get all outputs of a specific type
    #[must_use]
    pub fn get_outputs_by_type(&self, destination_type: DestinationType) -> Vec<&PatchOutput> {
        self.type_index
            .get(&destination_type)
            .map(|ids| ids.iter().filter_map(|id| self.outputs.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all active outputs
    #[must_use]
    pub fn get_active_outputs(&self) -> Vec<&PatchOutput> {
        self.outputs
            .values()
            .filter(|output| output.active)
            .collect()
    }

    /// Get all unmuted outputs
    #[must_use]
    pub fn get_unmuted_outputs(&self) -> Vec<&PatchOutput> {
        self.outputs
            .values()
            .filter(|output| !output.muted)
            .collect()
    }

    /// Get total number of outputs
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Find output by label
    #[must_use]
    pub fn find_by_label(&self, label: &str) -> Option<&PatchOutput> {
        self.outputs.values().find(|output| output.label == label)
    }

    /// Mute all outputs (e.g., for emergency situations)
    pub fn mute_all(&mut self) {
        for output in self.outputs.values_mut() {
            output.muted = true;
        }
    }

    /// Unmute all outputs
    pub fn unmute_all(&mut self) {
        for output in self.outputs.values_mut() {
            output.muted = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_creation() {
        let output = PatchOutput::new(
            OutputId::new(1),
            "Main L/R".to_string(),
            DestinationType::Monitor,
        );
        assert_eq!(output.label, "Main L/R");
        assert_eq!(output.destination_type, DestinationType::Monitor);
        assert!(output.active);
        assert!(!output.muted);
    }

    #[test]
    fn test_output_builder() {
        let output = PatchOutput::new(
            OutputId::new(1),
            "5.1 Monitor".to_string(),
            DestinationType::Monitor,
        )
        .with_sample_rate(96000)
        .with_bit_depth(24)
        .with_channel_count(6);

        assert_eq!(output.sample_rate, 96000);
        assert_eq!(output.bit_depth, 24);
        assert!(output.is_surround());
    }

    #[test]
    fn test_effective_level() {
        let mut output = PatchOutput::new(
            OutputId::new(1),
            "Monitor".to_string(),
            DestinationType::Monitor,
        );
        output.set_level(-6.0);

        // Normal level
        assert!((output.effective_level_db() - (-6.0)).abs() < f32::EPSILON);

        // Dim enabled
        output.set_dim(true);
        assert!((output.effective_level_db() - (-26.0)).abs() < f32::EPSILON);

        // Muted overrides everything
        output.set_mute(true);
        assert_eq!(output.effective_level_db(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_output_manager() {
        let mut manager = OutputManager::new();

        let id1 = manager.add_output("Main L/R".to_string(), DestinationType::Monitor);
        let _id2 = manager.add_output("HP 1".to_string(), DestinationType::Headphone);
        let _id3 = manager.add_output("Studio A".to_string(), DestinationType::Monitor);

        assert_eq!(manager.output_count(), 3);

        let monitors = manager.get_outputs_by_type(DestinationType::Monitor);
        assert_eq!(monitors.len(), 2);

        manager.remove_output(id1);
        assert_eq!(manager.output_count(), 2);
    }

    #[test]
    fn test_mute_operations() {
        let mut manager = OutputManager::new();

        let id1 = manager.add_output("Out 1".to_string(), DestinationType::Monitor);
        let _id2 = manager.add_output("Out 2".to_string(), DestinationType::Monitor);

        // Mute one output
        if let Some(output) = manager.get_output_mut(id1) {
            output.set_mute(true);
        }

        let unmuted = manager.get_unmuted_outputs();
        assert_eq!(unmuted.len(), 1);

        // Mute all
        manager.mute_all();
        let unmuted_after = manager.get_unmuted_outputs();
        assert_eq!(unmuted_after.len(), 0);
    }

    #[test]
    fn test_find_by_label() {
        let mut manager = OutputManager::new();
        manager.add_output("Studio Monitors".to_string(), DestinationType::Monitor);
        manager.add_output("Headphones".to_string(), DestinationType::Headphone);

        let found = manager.find_by_label("Studio Monitors");
        assert!(found.is_some());
        assert_eq!(
            found.expect("should succeed in test").destination_type,
            DestinationType::Monitor
        );

        let not_found = manager.find_by_label("Control Room");
        assert!(not_found.is_none());
    }
}
