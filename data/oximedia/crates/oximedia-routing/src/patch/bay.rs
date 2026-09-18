//! Virtual patch bay implementation.

use super::input::{InputId, InputManager, PatchInput};
use super::output::{OutputId, OutputManager, PatchOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a patch connection in the bay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    /// Input ID
    pub input: InputId,
    /// Output ID
    pub output: OutputId,
    /// Gain adjustment for this patch (in dB)
    pub gain_db: f32,
    /// Whether this patch is active
    pub active: bool,
}

impl Patch {
    /// Create a new patch
    #[must_use]
    pub const fn new(input: InputId, output: OutputId) -> Self {
        Self {
            input,
            output,
            gain_db: 0.0,
            active: true,
        }
    }

    /// Set the gain for this patch
    #[must_use]
    pub fn with_gain(mut self, gain_db: f32) -> Self {
        self.gain_db = gain_db;
        self
    }
}

/// Virtual patch bay for managing audio routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchBay {
    /// Input manager
    input_manager: InputManager,
    /// Output manager
    output_manager: OutputManager,
    /// Active patches
    patches: Vec<Patch>,
    /// Index of patches by input
    input_index: HashMap<InputId, Vec<usize>>,
    /// Index of patches by output
    output_index: HashMap<OutputId, Vec<usize>>,
}

impl Default for PatchBay {
    fn default() -> Self {
        Self::new()
    }
}

impl PatchBay {
    /// Create a new patch bay
    #[must_use]
    pub fn new() -> Self {
        Self {
            input_manager: InputManager::new(),
            output_manager: OutputManager::new(),
            patches: Vec::new(),
            input_index: HashMap::new(),
            output_index: HashMap::new(),
        }
    }

    /// Get the input manager
    #[must_use]
    pub const fn input_manager(&self) -> &InputManager {
        &self.input_manager
    }

    /// Get mutable input manager
    pub fn input_manager_mut(&mut self) -> &mut InputManager {
        &mut self.input_manager
    }

    /// Get the output manager
    #[must_use]
    pub const fn output_manager(&self) -> &OutputManager {
        &self.output_manager
    }

    /// Get mutable output manager
    pub fn output_manager_mut(&mut self) -> &mut OutputManager {
        &mut self.output_manager
    }

    /// Create a patch between input and output
    pub fn patch(
        &mut self,
        input: InputId,
        output: OutputId,
        gain_db: Option<f32>,
    ) -> Result<(), PatchError> {
        // Verify input and output exist
        if self.input_manager.get_input(input).is_none() {
            return Err(PatchError::InputNotFound(input));
        }
        if self.output_manager.get_output(output).is_none() {
            return Err(PatchError::OutputNotFound(output));
        }

        let patch = if let Some(gain) = gain_db {
            Patch::new(input, output).with_gain(gain)
        } else {
            Patch::new(input, output)
        };

        let patch_idx = self.patches.len();
        self.patches.push(patch);

        self.input_index.entry(input).or_default().push(patch_idx);
        self.output_index.entry(output).or_default().push(patch_idx);

        Ok(())
    }

    /// Remove a patch between input and output
    pub fn unpatch(&mut self, input: InputId, output: OutputId) -> Result<(), PatchError> {
        // Find the patch
        let patch_idx = self
            .patches
            .iter()
            .position(|p| p.input == input && p.output == output)
            .ok_or(PatchError::PatchNotFound(input, output))?;

        // Remove from patches
        self.patches.remove(patch_idx);

        // Rebuild indices (simpler than maintaining them)
        self.rebuild_indices();

        Ok(())
    }

    /// Rebuild patch indices
    fn rebuild_indices(&mut self) {
        self.input_index.clear();
        self.output_index.clear();

        for (idx, patch) in self.patches.iter().enumerate() {
            self.input_index.entry(patch.input).or_default().push(idx);
            self.output_index.entry(patch.output).or_default().push(idx);
        }
    }

    /// Get all patches from an input
    #[must_use]
    pub fn get_patches_from_input(&self, input: InputId) -> Vec<&Patch> {
        self.input_index
            .get(&input)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| self.patches.get(idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all patches to an output
    #[must_use]
    pub fn get_patches_to_output(&self, output: OutputId) -> Vec<&Patch> {
        self.output_index
            .get(&output)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| self.patches.get(idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if input and output are patched
    #[must_use]
    pub fn is_patched(&self, input: InputId, output: OutputId) -> bool {
        self.patches
            .iter()
            .any(|p| p.input == input && p.output == output && p.active)
    }

    /// Get a specific patch
    pub fn get_patch_mut(&mut self, input: InputId, output: OutputId) -> Option<&mut Patch> {
        self.patches
            .iter_mut()
            .find(|p| p.input == input && p.output == output)
    }

    /// Clear all patches
    pub fn clear_patches(&mut self) {
        self.patches.clear();
        self.input_index.clear();
        self.output_index.clear();
    }

    /// Get all active patches
    #[must_use]
    pub fn get_active_patches(&self) -> Vec<&Patch> {
        self.patches.iter().filter(|p| p.active).collect()
    }

    /// Get total number of patches
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Get input by ID
    #[must_use]
    pub fn get_input(&self, id: InputId) -> Option<&PatchInput> {
        self.input_manager.get_input(id)
    }

    /// Get output by ID
    #[must_use]
    pub fn get_output(&self, id: OutputId) -> Option<&PatchOutput> {
        self.output_manager.get_output(id)
    }

    /// Validate patch bay configuration
    pub fn validate(&self) -> Result<(), PatchError> {
        // Check for sample rate mismatches
        for patch in &self.patches {
            if let (Some(input), Some(output)) = (
                self.input_manager.get_input(patch.input),
                self.output_manager.get_output(patch.output),
            ) {
                if input.sample_rate != output.sample_rate {
                    return Err(PatchError::SampleRateMismatch {
                        input_rate: input.sample_rate,
                        output_rate: output.sample_rate,
                    });
                }
                if input.channel_count > output.channel_count {
                    return Err(PatchError::ChannelCountMismatch {
                        input_channels: input.channel_count,
                        output_channels: output.channel_count,
                    });
                }
            }
        }

        Ok(())
    }
}

/// Errors that can occur in patch bay operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum PatchError {
    /// Input not found
    #[error("Input not found: {0:?}")]
    InputNotFound(InputId),
    /// Output not found
    #[error("Output not found: {0:?}")]
    OutputNotFound(OutputId),
    /// Patch not found
    #[error("Patch not found between input {0:?} and output {1:?}")]
    PatchNotFound(InputId, OutputId),
    /// Sample rate mismatch
    #[error("Sample rate mismatch: input {input_rate} Hz, output {output_rate} Hz")]
    SampleRateMismatch { input_rate: u32, output_rate: u32 },
    /// Channel count mismatch
    #[error("Channel count mismatch: input {input_channels}, output {output_channels}")]
    ChannelCountMismatch {
        input_channels: u8,
        output_channels: u8,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::input::SourceType;
    use crate::patch::output::DestinationType;

    #[test]
    fn test_patch_bay_creation() {
        let bay = PatchBay::new();
        assert_eq!(bay.patch_count(), 0);
    }

    #[test]
    fn test_basic_patching() {
        let mut bay = PatchBay::new();

        let input = bay
            .input_manager_mut()
            .add_input("Mic 1".to_string(), SourceType::Microphone);
        let output = bay
            .output_manager_mut()
            .add_output("Monitor".to_string(), DestinationType::Monitor);

        bay.patch(input, output, None)
            .expect("should succeed in test");
        assert!(bay.is_patched(input, output));
        assert_eq!(bay.patch_count(), 1);
    }

    #[test]
    fn test_unpatch() {
        let mut bay = PatchBay::new();

        let input = bay
            .input_manager_mut()
            .add_input("Line 1".to_string(), SourceType::Line);
        let output = bay
            .output_manager_mut()
            .add_output("Out 1".to_string(), DestinationType::Line);

        bay.patch(input, output, None)
            .expect("should succeed in test");
        assert_eq!(bay.patch_count(), 1);

        bay.unpatch(input, output).expect("should succeed in test");
        assert_eq!(bay.patch_count(), 0);
        assert!(!bay.is_patched(input, output));
    }

    #[test]
    fn test_multiple_patches() {
        let mut bay = PatchBay::new();

        let input1 = bay
            .input_manager_mut()
            .add_input("In 1".to_string(), SourceType::Line);
        let input2 = bay
            .input_manager_mut()
            .add_input("In 2".to_string(), SourceType::Line);
        let output = bay
            .output_manager_mut()
            .add_output("Mix".to_string(), DestinationType::Virtual);

        bay.patch(input1, output, None)
            .expect("should succeed in test");
        bay.patch(input2, output, None)
            .expect("should succeed in test");

        let patches_to_output = bay.get_patches_to_output(output);
        assert_eq!(patches_to_output.len(), 2);
    }

    #[test]
    fn test_patch_with_gain() {
        let mut bay = PatchBay::new();

        let input = bay
            .input_manager_mut()
            .add_input("Vocal".to_string(), SourceType::Microphone);
        let output = bay
            .output_manager_mut()
            .add_output("Main".to_string(), DestinationType::Monitor);

        bay.patch(input, output, Some(-6.0))
            .expect("should succeed in test");

        let patches = bay.get_patches_from_input(input);
        assert_eq!(patches.len(), 1);
        assert!((patches[0].gain_db - (-6.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_invalid_patch() {
        let mut bay = PatchBay::new();

        let invalid_input = InputId::new(999);
        let invalid_output = OutputId::new(999);

        assert!(matches!(
            bay.patch(invalid_input, invalid_output, None),
            Err(PatchError::InputNotFound(_))
        ));
    }

    #[test]
    fn test_validation() {
        let mut bay = PatchBay::new();

        // Create input at 48kHz
        let input_id = bay
            .input_manager_mut()
            .add_input("Input".to_string(), SourceType::Line);
        if let Some(input) = bay.input_manager_mut().get_input_mut(input_id) {
            *input = input.clone().with_sample_rate(48000);
        }

        // Create output at 96kHz
        let output_id = bay
            .output_manager_mut()
            .add_output("Output".to_string(), DestinationType::Line);
        if let Some(output) = bay.output_manager_mut().get_output_mut(output_id) {
            *output = output.clone().with_sample_rate(96000);
        }

        bay.patch(input_id, output_id, None)
            .expect("should succeed in test");

        // Validation should fail due to sample rate mismatch
        assert!(matches!(
            bay.validate(),
            Err(PatchError::SampleRateMismatch { .. })
        ));
    }
}
