//! Input source management for video switchers.
//!
//! Handles multiple video and audio sources including SDI, NDI, files, and color generators.

use oximedia_codec::{AudioFrame, VideoFrame};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur with input sources.
#[derive(Error, Debug, Clone)]
pub enum InputError {
    #[error("Input {0} not found")]
    InputNotFound(usize),

    #[error("Invalid input ID: {0}")]
    InvalidInputId(usize),

    #[error("Input {0} is not active")]
    InputNotActive(usize),

    #[error("Format mismatch: expected {expected}, got {actual}")]
    FormatMismatch { expected: String, actual: String },

    #[error("No signal on input {0}")]
    NoSignal(usize),

    #[error("Input configuration error: {0}")]
    ConfigError(String),
}

/// Type of input source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InputType {
    /// SDI (Serial Digital Interface)
    Sdi { port: usize },
    /// NDI (Network Device Interface)
    Ndi { name: String, address: String },
    /// File playback
    File { path: PathBuf },
    /// Color generator
    ColorBars,
    /// Solid color
    Color { r: u8, g: u8, b: u8 },
    /// Black
    Black,
    /// Media pool still
    MediaPool { index: usize },
    /// Test pattern
    TestPattern { pattern: TestPatternType },
}

/// Test pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TestPatternType {
    ColorBars,
    Ramp,
    Checkerboard,
    Grid,
}

/// Input source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    /// Input ID
    pub id: usize,
    /// Input name
    pub name: String,
    /// Input type
    pub input_type: InputType,
    /// Long name/description
    pub long_name: String,
    /// Enable/disable input
    pub enabled: bool,
}

impl InputConfig {
    /// Create a new input configuration.
    pub fn new(id: usize, name: String, input_type: InputType) -> Self {
        Self {
            id,
            name: name.clone(),
            input_type,
            long_name: name,
            enabled: true,
        }
    }
}

/// Input status information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InputStatus {
    /// Whether the input has a valid signal
    pub has_signal: bool,
    /// Video format detected
    pub video_format: Option<String>,
    /// Audio channels detected
    pub audio_channels: usize,
    /// Frame count
    pub frame_count: u64,
    /// Whether the input is currently in use
    pub in_use: bool,
}

/// Video input source.
pub struct InputSource {
    config: InputConfig,
    status: InputStatus,
    current_video_frame: Option<VideoFrame>,
    current_audio_frame: Option<AudioFrame>,
}

impl InputSource {
    /// Create a new input source.
    pub fn new(config: InputConfig) -> Self {
        Self {
            config,
            status: InputStatus::default(),
            current_video_frame: None,
            current_audio_frame: None,
        }
    }

    /// Get the input ID.
    pub fn id(&self) -> usize {
        self.config.id
    }

    /// Get the input name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get the input configuration.
    pub fn config(&self) -> &InputConfig {
        &self.config
    }

    /// Get mutable configuration.
    pub fn config_mut(&mut self) -> &mut InputConfig {
        &mut self.config
    }

    /// Get the input status.
    pub fn status(&self) -> &InputStatus {
        &self.status
    }

    /// Update the input status.
    pub fn update_status(&mut self, status: InputStatus) {
        self.status = status;
    }

    /// Check if the input has a valid signal.
    pub fn has_signal(&self) -> bool {
        self.status.has_signal
    }

    /// Set signal status.
    pub fn set_signal(&mut self, has_signal: bool) {
        self.status.has_signal = has_signal;
    }

    /// Get the current video frame.
    pub fn current_video_frame(&self) -> Option<&VideoFrame> {
        self.current_video_frame.as_ref()
    }

    /// Update the current video frame.
    pub fn update_video_frame(&mut self, frame: VideoFrame) {
        self.current_video_frame = Some(frame);
        self.status.frame_count += 1;
    }

    /// Get the current audio frame.
    pub fn current_audio_frame(&self) -> Option<&AudioFrame> {
        self.current_audio_frame.as_ref()
    }

    /// Update the current audio frame.
    pub fn update_audio_frame(&mut self, frame: AudioFrame) {
        self.current_audio_frame = Some(frame);
    }

    /// Mark input as in use.
    pub fn set_in_use(&mut self, in_use: bool) {
        self.status.in_use = in_use;
    }

    /// Check if input is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Enable or disable the input.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }
}

/// Input router manages all input sources.
pub struct InputRouter {
    inputs: HashMap<usize, InputSource>,
    max_inputs: usize,
}

impl InputRouter {
    /// Create a new input router.
    pub fn new(max_inputs: usize) -> Self {
        Self {
            inputs: HashMap::new(),
            max_inputs,
        }
    }

    /// Add an input source.
    pub fn add_input(&mut self, config: InputConfig) -> Result<(), InputError> {
        if self.inputs.len() >= self.max_inputs {
            return Err(InputError::ConfigError(format!(
                "Maximum number of inputs ({}) reached",
                self.max_inputs
            )));
        }

        let id = config.id;
        self.inputs.insert(id, InputSource::new(config));
        Ok(())
    }

    /// Remove an input source.
    pub fn remove_input(&mut self, id: usize) -> Result<(), InputError> {
        self.inputs
            .remove(&id)
            .ok_or(InputError::InputNotFound(id))?;
        Ok(())
    }

    /// Get an input source.
    pub fn get_input(&self, id: usize) -> Result<&InputSource, InputError> {
        self.inputs.get(&id).ok_or(InputError::InputNotFound(id))
    }

    /// Get a mutable input source.
    pub fn get_input_mut(&mut self, id: usize) -> Result<&mut InputSource, InputError> {
        self.inputs
            .get_mut(&id)
            .ok_or(InputError::InputNotFound(id))
    }

    /// Get all input IDs.
    pub fn input_ids(&self) -> Vec<usize> {
        self.inputs.keys().copied().collect()
    }

    /// Get all inputs.
    pub fn inputs(&self) -> impl Iterator<Item = &InputSource> {
        self.inputs.values()
    }

    /// Get the number of inputs.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Check if an input exists.
    pub fn has_input(&self, id: usize) -> bool {
        self.inputs.contains_key(&id)
    }

    /// Get inputs with valid signals.
    pub fn active_inputs(&self) -> impl Iterator<Item = &InputSource> {
        self.inputs
            .values()
            .filter(|i| i.has_signal() && i.is_enabled())
    }

    /// Get the number of active inputs.
    pub fn active_count(&self) -> usize {
        self.active_inputs().count()
    }

    /// Update an input's video frame.
    pub fn update_video(&mut self, id: usize, frame: VideoFrame) -> Result<(), InputError> {
        let input = self.get_input_mut(id)?;
        input.update_video_frame(frame);
        Ok(())
    }

    /// Update an input's audio frame.
    pub fn update_audio(&mut self, id: usize, frame: AudioFrame) -> Result<(), InputError> {
        let input = self.get_input_mut(id)?;
        input.update_audio_frame(frame);
        Ok(())
    }

    /// Get video frame from an input.
    pub fn get_video(&self, id: usize) -> Result<Option<&VideoFrame>, InputError> {
        let input = self.get_input(id)?;
        Ok(input.current_video_frame())
    }

    /// Get audio frame from an input.
    pub fn get_audio(&self, id: usize) -> Result<Option<&AudioFrame>, InputError> {
        let input = self.get_input(id)?;
        Ok(input.current_audio_frame())
    }

    /// Clear all inputs.
    pub fn clear(&mut self) {
        self.inputs.clear();
    }
}

/// Input matrix for routing inputs to outputs.
pub struct InputMatrix {
    router: InputRouter,
    crosspoints: HashMap<(usize, usize), bool>,
}

impl InputMatrix {
    /// Create a new input matrix.
    pub fn new(max_inputs: usize) -> Self {
        Self {
            router: InputRouter::new(max_inputs),
            crosspoints: HashMap::new(),
        }
    }

    /// Get the input router.
    pub fn router(&self) -> &InputRouter {
        &self.router
    }

    /// Get mutable input router.
    pub fn router_mut(&mut self) -> &mut InputRouter {
        &mut self.router
    }

    /// Connect an input to an output.
    pub fn connect(&mut self, input_id: usize, output_id: usize) -> Result<(), InputError> {
        if !self.router.has_input(input_id) {
            return Err(InputError::InputNotFound(input_id));
        }
        self.crosspoints.insert((input_id, output_id), true);
        Ok(())
    }

    /// Disconnect an input from an output.
    pub fn disconnect(&mut self, input_id: usize, output_id: usize) {
        self.crosspoints.remove(&(input_id, output_id));
    }

    /// Check if an input is connected to an output.
    pub fn is_connected(&self, input_id: usize, output_id: usize) -> bool {
        self.crosspoints
            .get(&(input_id, output_id))
            .copied()
            .unwrap_or(false)
    }

    /// Get all outputs for an input.
    pub fn get_outputs_for_input(&self, input_id: usize) -> Vec<usize> {
        self.crosspoints
            .keys()
            .filter(|(i, _)| *i == input_id)
            .map(|(_, o)| *o)
            .collect()
    }

    /// Get the input for an output.
    pub fn get_input_for_output(&self, output_id: usize) -> Option<usize> {
        self.crosspoints
            .keys()
            .find(|(_, o)| *o == output_id)
            .map(|(i, _)| *i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_config_creation() {
        let config = InputConfig::new(1, "Camera 1".to_string(), InputType::Sdi { port: 0 });
        assert_eq!(config.id, 1);
        assert_eq!(config.name, "Camera 1");
        assert!(config.enabled);
    }

    #[test]
    fn test_input_source_creation() {
        let config = InputConfig::new(1, "Camera 1".to_string(), InputType::Sdi { port: 0 });
        let source = InputSource::new(config);
        assert_eq!(source.id(), 1);
        assert_eq!(source.name(), "Camera 1");
        assert!(!source.has_signal());
    }

    #[test]
    fn test_input_router() {
        let mut router = InputRouter::new(10);
        assert_eq!(router.input_count(), 0);

        let config = InputConfig::new(1, "Camera 1".to_string(), InputType::Sdi { port: 0 });
        router.add_input(config).expect("should succeed in test");
        assert_eq!(router.input_count(), 1);
        assert!(router.has_input(1));
        assert!(!router.has_input(2));
    }

    #[test]
    fn test_input_router_max_inputs() {
        let mut router = InputRouter::new(2);

        let config1 = InputConfig::new(1, "Input 1".to_string(), InputType::Black);
        let config2 = InputConfig::new(2, "Input 2".to_string(), InputType::Black);
        let config3 = InputConfig::new(3, "Input 3".to_string(), InputType::Black);

        assert!(router.add_input(config1).is_ok());
        assert!(router.add_input(config2).is_ok());
        assert!(router.add_input(config3).is_err());
    }

    #[test]
    fn test_input_signal_status() {
        let config = InputConfig::new(1, "Camera 1".to_string(), InputType::Sdi { port: 0 });
        let mut source = InputSource::new(config);

        assert!(!source.has_signal());
        source.set_signal(true);
        assert!(source.has_signal());
    }

    #[test]
    fn test_input_matrix() {
        let mut matrix = InputMatrix::new(10);

        let config = InputConfig::new(1, "Camera 1".to_string(), InputType::Sdi { port: 0 });
        matrix
            .router_mut()
            .add_input(config)
            .expect("should succeed in test");

        // Connect input 1 to output 0
        matrix.connect(1, 0).expect("should succeed in test");
        assert!(matrix.is_connected(1, 0));
        assert!(!matrix.is_connected(1, 1));

        // Disconnect
        matrix.disconnect(1, 0);
        assert!(!matrix.is_connected(1, 0));
    }

    #[test]
    fn test_input_type_variants() {
        let sdi = InputType::Sdi { port: 0 };
        let ndi = InputType::Ndi {
            name: "Camera 1".to_string(),
            address: "192.168.1.100".to_string(),
        };
        let color = InputType::Color { r: 255, g: 0, b: 0 };
        let black = InputType::Black;

        assert!(matches!(sdi, InputType::Sdi { .. }));
        assert!(matches!(ndi, InputType::Ndi { .. }));
        assert!(matches!(color, InputType::Color { .. }));
        assert!(matches!(black, InputType::Black));
    }

    #[test]
    fn test_active_inputs() {
        let mut router = InputRouter::new(10);

        let config1 = InputConfig::new(1, "Camera 1".to_string(), InputType::Sdi { port: 0 });
        let config2 = InputConfig::new(2, "Camera 2".to_string(), InputType::Sdi { port: 1 });

        router.add_input(config1).expect("should succeed in test");
        router.add_input(config2).expect("should succeed in test");

        assert_eq!(router.active_count(), 0);

        router
            .get_input_mut(1)
            .expect("should succeed in test")
            .set_signal(true);
        assert_eq!(router.active_count(), 1);

        router
            .get_input_mut(2)
            .expect("should succeed in test")
            .set_signal(true);
        assert_eq!(router.active_count(), 2);
    }
}
