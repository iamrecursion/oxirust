//! Microphone capture.

use crate::GamingResult;

/// Microphone capture.
#[allow(dead_code)]
pub struct MicrophoneCapture {
    config: MicConfig,
}

/// Microphone configuration.
#[derive(Debug, Clone)]
pub struct MicConfig {
    /// Device ID
    pub device_id: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Noise suppression
    pub noise_suppression: bool,
    /// Echo cancellation
    pub echo_cancellation: bool,
}

impl MicrophoneCapture {
    /// Create a new microphone capture.
    #[must_use]
    pub fn new(config: MicConfig) -> Self {
        Self { config }
    }

    /// Start capture.
    pub fn start(&mut self) -> GamingResult<()> {
        Ok(())
    }

    /// Stop capture.
    pub fn stop(&mut self) {}
}

impl Default for MicConfig {
    fn default() -> Self {
        Self {
            device_id: "default".to_string(),
            sample_rate: 48000,
            noise_suppression: true,
            echo_cancellation: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mic_creation() {
        let _mic = MicrophoneCapture::new(MicConfig::default());
    }
}
