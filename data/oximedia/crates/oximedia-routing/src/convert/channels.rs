//! Channel count conversion configuration.

use serde::{Deserialize, Serialize};

/// Channel count conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelCountConverter {
    /// Input channel count
    pub input_channels: u8,
    /// Output channel count
    pub output_channels: u8,
    /// Conversion mode
    pub mode: ChannelConversionMode,
}

/// Channel conversion mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ChannelConversionMode {
    /// Simple truncate (drop extra channels) or zero-pad
    Simple,
    /// Mix down using standard coefficients
    #[default]
    MixDown,
    /// Duplicate channels for upmix
    Duplicate,
    /// Custom mapping (requires external channel mapper)
    Custom,
}

impl ChannelCountConverter {
    /// Create a new channel count converter
    #[must_use]
    pub fn new(input_channels: u8, output_channels: u8) -> Self {
        let mode = if input_channels > output_channels {
            ChannelConversionMode::MixDown
        } else if input_channels < output_channels {
            ChannelConversionMode::Duplicate
        } else {
            ChannelConversionMode::Simple
        };

        Self {
            input_channels,
            output_channels,
            mode,
        }
    }

    /// Set conversion mode
    #[must_use]
    pub fn with_mode(mut self, mode: ChannelConversionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Check if conversion is needed
    #[must_use]
    pub const fn needs_conversion(&self) -> bool {
        self.input_channels != self.output_channels
    }

    /// Check if this is a downmix
    #[must_use]
    pub const fn is_downmix(&self) -> bool {
        self.input_channels > self.output_channels
    }

    /// Check if this is an upmix
    #[must_use]
    pub const fn is_upmix(&self) -> bool {
        self.input_channels < self.output_channels
    }

    /// Get channel difference
    #[must_use]
    pub const fn channel_difference(&self) -> i8 {
        self.output_channels as i8 - self.input_channels as i8
    }

    /// Create stereo to mono downmix
    #[must_use]
    pub fn stereo_to_mono() -> Self {
        Self::new(2, 1).with_mode(ChannelConversionMode::MixDown)
    }

    /// Create mono to stereo upmix
    #[must_use]
    pub fn mono_to_stereo() -> Self {
        Self::new(1, 2).with_mode(ChannelConversionMode::Duplicate)
    }

    /// Create 5.1 to stereo downmix
    #[must_use]
    pub fn surround_51_to_stereo() -> Self {
        Self::new(6, 2).with_mode(ChannelConversionMode::MixDown)
    }

    /// Create stereo to 5.1 upmix
    #[must_use]
    pub fn stereo_to_surround_51() -> Self {
        Self::new(2, 6).with_mode(ChannelConversionMode::Custom)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), super::sample_rate::ConvertError> {
        if self.input_channels == 0 {
            return Err(super::sample_rate::ConvertError::InvalidChannelCount(
                self.input_channels,
            ));
        }
        if self.output_channels == 0 {
            return Err(super::sample_rate::ConvertError::InvalidChannelCount(
                self.output_channels,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = ChannelCountConverter::new(2, 1);
        assert_eq!(converter.input_channels, 2);
        assert_eq!(converter.output_channels, 1);
        assert!(converter.needs_conversion());
    }

    #[test]
    fn test_downmix_upmix() {
        let downmix = ChannelCountConverter::new(6, 2);
        assert!(downmix.is_downmix());
        assert!(!downmix.is_upmix());

        let upmix = ChannelCountConverter::new(2, 6);
        assert!(upmix.is_upmix());
        assert!(!upmix.is_downmix());
    }

    #[test]
    fn test_channel_difference() {
        let downmix = ChannelCountConverter::new(6, 2);
        assert_eq!(downmix.channel_difference(), -4);

        let upmix = ChannelCountConverter::new(2, 6);
        assert_eq!(upmix.channel_difference(), 4);
    }

    #[test]
    fn test_preset_converters() {
        let stereo_to_mono = ChannelCountConverter::stereo_to_mono();
        assert_eq!(stereo_to_mono.input_channels, 2);
        assert_eq!(stereo_to_mono.output_channels, 1);
        assert_eq!(stereo_to_mono.mode, ChannelConversionMode::MixDown);

        let mono_to_stereo = ChannelCountConverter::mono_to_stereo();
        assert_eq!(mono_to_stereo.input_channels, 1);
        assert_eq!(mono_to_stereo.output_channels, 2);
        assert_eq!(mono_to_stereo.mode, ChannelConversionMode::Duplicate);
    }

    #[test]
    fn test_no_conversion_needed() {
        let converter = ChannelCountConverter::new(2, 2);
        assert!(!converter.needs_conversion());
    }

    #[test]
    fn test_mode_setting() {
        let converter = ChannelCountConverter::new(2, 1).with_mode(ChannelConversionMode::Simple);
        assert_eq!(converter.mode, ChannelConversionMode::Simple);
    }
}
