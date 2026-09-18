//! Sample rate conversion configuration.

use serde::{Deserialize, Serialize};

/// Sample rate conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRateConverter {
    /// Input sample rate in Hz
    pub input_rate: u32,
    /// Output sample rate in Hz
    pub output_rate: u32,
    /// Quality setting for conversion
    pub quality: ConversionQuality,
    /// Number of channels to convert
    pub channels: u8,
}

/// Quality settings for sample rate conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ConversionQuality {
    /// Low quality, fastest processing
    Low,
    /// Medium quality, balanced
    Medium,
    /// High quality, slower processing
    #[default]
    High,
    /// Maximum quality, slowest processing
    Maximum,
}

impl SampleRateConverter {
    /// Create a new sample rate converter
    #[must_use]
    pub fn new(input_rate: u32, output_rate: u32, channels: u8) -> Self {
        Self {
            input_rate,
            output_rate,
            quality: ConversionQuality::High,
            channels,
        }
    }

    /// Set quality
    #[must_use]
    pub fn with_quality(mut self, quality: ConversionQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Check if conversion is needed
    #[must_use]
    pub const fn needs_conversion(&self) -> bool {
        self.input_rate != self.output_rate
    }

    /// Get conversion ratio
    #[must_use]
    pub fn conversion_ratio(&self) -> f64 {
        f64::from(self.output_rate) / f64::from(self.input_rate)
    }

    /// Check if this is an upsample operation
    #[must_use]
    pub const fn is_upsample(&self) -> bool {
        self.output_rate > self.input_rate
    }

    /// Check if this is a downsample operation
    #[must_use]
    pub const fn is_downsample(&self) -> bool {
        self.output_rate < self.input_rate
    }

    /// Create converter for 44.1 kHz to 48 kHz
    #[must_use]
    pub fn cd_to_broadcast(channels: u8) -> Self {
        Self::new(44100, 48000, channels)
    }

    /// Create converter for 48 kHz to 96 kHz
    #[must_use]
    pub fn broadcast_to_hires(channels: u8) -> Self {
        Self::new(48000, 96000, channels)
    }

    /// Create converter for 96 kHz to 48 kHz
    #[must_use]
    pub fn hires_to_broadcast(channels: u8) -> Self {
        Self::new(96000, 48000, channels)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConvertError> {
        if self.input_rate == 0 {
            return Err(ConvertError::InvalidSampleRate(self.input_rate));
        }
        if self.output_rate == 0 {
            return Err(ConvertError::InvalidSampleRate(self.output_rate));
        }
        if self.channels == 0 {
            return Err(ConvertError::InvalidChannelCount(self.channels));
        }
        Ok(())
    }
}

/// Errors that can occur in conversion
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConvertError {
    /// Invalid sample rate
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),
    /// Invalid channel count
    #[error("Invalid channel count: {0}")]
    InvalidChannelCount(u8),
    /// Invalid bit depth
    #[error("Invalid bit depth: {0}")]
    InvalidBitDepth(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = SampleRateConverter::new(48000, 96000, 2);
        assert_eq!(converter.input_rate, 48000);
        assert_eq!(converter.output_rate, 96000);
        assert!(converter.needs_conversion());
    }

    #[test]
    fn test_conversion_ratio() {
        let converter = SampleRateConverter::new(48000, 96000, 2);
        assert!((converter.conversion_ratio() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_upsample_downsample() {
        let upsample = SampleRateConverter::new(48000, 96000, 2);
        assert!(upsample.is_upsample());
        assert!(!upsample.is_downsample());

        let downsample = SampleRateConverter::new(96000, 48000, 2);
        assert!(downsample.is_downsample());
        assert!(!downsample.is_upsample());
    }

    #[test]
    fn test_no_conversion_needed() {
        let converter = SampleRateConverter::new(48000, 48000, 2);
        assert!(!converter.needs_conversion());
    }

    #[test]
    fn test_preset_converters() {
        let cd_to_broadcast = SampleRateConverter::cd_to_broadcast(2);
        assert_eq!(cd_to_broadcast.input_rate, 44100);
        assert_eq!(cd_to_broadcast.output_rate, 48000);

        let broadcast_to_hires = SampleRateConverter::broadcast_to_hires(2);
        assert_eq!(broadcast_to_hires.input_rate, 48000);
        assert_eq!(broadcast_to_hires.output_rate, 96000);
    }

    #[test]
    fn test_quality_settings() {
        let converter =
            SampleRateConverter::new(48000, 96000, 2).with_quality(ConversionQuality::Maximum);
        assert_eq!(converter.quality, ConversionQuality::Maximum);
    }

    #[test]
    fn test_validation() {
        let valid = SampleRateConverter::new(48000, 96000, 2);
        assert!(valid.validate().is_ok());

        let invalid_rate = SampleRateConverter::new(0, 96000, 2);
        assert!(matches!(
            invalid_rate.validate(),
            Err(ConvertError::InvalidSampleRate(0))
        ));

        let invalid_channels = SampleRateConverter::new(48000, 96000, 0);
        assert!(matches!(
            invalid_channels.validate(),
            Err(ConvertError::InvalidChannelCount(0))
        ));
    }
}
