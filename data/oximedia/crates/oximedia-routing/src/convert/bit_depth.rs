//! Bit depth conversion configuration.

use serde::{Deserialize, Serialize};

/// Bit depth conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitDepthConverter {
    /// Input bit depth
    pub input_depth: u8,
    /// Output bit depth
    pub output_depth: u8,
    /// Dithering configuration
    pub dither: DitherType,
    /// Number of channels
    pub channels: u8,
}

/// Dithering type for bit depth reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DitherType {
    /// No dithering
    None,
    /// Rectangular dither
    Rectangular,
    /// Triangular dither (TPDF)
    #[default]
    Triangular,
    /// Shaped dither with noise shaping
    Shaped,
}

impl BitDepthConverter {
    /// Create a new bit depth converter
    #[must_use]
    pub fn new(input_depth: u8, output_depth: u8, channels: u8) -> Self {
        Self {
            input_depth,
            output_depth,
            dither: DitherType::Triangular,
            channels,
        }
    }

    /// Set dithering type
    #[must_use]
    pub fn with_dither(mut self, dither: DitherType) -> Self {
        self.dither = dither;
        self
    }

    /// Check if conversion is needed
    #[must_use]
    pub const fn needs_conversion(&self) -> bool {
        self.input_depth != self.output_depth
    }

    /// Check if this is a reduction in bit depth
    #[must_use]
    pub const fn is_reduction(&self) -> bool {
        self.output_depth < self.input_depth
    }

    /// Check if dithering is recommended
    #[must_use]
    pub const fn should_dither(&self) -> bool {
        self.is_reduction()
    }

    /// Get bit depth change
    #[must_use]
    pub const fn depth_change(&self) -> i8 {
        self.output_depth as i8 - self.input_depth as i8
    }

    /// Create converter for 24-bit to 16-bit (with dithering)
    #[must_use]
    pub fn hires_to_cd(channels: u8) -> Self {
        Self::new(24, 16, channels).with_dither(DitherType::Triangular)
    }

    /// Create converter for 16-bit to 24-bit
    #[must_use]
    pub fn cd_to_hires(channels: u8) -> Self {
        Self::new(16, 24, channels)
    }

    /// Create converter for 32-bit float to 24-bit
    #[must_use]
    pub fn float_to_fixed(channels: u8) -> Self {
        Self::new(32, 24, channels).with_dither(DitherType::Shaped)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), super::sample_rate::ConvertError> {
        if self.input_depth == 0 || self.input_depth > 32 {
            return Err(super::sample_rate::ConvertError::InvalidBitDepth(
                self.input_depth,
            ));
        }
        if self.output_depth == 0 || self.output_depth > 32 {
            return Err(super::sample_rate::ConvertError::InvalidBitDepth(
                self.output_depth,
            ));
        }
        if self.channels == 0 {
            return Err(super::sample_rate::ConvertError::InvalidChannelCount(
                self.channels,
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
        let converter = BitDepthConverter::new(24, 16, 2);
        assert_eq!(converter.input_depth, 24);
        assert_eq!(converter.output_depth, 16);
        assert!(converter.needs_conversion());
    }

    #[test]
    fn test_is_reduction() {
        let reduction = BitDepthConverter::new(24, 16, 2);
        assert!(reduction.is_reduction());

        let expansion = BitDepthConverter::new(16, 24, 2);
        assert!(!expansion.is_reduction());
    }

    #[test]
    fn test_should_dither() {
        let reduction = BitDepthConverter::new(24, 16, 2);
        assert!(reduction.should_dither());

        let expansion = BitDepthConverter::new(16, 24, 2);
        assert!(!expansion.should_dither());
    }

    #[test]
    fn test_depth_change() {
        let reduction = BitDepthConverter::new(24, 16, 2);
        assert_eq!(reduction.depth_change(), -8);

        let expansion = BitDepthConverter::new(16, 24, 2);
        assert_eq!(expansion.depth_change(), 8);
    }

    #[test]
    fn test_preset_converters() {
        let hires_to_cd = BitDepthConverter::hires_to_cd(2);
        assert_eq!(hires_to_cd.input_depth, 24);
        assert_eq!(hires_to_cd.output_depth, 16);
        assert_eq!(hires_to_cd.dither, DitherType::Triangular);

        let cd_to_hires = BitDepthConverter::cd_to_hires(2);
        assert_eq!(cd_to_hires.input_depth, 16);
        assert_eq!(cd_to_hires.output_depth, 24);
    }

    #[test]
    fn test_dither_types() {
        let converter = BitDepthConverter::new(24, 16, 2).with_dither(DitherType::Shaped);
        assert_eq!(converter.dither, DitherType::Shaped);
    }

    #[test]
    fn test_no_conversion_needed() {
        let converter = BitDepthConverter::new(24, 24, 2);
        assert!(!converter.needs_conversion());
    }
}
