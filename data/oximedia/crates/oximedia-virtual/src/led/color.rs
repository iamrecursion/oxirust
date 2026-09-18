//! LED wall color matching and calibration
//!
//! Provides color space conversion and matching between cameras
//! and LED walls for accurate color reproduction.

use crate::Result;
use serde::{Deserialize, Serialize};

/// Color space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// sRGB
    Srgb,
    /// Rec.709
    Rec709,
    /// Rec.2020
    Rec2020,
    /// DCI-P3
    DciP3,
    /// Adobe RGB
    AdobeRgb,
}

/// Color temperature in Kelvin
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColorTemperature(pub f32);

impl ColorTemperature {
    /// Standard D65 illuminant (6500K)
    pub const D65: Self = Self(6500.0);

    /// Standard D50 illuminant (5000K)
    pub const D50: Self = Self(5000.0);

    /// Tungsten lighting (3200K)
    pub const TUNGSTEN: Self = Self(3200.0);
}

/// White point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WhitePoint {
    /// X chromaticity coordinate
    pub x: f32,
    /// Y chromaticity coordinate
    pub y: f32,
    /// Luminance
    pub luminance: f32,
}

impl WhitePoint {
    /// D65 white point
    pub const D65: Self = Self {
        x: 0.3127,
        y: 0.3290,
        luminance: 1.0,
    };

    /// D50 white point
    pub const D50: Self = Self {
        x: 0.3457,
        y: 0.3585,
        luminance: 1.0,
    };
}

/// Color matching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorMatchConfig {
    /// Source color space (camera)
    pub source_space: ColorSpace,
    /// Target color space (LED wall)
    pub target_space: ColorSpace,
    /// Source white point
    pub source_white: WhitePoint,
    /// Target white point
    pub target_white: WhitePoint,
    /// Enable chromatic adaptation
    pub chromatic_adaptation: bool,
}

impl Default for ColorMatchConfig {
    fn default() -> Self {
        Self {
            source_space: ColorSpace::Rec709,
            target_space: ColorSpace::Rec709,
            source_white: WhitePoint::D65,
            target_white: WhitePoint::D65,
            chromatic_adaptation: true,
        }
    }
}

/// LED color matcher
pub struct LedColorMatcher {
    config: ColorMatchConfig,
    transform_matrix: [[f32; 3]; 3],
}

impl LedColorMatcher {
    /// Create new color matcher
    pub fn new(config: ColorMatchConfig) -> Result<Self> {
        let transform_matrix = Self::compute_transform_matrix(&config)?;

        Ok(Self {
            config,
            transform_matrix,
        })
    }

    /// Compute color space transformation matrix
    fn compute_transform_matrix(config: &ColorMatchConfig) -> Result<[[f32; 3]; 3]> {
        // Get primary matrices for source and target
        let source_primaries = Self::get_primaries(config.source_space);
        let target_primaries = Self::get_primaries(config.target_space);

        // Compute transformation matrix
        // For simplicity, using identity when spaces match
        if config.source_space == config.target_space {
            Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        } else {
            // In a real implementation, this would compute the proper matrix
            // using the XYZ connection space
            Ok(Self::bradford_adaptation(
                &source_primaries,
                &target_primaries,
            ))
        }
    }

    /// Get RGB primaries for color space
    fn get_primaries(space: ColorSpace) -> [[f32; 3]; 3] {
        match space {
            ColorSpace::Srgb | ColorSpace::Rec709 => [
                [0.64, 0.33, 0.03], // Red
                [0.30, 0.60, 0.10], // Green
                [0.15, 0.06, 0.79], // Blue
            ],
            ColorSpace::Rec2020 => [
                [0.708, 0.292, 0.000],
                [0.170, 0.797, 0.033],
                [0.131, 0.046, 0.823],
            ],
            ColorSpace::DciP3 => [
                [0.680, 0.320, 0.000],
                [0.265, 0.690, 0.045],
                [0.150, 0.060, 0.790],
            ],
            ColorSpace::AdobeRgb => [[0.64, 0.33, 0.03], [0.21, 0.71, 0.08], [0.15, 0.06, 0.79]],
        }
    }

    /// Bradford chromatic adaptation
    fn bradford_adaptation(_source: &[[f32; 3]; 3], _target: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
        // Simplified Bradford adaptation
        // Real implementation would use proper XYZ transformation
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    /// Transform RGB color from source to target space
    #[must_use]
    pub fn transform_color(&self, rgb: [f32; 3]) -> [f32; 3] {
        let r = rgb[0] * self.transform_matrix[0][0]
            + rgb[1] * self.transform_matrix[0][1]
            + rgb[2] * self.transform_matrix[0][2];

        let g = rgb[0] * self.transform_matrix[1][0]
            + rgb[1] * self.transform_matrix[1][1]
            + rgb[2] * self.transform_matrix[1][2];

        let b = rgb[0] * self.transform_matrix[2][0]
            + rgb[1] * self.transform_matrix[2][1]
            + rgb[2] * self.transform_matrix[2][2];

        [
            r.max(0.0).min(1.0),
            g.max(0.0).min(1.0),
            b.max(0.0).min(1.0),
        ]
    }

    /// Transform RGB color (8-bit)
    #[must_use]
    pub fn transform_color_u8(&self, rgb: [u8; 3]) -> [u8; 3] {
        let normalized = [
            f32::from(rgb[0]) / 255.0,
            f32::from(rgb[1]) / 255.0,
            f32::from(rgb[2]) / 255.0,
        ];

        let transformed = self.transform_color(normalized);

        [
            (transformed[0] * 255.0) as u8,
            (transformed[1] * 255.0) as u8,
            (transformed[2] * 255.0) as u8,
        ]
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ColorMatchConfig) -> Result<()> {
        self.transform_matrix = Self::compute_transform_matrix(&config)?;
        self.config = config;
        Ok(())
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &ColorMatchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_temperature() {
        let temp = ColorTemperature::D65;
        assert_eq!(temp.0, 6500.0);
    }

    #[test]
    fn test_white_point() {
        let wp = WhitePoint::D65;
        assert!((wp.x - 0.3127).abs() < 1e-4);
    }

    #[test]
    fn test_color_matcher_creation() {
        let config = ColorMatchConfig::default();
        let matcher = LedColorMatcher::new(config);
        assert!(matcher.is_ok());
    }

    #[test]
    fn test_color_transform_identity() {
        let config = ColorMatchConfig::default();
        let matcher = LedColorMatcher::new(config).expect("should succeed in test");

        let input = [1.0, 0.5, 0.25];
        let output = matcher.transform_color(input);

        // Should be identity when source == target
        assert!((output[0] - input[0]).abs() < 1e-4);
        assert!((output[1] - input[1]).abs() < 1e-4);
        assert!((output[2] - input[2]).abs() < 1e-4);
    }

    #[test]
    fn test_color_transform_u8() {
        let config = ColorMatchConfig::default();
        let matcher = LedColorMatcher::new(config).expect("should succeed in test");

        let input = [255, 128, 64];
        let output = matcher.transform_color_u8(input);

        assert_eq!(output[0], 255);
        assert_eq!(output[1], 128);
        assert_eq!(output[2], 64);
    }

    #[test]
    fn test_color_spaces() {
        let rec709 = LedColorMatcher::get_primaries(ColorSpace::Rec709);
        let rec2020 = LedColorMatcher::get_primaries(ColorSpace::Rec2020);

        // Primaries should be different
        assert_ne!(rec709[0], rec2020[0]);
    }
}
