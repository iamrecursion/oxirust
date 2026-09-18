//! Gamma correction and measurement.
//!
//! This module provides tools for gamma correction and gamma curve measurement.

use crate::Rgb;
use serde::{Deserialize, Serialize};

/// Gamma curve representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GammaCurve {
    /// Gamma value (typically 1.8-2.6 for displays).
    pub gamma: f64,
    /// Black point offset.
    pub black_offset: f64,
    /// White point offset.
    pub white_offset: f64,
}

impl GammaCurve {
    /// Create a new gamma curve with the given gamma value.
    #[must_use]
    pub fn new(gamma: f64) -> Self {
        Self {
            gamma,
            black_offset: 0.0,
            white_offset: 1.0,
        }
    }

    /// Create a gamma curve with black and white point offsets.
    #[must_use]
    pub fn with_offsets(gamma: f64, black_offset: f64, white_offset: f64) -> Self {
        Self {
            gamma,
            black_offset,
            white_offset,
        }
    }

    /// Apply gamma correction to a linear RGB value.
    #[must_use]
    pub fn apply(&self, rgb: &Rgb) -> Rgb {
        [
            self.apply_to_channel(rgb[0]),
            self.apply_to_channel(rgb[1]),
            self.apply_to_channel(rgb[2]),
        ]
    }

    /// Apply gamma correction to a single channel.
    fn apply_to_channel(&self, value: f64) -> f64 {
        let normalized = (value - self.black_offset) / (self.white_offset - self.black_offset);
        let corrected = normalized.clamp(0.0, 1.0).powf(1.0 / self.gamma);
        self.black_offset + corrected * (self.white_offset - self.black_offset)
    }

    /// Remove gamma correction from a gamma-corrected RGB value (linearize).
    #[must_use]
    pub fn linearize(&self, rgb: &Rgb) -> Rgb {
        [
            self.linearize_channel(rgb[0]),
            self.linearize_channel(rgb[1]),
            self.linearize_channel(rgb[2]),
        ]
    }

    /// Remove gamma correction from a single channel.
    fn linearize_channel(&self, value: f64) -> f64 {
        let normalized = (value - self.black_offset) / (self.white_offset - self.black_offset);
        let linear = normalized.clamp(0.0, 1.0).powf(self.gamma);
        self.black_offset + linear * (self.white_offset - self.black_offset)
    }
}

/// Gamma correction presets.
pub struct GammaCorrection;

impl GammaCorrection {
    /// sRGB gamma curve (piecewise, approximates gamma 2.2).
    #[must_use]
    pub fn srgb() -> GammaCurve {
        GammaCurve::new(2.2)
    }

    /// Rec.709 gamma curve (same as sRGB).
    #[must_use]
    pub fn rec709() -> GammaCurve {
        GammaCurve::new(2.2)
    }

    /// Rec.2020 gamma curve (same as sRGB).
    #[must_use]
    pub fn rec2020() -> GammaCurve {
        GammaCurve::new(2.2)
    }

    /// Apple Display (gamma 1.8).
    #[must_use]
    pub fn apple_display() -> GammaCurve {
        GammaCurve::new(1.8)
    }

    /// Standard monitor (gamma 2.2).
    #[must_use]
    pub fn standard_monitor() -> GammaCurve {
        GammaCurve::new(2.2)
    }

    /// Bright room viewing (gamma 2.4).
    #[must_use]
    pub fn bright_room() -> GammaCurve {
        GammaCurve::new(2.4)
    }

    /// Dark room viewing (gamma 2.6).
    #[must_use]
    pub fn dark_room() -> GammaCurve {
        GammaCurve::new(2.6)
    }

    /// Linear (gamma 1.0, no correction).
    #[must_use]
    pub fn linear() -> GammaCurve {
        GammaCurve::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_curve_new() {
        let curve = GammaCurve::new(2.2);
        assert!((curve.gamma - 2.2).abs() < 1e-10);
        assert!((curve.black_offset - 0.0).abs() < 1e-10);
        assert!((curve.white_offset - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_curve_with_offsets() {
        let curve = GammaCurve::with_offsets(2.2, 0.1, 0.9);
        assert!((curve.gamma - 2.2).abs() < 1e-10);
        assert!((curve.black_offset - 0.1).abs() < 1e-10);
        assert!((curve.white_offset - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_curve_apply() {
        let curve = GammaCurve::new(2.2);
        let rgb = [0.5, 0.5, 0.5];
        let corrected = curve.apply(&rgb);

        // 0.5^(1/2.2) ≈ 0.729
        assert!((corrected[0] - 0.729).abs() < 0.001);
        assert!((corrected[1] - 0.729).abs() < 0.001);
        assert!((corrected[2] - 0.729).abs() < 0.001);
    }

    #[test]
    fn test_gamma_curve_linearize() {
        let curve = GammaCurve::new(2.2);
        let rgb = [0.729, 0.729, 0.729];
        let linear = curve.linearize(&rgb);

        // 0.729^2.2 ≈ 0.5
        assert!((linear[0] - 0.5).abs() < 0.01);
        assert!((linear[1] - 0.5).abs() < 0.01);
        assert!((linear[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gamma_curve_roundtrip() {
        let curve = GammaCurve::new(2.2);
        let original = [0.3, 0.5, 0.7];
        let corrected = curve.apply(&original);
        let restored = curve.linearize(&corrected);

        assert!((restored[0] - original[0]).abs() < 1e-10);
        assert!((restored[1] - original[1]).abs() < 1e-10);
        assert!((restored[2] - original[2]).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_correction_presets() {
        assert!((GammaCorrection::srgb().gamma - 2.2).abs() < 1e-10);
        assert!((GammaCorrection::rec709().gamma - 2.2).abs() < 1e-10);
        assert!((GammaCorrection::rec2020().gamma - 2.2).abs() < 1e-10);
        assert!((GammaCorrection::apple_display().gamma - 1.8).abs() < 1e-10);
        assert!((GammaCorrection::standard_monitor().gamma - 2.2).abs() < 1e-10);
        assert!((GammaCorrection::bright_room().gamma - 2.4).abs() < 1e-10);
        assert!((GammaCorrection::dark_room().gamma - 2.6).abs() < 1e-10);
        assert!((GammaCorrection::linear().gamma - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_curve_linear() {
        let curve = GammaCurve::new(1.0);
        let rgb = [0.5, 0.6, 0.7];
        let corrected = curve.apply(&rgb);

        // With gamma 1.0, output should equal input
        assert!((corrected[0] - rgb[0]).abs() < 1e-10);
        assert!((corrected[1] - rgb[1]).abs() < 1e-10);
        assert!((corrected[2] - rgb[2]).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_curve_clamping() {
        let curve = GammaCurve::new(2.2);
        let rgb = [1.5, -0.5, 0.5];
        let corrected = curve.apply(&rgb);

        // Values should be clamped to [0.0, 1.0]
        assert!(corrected[0] >= 0.0 && corrected[0] <= 1.0);
        assert!(corrected[1] >= 0.0 && corrected[1] <= 1.0);
        assert!(corrected[2] >= 0.0 && corrected[2] <= 1.0);
    }
}
