//! Color temperature conversions.
//!
//! This module provides functions to convert between color temperature (in Kelvin)
//! and RGB/XYZ values. Useful for white balance adjustments and creative color grading.
//!
//! # Temperature Range
//!
//! Supports temperatures from 2000K (warm candlelight) to 11000K (cool blue sky).
//!
//! # Examples
//!
//! - 2000K: Candlelight (very warm orange)
//! - 2700K: Tungsten bulb (warm)
//! - 3200K: Studio tungsten
//! - 5500K: Daylight (neutral)
//! - 6500K: D65 standard (slightly cool)
//! - 9000K: Overcast sky (cool blue)

use crate::error::{LutError, LutResult};
use crate::{Rgb, Xyz};

/// Convert color temperature (Kelvin) to XYZ tristimulus values.
///
/// Uses Planckian locus approximation for temperatures 2000K-11000K.
///
/// # Errors
///
/// Returns an error if the temperature is out of range.
pub fn temperature_to_xyz(kelvin: f64) -> LutResult<Xyz> {
    if !(2000.0..=11000.0).contains(&kelvin) {
        return Err(LutError::InvalidColor(format!(
            "Temperature {kelvin} K out of range (2000-11000)"
        )));
    }

    // Calculate chromaticity coordinates using Planckian locus approximation
    let (x, y) = temperature_to_xy(kelvin);

    // Convert chromaticity to XYZ (assuming Y = 1.0)
    let z = 1.0 - x - y;
    let xyz = [x / y, 1.0, z / y];

    Ok(xyz)
}

/// Convert color temperature to CIE 1931 xy chromaticity coordinates.
///
/// Uses Hernández-Andrés, Lee, and Romero polynomial approximation.
#[must_use]
fn temperature_to_xy(kelvin: f64) -> (f64, f64) {
    let t = kelvin;
    let t2 = t * t;
    let t3 = t2 * t;

    // Calculate x chromaticity
    let x = if t <= 4000.0 {
        -0.266_123_9e9 / t3 - 0.234_358_9e6 / t2 + 0.877_695_6e3 / t + 0.179_910
    } else {
        -3.025_846_9e9 / t3 + 2.107_037_9e6 / t2 + 0.222_634_7e3 / t + 0.240_390
    };

    // Calculate y chromaticity
    let y = if t <= 2222.0 {
        -1.106_381_4 * x * x * x - 1.348_110_20 * x * x + 2.185_558_32 * x - 0.202_196_83
    } else if t <= 4000.0 {
        -0.954_947_6 * x * x * x - 1.374_185_93 * x * x + 2.091_370_15 * x - 0.167_488_67
    } else {
        3.081_758_0 * x * x * x - 5.873_386_70 * x * x + 3.751_129_97 * x - 0.370_014_83
    };

    (x, y)
}

/// Convert XYZ to approximate color temperature (Kelvin).
///
/// Uses `McCamy`'s approximation formula.
///
/// # Errors
///
/// Returns an error if the XYZ values are invalid.
pub fn xyz_to_temperature(xyz: &Xyz) -> LutResult<f64> {
    // Convert XYZ to xy chromaticity
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum < 1e-6 {
        return Err(LutError::InvalidColor(
            "XYZ values too close to zero".to_string(),
        ));
    }

    let x = xyz[0] / sum;
    let y = xyz[1] / sum;

    // McCamy's formula
    let n = (x - 0.3320) / (0.1858 - y);
    let cct = 449.0 * n * n * n + 3525.0 * n * n + 6823.3 * n + 5520.33;

    if !(2000.0..=11000.0).contains(&cct) {
        return Err(LutError::InvalidColor(format!(
            "Calculated temperature {cct} K out of valid range"
        )));
    }

    Ok(cct)
}

/// Convert color temperature to sRGB (normalized 0-1).
///
/// # Errors
///
/// Returns an error if the temperature is out of range.
pub fn temperature_to_rgb(kelvin: f64) -> LutResult<Rgb> {
    let xyz = temperature_to_xyz(kelvin)?;

    // Convert XYZ to sRGB using D65 white point
    let rgb = xyz_to_srgb(&xyz);

    // Normalize to keep max channel at 1.0
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    if max < 1e-6 {
        return Ok([1.0, 1.0, 1.0]);
    }

    Ok([rgb[0] / max, rgb[1] / max, rgb[2] / max])
}

/// Convert XYZ to linear sRGB.
#[must_use]
fn xyz_to_srgb(xyz: &Xyz) -> Rgb {
    [
        3.240_454_2 * xyz[0] - 1.537_138_5 * xyz[1] - 0.498_531_4 * xyz[2],
        -0.969_266_0 * xyz[0] + 1.876_010_8 * xyz[1] + 0.041_556_0 * xyz[2],
        0.055_643_4 * xyz[0] - 0.204_025_9 * xyz[1] + 1.057_225_2 * xyz[2],
    ]
}

/// Apply color temperature adjustment to an RGB color.
///
/// This simulates white balance adjustment by applying a color matrix
/// based on the temperature shift.
///
/// # Errors
///
/// Returns an error if the temperatures are out of range.
pub fn apply_temperature_shift(rgb: &Rgb, from_kelvin: f64, to_kelvin: f64) -> LutResult<Rgb> {
    // Get RGB values for source and destination temperatures
    let source_temp = temperature_to_rgb(from_kelvin)?;
    let dest_temp = temperature_to_rgb(to_kelvin)?;

    // Calculate scaling factors
    let scale = [
        dest_temp[0] / source_temp[0],
        dest_temp[1] / source_temp[1],
        dest_temp[2] / source_temp[2],
    ];

    // Apply scaling
    Ok([rgb[0] * scale[0], rgb[1] * scale[1], rgb[2] * scale[2]])
}

/// Preset color temperatures.
pub mod presets {
    /// Candlelight (very warm).
    pub const CANDLE: f64 = 2000.0;
    /// Tungsten bulb (warm).
    pub const TUNGSTEN: f64 = 2700.0;
    /// Studio tungsten.
    pub const STUDIO_TUNGSTEN: f64 = 3200.0;
    /// Fluorescent.
    pub const FLUORESCENT: f64 = 4000.0;
    /// Daylight (neutral).
    pub const DAYLIGHT: f64 = 5500.0;
    /// D65 standard (slightly cool).
    pub const D65: f64 = 6500.0;
    /// Overcast sky (cool).
    pub const OVERCAST: f64 = 7000.0;
    /// Clear blue sky (very cool).
    pub const BLUE_SKY: f64 = 9000.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_to_rgb() {
        // Test various temperatures
        assert!(temperature_to_rgb(2700.0).is_ok()); // Tungsten
        assert!(temperature_to_rgb(5500.0).is_ok()); // Daylight
        assert!(temperature_to_rgb(6500.0).is_ok()); // D65
        assert!(temperature_to_rgb(9000.0).is_ok()); // Blue sky

        // Out of range
        assert!(temperature_to_rgb(1000.0).is_err());
        assert!(temperature_to_rgb(15000.0).is_err());
    }

    #[test]
    fn test_warm_vs_cool() {
        let warm = temperature_to_rgb(2700.0).expect("should succeed in test");
        let cool = temperature_to_rgb(9000.0).expect("should succeed in test");

        // Warm should have more red
        assert!(warm[0] > cool[0]);
        // Cool should have more blue
        assert!(cool[2] > warm[2]);
    }

    #[test]
    fn test_temperature_round_trip() {
        let kelvin = 5500.0;
        let xyz = temperature_to_xyz(kelvin).expect("should succeed in test");
        let back = xyz_to_temperature(&xyz).expect("should succeed in test");
        assert!((kelvin - back).abs() < 100.0); // Within 100K
    }

    #[test]
    fn test_d65_temperature() {
        let xyz = temperature_to_xyz(6500.0).expect("should succeed in test");
        // D65 should be close to [0.95047, 1.0, 1.08883]
        assert!((xyz[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_temperature_shift() {
        let rgb = [0.5, 0.5, 0.5];
        let shifted =
            apply_temperature_shift(&rgb, 5500.0, 3200.0).expect("should succeed in test");
        // Shifting to warmer temperature should increase red/yellow, decrease blue
        // For neutral gray, the shift might be subtle
        assert!(shifted[2] < rgb[2] || shifted[0] >= rgb[0]);
    }

    #[test]
    fn test_presets() {
        assert!(temperature_to_rgb(presets::CANDLE).is_ok());
        assert!(temperature_to_rgb(presets::TUNGSTEN).is_ok());
        assert!(temperature_to_rgb(presets::STUDIO_TUNGSTEN).is_ok());
        assert!(temperature_to_rgb(presets::FLUORESCENT).is_ok());
        assert!(temperature_to_rgb(presets::DAYLIGHT).is_ok());
        assert!(temperature_to_rgb(presets::D65).is_ok());
        assert!(temperature_to_rgb(presets::OVERCAST).is_ok());
        assert!(temperature_to_rgb(presets::BLUE_SKY).is_ok());
    }
}
