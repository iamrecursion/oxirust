//! Color temperature shift and conversion.
//!
//! This module provides tools for shifting color temperature and converting
//! temperature values to RGB multipliers.

use crate::Rgb;

/// Convert color temperature (in Kelvin) to RGB multipliers.
///
/// Uses Tanner Helland's approximation algorithm.
///
/// # Arguments
///
/// * `temperature` - Color temperature in Kelvin (1000-40000)
///
/// # Returns
///
/// RGB multipliers for the given temperature.
#[must_use]
pub fn temperature_to_rgb(temperature: u32) -> Rgb {
    let temp = (temperature as f64 / 100.0).clamp(10.0, 400.0);

    // Calculate red
    let red = if temp <= 66.0 {
        1.0
    } else {
        let r = temp - 60.0;
        (329.698_727_446 * r.powf(-0.133_204_759_2)) / 255.0
    };

    // Calculate green
    let green = if temp <= 66.0 {
        let g = temp;
        (99.470_802_586_1 * g.ln() - 161.119_568_166_1) / 255.0
    } else {
        let g = temp - 60.0;
        (288.122_169_53 * g.powf(-0.075_514_849_2)) / 255.0
    };

    // Calculate blue
    let blue = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        let b = temp - 10.0;
        (138.517_731_223_1 * b.ln() - 305.044_792_730_7) / 255.0
    };

    [
        red.clamp(0.0, 1.0),
        green.clamp(0.0, 1.0),
        blue.clamp(0.0, 1.0),
    ]
}

/// Apply a color temperature shift to an RGB color.
///
/// # Arguments
///
/// * `rgb` - Input RGB color
/// * `from_temp` - Current color temperature in Kelvin
/// * `to_temp` - Target color temperature in Kelvin
///
/// # Returns
///
/// Color-corrected RGB value.
#[must_use]
pub fn apply_temperature_shift(rgb: &Rgb, from_temp: u32, to_temp: u32) -> Rgb {
    let from_rgb = temperature_to_rgb(from_temp);
    let to_rgb = temperature_to_rgb(to_temp);

    // Calculate multipliers
    let r_mult = to_rgb[0] / from_rgb[0].max(0.001);
    let g_mult = to_rgb[1] / from_rgb[1].max(0.001);
    let b_mult = to_rgb[2] / from_rgb[2].max(0.001);

    [
        (rgb[0] * r_mult).clamp(0.0, 1.0),
        (rgb[1] * g_mult).clamp(0.0, 1.0),
        (rgb[2] * b_mult).clamp(0.0, 1.0),
    ]
}

/// Apply a color temperature shift to an entire image.
///
/// # Arguments
///
/// * `image_data` - Raw image data (RGB format)
/// * `from_temp` - Current color temperature in Kelvin
/// * `to_temp` - Target color temperature in Kelvin
///
/// # Returns
///
/// Temperature-corrected image data.
#[must_use]
pub fn apply_temperature_shift_to_image(
    image_data: &[u8],
    from_temp: u32,
    to_temp: u32,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(image_data.len());

    for chunk in image_data.chunks_exact(3) {
        let r = f64::from(chunk[0]) / 255.0;
        let g = f64::from(chunk[1]) / 255.0;
        let b = f64::from(chunk[2]) / 255.0;

        let shifted = apply_temperature_shift(&[r, g, b], from_temp, to_temp);

        output.push((shifted[0] * 255.0).round() as u8);
        output.push((shifted[1] * 255.0).round() as u8);
        output.push((shifted[2] * 255.0).round() as u8);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_to_rgb_daylight() {
        let rgb = temperature_to_rgb(6500); // D65 daylight

        // At 6500K, should be close to neutral white
        assert!(rgb[0] > 0.9);
        assert!(rgb[1] > 0.9);
        assert!(rgb[2] > 0.9);
    }

    #[test]
    fn test_temperature_to_rgb_tungsten() {
        let rgb = temperature_to_rgb(3000); // Tungsten

        // At 3000K, should be warm (more red, less blue)
        assert!(rgb[0] > rgb[2]);
    }

    #[test]
    fn test_temperature_to_rgb_cool() {
        let rgb = temperature_to_rgb(9000); // Cool/shade

        // At 9000K, should be cool (more blue, less red)
        assert!(rgb[2] > rgb[0]);
    }

    #[test]
    fn test_temperature_to_rgb_range() {
        // Test various temperatures
        for temp in (2000..=10000).step_by(1000) {
            let rgb = temperature_to_rgb(temp);

            // All values should be in valid range
            assert!(rgb[0] >= 0.0 && rgb[0] <= 1.0);
            assert!(rgb[1] >= 0.0 && rgb[1] <= 1.0);
            assert!(rgb[2] >= 0.0 && rgb[2] <= 1.0);
        }
    }

    #[test]
    fn test_apply_temperature_shift_same() {
        let rgb = [0.5, 0.5, 0.5];
        let result = apply_temperature_shift(&rgb, 6500, 6500);

        // No shift should result in same color
        assert!((result[0] - rgb[0]).abs() < 0.1);
        assert!((result[1] - rgb[1]).abs() < 0.1);
        assert!((result[2] - rgb[2]).abs() < 0.1);
    }

    #[test]
    fn test_apply_temperature_shift_warm_to_cool() {
        let rgb = [0.5, 0.5, 0.5];
        let result = apply_temperature_shift(&rgb, 3000, 6500);

        // Shifting from warm to cool should increase blue relative to red
        assert!(result[2] > result[0]);
    }

    #[test]
    fn test_apply_temperature_shift_to_image() {
        let image = vec![128, 128, 128, 255, 0, 0];
        let output = apply_temperature_shift_to_image(&image, 6500, 6500);

        assert_eq!(output.len(), image.len());
    }
}
