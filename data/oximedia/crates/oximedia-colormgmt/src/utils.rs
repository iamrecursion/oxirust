//! Utility functions for color management.

use crate::xyz::Lab;

pub mod delta_e;
pub mod image;
pub mod statistics;

pub use delta_e::{cie2000, cie76, cie94, cmc};
pub use image::RgbImage;
pub use statistics::{ColorHistogram, ColorStatistics};

/// Calculates the perceptual difference between two colors in Lab space.
///
/// Uses CIE Delta E 2000 formula for accurate perceptual comparison.
///
/// # Arguments
///
/// * `color1` - First color in Lab
/// * `color2` - Second color in Lab
///
/// # Returns
///
/// Delta E value. Values < 1 are imperceptible, < 2.3 are acceptable for most applications.
#[must_use]
pub fn color_difference(color1: &Lab, color2: &Lab) -> f64 {
    crate::delta_e::delta_e_2000(color1, color2)
}

/// Checks if two colors are perceptually identical.
///
/// # Arguments
///
/// * `color1` - First color in Lab
/// * `color2` - Second color in Lab
/// * `threshold` - Delta E threshold (typically 1.0)
#[must_use]
pub fn colors_match(color1: &Lab, color2: &Lab, threshold: f64) -> bool {
    color_difference(color1, color2) < threshold
}

/// Converts RGB to grayscale using perceptual weights.
///
/// Uses Rec.709 luma coefficients: Y = 0.2126*R + 0.7152*G + 0.0722*B
///
/// # Arguments
///
/// * `rgb` - Input RGB values [0, 1]
#[must_use]
pub fn rgb_to_grayscale(rgb: [f64; 3]) -> f64 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Calculates relative luminance from RGB (Rec.709).
///
/// This is the Y component in XYZ color space.
///
/// # Arguments
///
/// * `rgb` - Linear RGB values
#[must_use]
pub fn relative_luminance(rgb: [f64; 3]) -> f64 {
    rgb_to_grayscale(rgb)
}

/// Calculates contrast ratio between two colors (WCAG formula).
///
/// # Arguments
///
/// * `rgb1` - First color in linear RGB
/// * `rgb2` - Second color in linear RGB
///
/// # Returns
///
/// Contrast ratio (1:1 to 21:1). Values >= 4.5:1 are acceptable for normal text.
#[must_use]
pub fn contrast_ratio(rgb1: [f64; 3], rgb2: [f64; 3]) -> f64 {
    let l1 = relative_luminance(rgb1);
    let l2 = relative_luminance(rgb2);

    let lighter = l1.max(l2);
    let darker = l1.min(l2);

    (lighter + 0.05) / (darker + 0.05)
}

/// Checks if color combination meets WCAG AA contrast requirements.
///
/// # Arguments
///
/// * `foreground` - Foreground color in linear RGB
/// * `background` - Background color in linear RGB
/// * `large_text` - Whether this is large text (>= 18pt or >= 14pt bold)
#[must_use]
pub fn meets_wcag_aa(foreground: [f64; 3], background: [f64; 3], large_text: bool) -> bool {
    let ratio = contrast_ratio(foreground, background);
    let threshold = if large_text { 3.0 } else { 4.5 };
    ratio >= threshold
}

/// Checks if color combination meets WCAG AAA contrast requirements.
///
/// # Arguments
///
/// * `foreground` - Foreground color in linear RGB
/// * `background` - Background color in linear RGB
/// * `large_text` - Whether this is large text (>= 18pt or >= 14pt bold)
#[must_use]
pub fn meets_wcag_aaa(foreground: [f64; 3], background: [f64; 3], large_text: bool) -> bool {
    let ratio = contrast_ratio(foreground, background);
    let threshold = if large_text { 4.5 } else { 7.0 };
    ratio >= threshold
}

/// Color temperature utilities (CCT - Correlated Color Temperature).
pub mod temperature {
    /// Converts color temperature (in Kelvin) to approximate xy chromaticity.
    ///
    /// Uses `McCamy`'s approximation formula.
    ///
    /// # Arguments
    ///
    /// * `kelvin` - Color temperature in Kelvin (1000-40000)
    ///
    /// # Returns
    ///
    /// (x, y) chromaticity coordinates
    #[must_use]
    pub fn kelvin_to_xy(kelvin: f64) -> (f64, f64) {
        let temp = kelvin.clamp(1000.0, 40_000.0);

        // Planckian locus approximation
        let x = if temp <= 7000.0 {
            -4.6070e9 / temp.powi(3) + 2.9678e6 / temp.powi(2) + 0.09911e3 / temp + 0.244063
        } else {
            -2.0064e9 / temp.powi(3) + 1.9018e6 / temp.powi(2) + 0.24748e3 / temp + 0.237040
        };

        let y = -3.0 * x * x + 2.87 * x - 0.275;

        (x, y)
    }

    /// Estimates color temperature from xy chromaticity.
    ///
    /// Uses `McCamy`'s formula (inverse).
    ///
    /// # Arguments
    ///
    /// * `x` - x chromaticity coordinate
    /// * `y` - y chromaticity coordinate
    ///
    /// # Returns
    ///
    /// Estimated color temperature in Kelvin
    #[must_use]
    pub fn xy_to_kelvin(x: f64, y: f64) -> f64 {
        let n = (x - 0.3320) / (0.1858 - y);
        437.0 * n.powi(3) + 3601.0 * n.powi(2) + 6861.0 * n + 5517.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_grayscale() {
        let white = [1.0, 1.0, 1.0];
        assert!((rgb_to_grayscale(white) - 1.0).abs() < 1e-10);

        let black = [0.0, 0.0, 0.0];
        assert!((rgb_to_grayscale(black) - 0.0).abs() < 1e-10);

        let pure_green = [0.0, 1.0, 0.0];
        assert!((rgb_to_grayscale(pure_green) - 0.7152).abs() < 1e-4);
    }

    #[test]
    fn test_contrast_ratio() {
        let white = [1.0, 1.0, 1.0];
        let black = [0.0, 0.0, 0.0];

        let ratio = contrast_ratio(white, black);
        assert!((ratio - 21.0).abs() < 0.1);

        let same = contrast_ratio(white, white);
        assert!((same - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_wcag_compliance() {
        let white = [1.0, 1.0, 1.0];
        let black = [0.0, 0.0, 0.0];

        assert!(meets_wcag_aa(black, white, false));
        assert!(meets_wcag_aa(black, white, true));
        assert!(meets_wcag_aaa(black, white, false));
        assert!(meets_wcag_aaa(black, white, true));
    }

    #[test]
    fn test_color_temperature_conversion() {
        let (x, y) = temperature::kelvin_to_xy(6500.0);
        let kelvin = temperature::xy_to_kelvin(x, y);

        // Should be close to 6500K (D65)
        assert!((kelvin - 6500.0).abs() < 100.0);
    }
}
