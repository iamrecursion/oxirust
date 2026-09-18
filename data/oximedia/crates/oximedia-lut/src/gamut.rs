//! Gamut mapping algorithms.
//!
//! This module provides algorithms for handling out-of-gamut colors by mapping them
//! into the displayable color space. This is essential when converting between
//! color spaces with different gamuts (e.g., Rec.2020 to Rec.709).
//!
//! # Algorithms
//!
//! - **Clip**: Simple hard clipping (fastest, but can cause hue shifts)
//! - **Soft Clip**: Gradual compression near gamut boundary
//! - **Desaturate**: Reduce saturation while preserving luminance
//! - **Roll-off**: Smooth compression using a curve
//! - **Adaptive**: Combination of desaturation and soft-clipping

use crate::Rgb;

/// Gamut mapping method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamutMethod {
    /// Hard clip to gamut boundaries.
    Clip,
    /// Soft clip with gradual compression.
    SoftClip,
    /// Desaturate while preserving luminance.
    Desaturate,
    /// Smooth roll-off compression.
    Rolloff,
    /// Adaptive method combining multiple techniques.
    Adaptive,
}

/// Gamut compression parameters.
#[derive(Clone, Copy, Debug)]
pub struct GamutParams {
    /// Compression threshold (0.0-1.0).
    pub threshold: f64,
    /// Compression strength (0.0-1.0).
    pub strength: f64,
    /// Preserve luminance flag.
    pub preserve_luma: bool,
}

impl Default for GamutParams {
    fn default() -> Self {
        Self {
            threshold: 0.9,
            strength: 0.5,
            preserve_luma: true,
        }
    }
}

/// Apply gamut mapping to an RGB color.
#[must_use]
pub fn map_gamut(rgb: &Rgb, method: GamutMethod, params: &GamutParams) -> Rgb {
    match method {
        GamutMethod::Clip => clip_gamut(rgb),
        GamutMethod::SoftClip => soft_clip_gamut(rgb, params),
        GamutMethod::Desaturate => desaturate_gamut(rgb, params),
        GamutMethod::Rolloff => rolloff_gamut(rgb, params),
        GamutMethod::Adaptive => adaptive_gamut(rgb, params),
    }
}

/// Hard clip RGB values to [0, 1] range.
#[must_use]
pub fn clip_gamut(rgb: &Rgb) -> Rgb {
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

/// Soft clip with gradual compression near boundaries.
#[must_use]
pub fn soft_clip_gamut(rgb: &Rgb, params: &GamutParams) -> Rgb {
    let threshold = params.threshold;
    let strength = params.strength;

    [
        soft_clip_channel(rgb[0], threshold, strength),
        soft_clip_channel(rgb[1], threshold, strength),
        soft_clip_channel(rgb[2], threshold, strength),
    ]
}

/// Soft clip a single channel.
#[must_use]
fn soft_clip_channel(value: f64, threshold: f64, strength: f64) -> f64 {
    if value < threshold {
        value
    } else if value > 1.0 {
        let excess = value - threshold;
        let compressed = excess * (1.0 - strength);
        threshold + compressed.min(1.0 - threshold)
    } else {
        // Smooth transition in threshold to 1.0 range
        let t = (value - threshold) / (1.0 - threshold);
        let compressed = t * (1.0 - strength * 0.5);
        threshold + compressed * (1.0 - threshold)
    }
}

/// Desaturate out-of-gamut colors while preserving luminance.
#[must_use]
pub fn desaturate_gamut(rgb: &Rgb, params: &GamutParams) -> Rgb {
    // Calculate how far out of gamut we are
    let max_channel = rgb[0].max(rgb[1]).max(rgb[2]);
    let min_channel = rgb[0].min(rgb[1]).min(rgb[2]);

    if max_channel <= 1.0 && min_channel >= 0.0 {
        // Already in gamut
        return *rgb;
    }

    // Calculate luminance (Rec.709 coefficients)
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

    // How much do we need to desaturate?
    let out_of_gamut = if max_channel > 1.0 {
        (max_channel - 1.0) / max_channel
    } else {
        (-min_channel) / (1.0 - min_channel)
    };

    // Blend between original and luma based on out-of-gamut amount
    let factor = (out_of_gamut * params.strength).clamp(0.0, 1.0);

    let desaturated = [
        rgb[0] * (1.0 - factor) + luma * factor,
        rgb[1] * (1.0 - factor) + luma * factor,
        rgb[2] * (1.0 - factor) + luma * factor,
    ];

    // Final clip
    clip_gamut(&desaturated)
}

/// Roll-off compression using a smooth curve.
#[must_use]
pub fn rolloff_gamut(rgb: &Rgb, params: &GamutParams) -> Rgb {
    let threshold = params.threshold;
    let strength = params.strength;

    [
        rolloff_channel(rgb[0], threshold, strength),
        rolloff_channel(rgb[1], threshold, strength),
        rolloff_channel(rgb[2], threshold, strength),
    ]
}

/// Roll-off compression for a single channel.
#[must_use]
fn rolloff_channel(value: f64, threshold: f64, strength: f64) -> f64 {
    if value < threshold {
        value
    } else {
        // Use a sigmoid-like curve for smooth compression
        let excess = value - threshold;
        let range = 1.0 - threshold;
        let compressed = excess / (1.0 + excess * strength / range);
        threshold + compressed
    }
}

/// Adaptive gamut mapping combining desaturation and soft-clipping.
#[must_use]
pub fn adaptive_gamut(rgb: &Rgb, params: &GamutParams) -> Rgb {
    // First desaturate if severely out of gamut
    let max_channel = rgb[0].max(rgb[1]).max(rgb[2]);
    let min_channel = rgb[0].min(rgb[1]).min(rgb[2]);

    let result = if max_channel > 1.2 || min_channel < -0.2 {
        // Severe out-of-gamut: desaturate first
        desaturate_gamut(rgb, params)
    } else {
        *rgb
    };

    // Then apply soft clipping for fine control
    soft_clip_gamut(&result, params)
}

/// Calculate the distance of a color from the gamut boundary.
///
/// Returns 0.0 if in gamut, positive value if out of gamut.
#[must_use]
pub fn gamut_distance(rgb: &Rgb) -> f64 {
    let max_excess = (rgb[0] - 1.0)
        .max(0.0)
        .max((rgb[1] - 1.0).max(0.0))
        .max((rgb[2] - 1.0).max(0.0));
    let min_excess = (-rgb[0])
        .max(0.0)
        .max((-rgb[1]).max(0.0))
        .max((-rgb[2]).max(0.0));
    max_excess + min_excess
}

/// Check if a color is within gamut.
#[must_use]
pub fn is_in_gamut(rgb: &Rgb) -> bool {
    rgb[0] >= 0.0
        && rgb[0] <= 1.0
        && rgb[1] >= 0.0
        && rgb[1] <= 1.0
        && rgb[2] >= 0.0
        && rgb[2] <= 1.0
}

/// Expand gamut from one color space to another.
///
/// This is the inverse of gamut compression - it expands colors to fill
/// a larger gamut (e.g., Rec.709 to Rec.2020).
#[must_use]
pub fn expand_gamut(rgb: &Rgb, expansion: f64) -> Rgb {
    // Calculate luminance
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

    // Expand away from luminance
    [
        luma + (rgb[0] - luma) * expansion,
        luma + (rgb[1] - luma) * expansion,
        luma + (rgb[2] - luma) * expansion,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_gamut() {
        let rgb = [1.5, -0.2, 0.5];
        let clipped = clip_gamut(&rgb);
        assert_eq!(clipped[0], 1.0);
        assert_eq!(clipped[1], 0.0);
        assert_eq!(clipped[2], 0.5);
    }

    #[test]
    fn test_is_in_gamut() {
        assert!(is_in_gamut(&[0.5, 0.3, 0.7]));
        assert!(!is_in_gamut(&[1.5, 0.3, 0.7]));
        assert!(!is_in_gamut(&[0.5, -0.1, 0.7]));
    }

    #[test]
    fn test_gamut_distance() {
        assert_eq!(gamut_distance(&[0.5, 0.3, 0.7]), 0.0);
        assert!((gamut_distance(&[1.5, 0.3, 0.7]) - 0.5).abs() < 1e-10);
        assert!((gamut_distance(&[0.5, -0.2, 0.7]) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_soft_clip() {
        let params = GamutParams::default();
        let rgb = [1.2, 0.5, 0.3];
        let clipped = soft_clip_gamut(&rgb, &params);
        assert!(clipped[0] <= 1.0);
        assert!(clipped[0] > rgb[0] * 0.8); // Should compress, not hard clip
    }

    #[test]
    fn test_desaturate() {
        let params = GamutParams::default();
        let rgb = [1.5, 0.5, 0.3];
        let result = desaturate_gamut(&rgb, &params);
        assert!(is_in_gamut(&result));
    }

    #[test]
    fn test_gamut_methods_preserve_in_gamut() {
        let rgb = [0.5, 0.3, 0.7];
        let params = GamutParams::default();

        for method in &[
            GamutMethod::Clip,
            GamutMethod::SoftClip,
            GamutMethod::Desaturate,
            GamutMethod::Rolloff,
            GamutMethod::Adaptive,
        ] {
            let result = map_gamut(&rgb, *method, &params);
            assert!((result[0] - rgb[0]).abs() < 0.1);
            assert!((result[1] - rgb[1]).abs() < 0.1);
            assert!((result[2] - rgb[2]).abs() < 0.1);
        }
    }
}
