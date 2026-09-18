//! Gamut mapping strategies.
//!
//! This module provides tools for mapping colors from one gamut to another.

use crate::Rgb;

/// Gamut mapping strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamutMappingStrategy {
    /// Simple clipping to gamut boundaries.
    Clip,
    /// Perceptual mapping (preserves hue and lightness).
    Perceptual,
    /// Saturation mapping (preserves saturation).
    Saturation,
    /// Relative colorimetric (preserves color accuracy).
    RelativeColorimetric,
    /// Absolute colorimetric (preserves absolute colors).
    AbsoluteColorimetric,
}

/// Gamut mapper for mapping colors between color spaces.
pub struct GamutMapper {
    strategy: GamutMappingStrategy,
}

impl GamutMapper {
    /// Create a new gamut mapper with the given strategy.
    #[must_use]
    pub fn new(strategy: GamutMappingStrategy) -> Self {
        Self { strategy }
    }

    /// Map an RGB color to the target gamut.
    ///
    /// # Arguments
    ///
    /// * `rgb` - Input RGB color (may be out of gamut)
    ///
    /// # Returns
    ///
    /// In-gamut RGB color.
    #[must_use]
    pub fn map_color(&self, rgb: &Rgb) -> Rgb {
        match self.strategy {
            GamutMappingStrategy::Clip => self.clip(rgb),
            GamutMappingStrategy::Perceptual => self.perceptual(rgb),
            GamutMappingStrategy::Saturation => self.saturation(rgb),
            GamutMappingStrategy::RelativeColorimetric => self.relative_colorimetric(rgb),
            GamutMappingStrategy::AbsoluteColorimetric => self.absolute_colorimetric(rgb),
        }
    }

    /// Simple clip to [0, 1] range.
    fn clip(&self, rgb: &Rgb) -> Rgb {
        [
            rgb[0].clamp(0.0, 1.0),
            rgb[1].clamp(0.0, 1.0),
            rgb[2].clamp(0.0, 1.0),
        ]
    }

    /// Perceptual gamut mapping (soft-clip with saturation reduction).
    fn perceptual(&self, rgb: &Rgb) -> Rgb {
        // Check if in gamut
        if rgb[0] >= 0.0
            && rgb[0] <= 1.0
            && rgb[1] >= 0.0
            && rgb[1] <= 1.0
            && rgb[2] >= 0.0
            && rgb[2] <= 1.0
        {
            return *rgb;
        }

        // Calculate lightness and chroma
        let max_val = rgb[0].max(rgb[1]).max(rgb[2]);
        let min_val = rgb[0].min(rgb[1]).min(rgb[2]);
        let lightness = (max_val + min_val) / 2.0;

        // Reduce saturation while preserving lightness
        let scale = if max_val > 1.0 {
            1.0 / max_val
        } else if min_val < 0.0 {
            -min_val / (min_val - lightness).abs().max(0.001)
        } else {
            1.0
        };

        [
            (lightness + (rgb[0] - lightness) * scale).clamp(0.0, 1.0),
            (lightness + (rgb[1] - lightness) * scale).clamp(0.0, 1.0),
            (lightness + (rgb[2] - lightness) * scale).clamp(0.0, 1.0),
        ]
    }

    /// Saturation mapping (preserve saturation, clip lightness).
    fn saturation(&self, rgb: &Rgb) -> Rgb {
        let max_val = rgb[0].max(rgb[1]).max(rgb[2]);

        if max_val <= 1.0 && rgb[0] >= 0.0 && rgb[1] >= 0.0 && rgb[2] >= 0.0 {
            return *rgb;
        }

        let scale = if max_val > 1.0 { 1.0 / max_val } else { 1.0 };

        [
            (rgb[0] * scale).clamp(0.0, 1.0),
            (rgb[1] * scale).clamp(0.0, 1.0),
            (rgb[2] * scale).clamp(0.0, 1.0),
        ]
    }

    /// Relative colorimetric (preserve hue and lightness, clip chroma).
    fn relative_colorimetric(&self, rgb: &Rgb) -> Rgb {
        self.perceptual(rgb)
    }

    /// Absolute colorimetric (simple clip).
    fn absolute_colorimetric(&self, rgb: &Rgb) -> Rgb {
        self.clip(rgb)
    }

    /// Check if a color is within the standard RGB gamut.
    #[must_use]
    pub fn is_in_gamut(rgb: &Rgb) -> bool {
        rgb[0] >= 0.0
            && rgb[0] <= 1.0
            && rgb[1] >= 0.0
            && rgb[1] <= 1.0
            && rgb[2] >= 0.0
            && rgb[2] <= 1.0
    }

    /// Map an entire image to the target gamut.
    #[must_use]
    pub fn map_image(&self, image_data: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(image_data.len());

        for chunk in image_data.chunks_exact(3) {
            let r = f64::from(chunk[0]) / 255.0;
            let g = f64::from(chunk[1]) / 255.0;
            let b = f64::from(chunk[2]) / 255.0;

            let mapped = self.map_color(&[r, g, b]);

            output.push((mapped[0] * 255.0).round() as u8);
            output.push((mapped[1] * 255.0).round() as u8);
            output.push((mapped[2] * 255.0).round() as u8);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamut_mapper_new() {
        let mapper = GamutMapper::new(GamutMappingStrategy::Perceptual);
        assert_eq!(mapper.strategy, GamutMappingStrategy::Perceptual);
    }

    #[test]
    fn test_clip_in_gamut() {
        let mapper = GamutMapper::new(GamutMappingStrategy::Clip);
        let rgb = [0.5, 0.6, 0.7];
        let result = mapper.map_color(&rgb);

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.6).abs() < 1e-10);
        assert!((result[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_clip_out_of_gamut() {
        let mapper = GamutMapper::new(GamutMappingStrategy::Clip);
        let rgb = [1.5, -0.2, 0.7];
        let result = mapper.map_color(&rgb);

        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_is_in_gamut() {
        assert!(GamutMapper::is_in_gamut(&[0.5, 0.5, 0.5]));
        assert!(GamutMapper::is_in_gamut(&[0.0, 0.0, 0.0]));
        assert!(GamutMapper::is_in_gamut(&[1.0, 1.0, 1.0]));
        assert!(!GamutMapper::is_in_gamut(&[1.5, 0.5, 0.5]));
        assert!(!GamutMapper::is_in_gamut(&[0.5, -0.1, 0.5]));
    }

    #[test]
    fn test_map_image() {
        let mapper = GamutMapper::new(GamutMappingStrategy::Clip);
        let image = vec![128, 128, 128, 255, 0, 0];
        let output = mapper.map_image(&image);

        assert_eq!(output.len(), image.len());
    }

    #[test]
    fn test_perceptual_in_gamut() {
        let mapper = GamutMapper::new(GamutMappingStrategy::Perceptual);
        let rgb = [0.5, 0.6, 0.7];
        let result = mapper.map_color(&rgb);

        // In-gamut color should remain unchanged
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.6).abs() < 1e-10);
        assert!((result[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_saturation_mapping() {
        let mapper = GamutMapper::new(GamutMappingStrategy::Saturation);
        let rgb = [1.5, 0.9, 0.6];
        let result = mapper.map_color(&rgb);

        // Should scale down proportionally
        assert!(result[0] <= 1.0);
        assert!(result[1] <= 1.0);
        assert!(result[2] <= 1.0);
    }
}
