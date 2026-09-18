//! Dolby Vision enhancement layer structures and utilities.

#![allow(dead_code)]

/// Represents a Dolby Vision enhancement layer applied on top of the base layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancementLayer {
    /// Width of the enhancement layer in pixels.
    pub width: u32,
    /// Height of the enhancement layer in pixels.
    pub height: u32,
    /// Enhancement factor (0.0 = passthrough / inactive).
    pub enhancement_factor: f32,
}

impl EnhancementLayer {
    /// Create a new enhancement layer.
    #[must_use]
    pub fn new(width: u32, height: u32, enhancement_factor: f32) -> Self {
        Self {
            width,
            height,
            enhancement_factor,
        }
    }

    /// Returns `true` if the enhancement layer is active (factor > 0.0).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.enhancement_factor > 0.0
    }
}

// ---------------------------------------------------------------------------

/// Describes a resolution enhancement relationship between a base and enhanced image.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolutionEnhancement {
    /// Width of the base (standard) layer.
    pub base_width: u32,
    /// Height of the base (standard) layer.
    pub base_height: u32,
    /// Width of the enhanced output layer.
    pub enhanced_width: u32,
    /// Height of the enhanced output layer.
    pub enhanced_height: u32,
}

impl ResolutionEnhancement {
    /// Create a new resolution enhancement descriptor.
    #[must_use]
    pub fn new(
        base_width: u32,
        base_height: u32,
        enhanced_width: u32,
        enhanced_height: u32,
    ) -> Self {
        Self {
            base_width,
            base_height,
            enhanced_width,
            enhanced_height,
        }
    }

    /// Horizontal scale factor (enhanced / base).
    ///
    /// Returns 0.0 if `base_width` is zero to avoid division by zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn scale_factor_x(&self) -> f32 {
        if self.base_width == 0 {
            return 0.0;
        }
        self.enhanced_width as f32 / self.base_width as f32
    }

    /// Vertical scale factor (enhanced / base).
    ///
    /// Returns 0.0 if `base_height` is zero to avoid division by zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn scale_factor_y(&self) -> f32 {
        if self.base_height == 0 {
            return 0.0;
        }
        self.enhanced_height as f32 / self.base_height as f32
    }

    /// Combined upscale ratio (geometric mean of x and y factors).
    #[must_use]
    pub fn upscale_ratio(&self) -> f32 {
        let sx = self.scale_factor_x();
        let sy = self.scale_factor_y();
        (sx * sy).sqrt()
    }
}

// ---------------------------------------------------------------------------

/// Configuration for the Dolby Vision enhancement processing pass.
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancementConfig {
    /// Spatial scaling strength (0.0 = no spatial enhancement).
    pub spatial_scale: f32,
    /// Temporal filter strength (0.0 = no temporal filtering).
    pub temporal_filter: f32,
    /// Sharpness boost amount (0.0 = no sharpening).
    pub sharpness_boost: f32,
}

impl EnhancementConfig {
    /// Create a new enhancement configuration.
    #[must_use]
    pub fn new(spatial_scale: f32, temporal_filter: f32, sharpness_boost: f32) -> Self {
        Self {
            spatial_scale,
            temporal_filter,
            sharpness_boost,
        }
    }

    /// Returns `true` if this configuration is a pure passthrough (all values are 0).
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.spatial_scale == 0.0 && self.temporal_filter == 0.0 && self.sharpness_boost == 0.0
    }
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- EnhancementLayer ---

    #[test]
    fn test_enhancement_layer_active_positive() {
        let layer = EnhancementLayer::new(1920, 1080, 1.5);
        assert!(layer.is_active());
    }

    #[test]
    fn test_enhancement_layer_inactive_zero() {
        let layer = EnhancementLayer::new(1920, 1080, 0.0);
        assert!(!layer.is_active());
    }

    #[test]
    fn test_enhancement_layer_inactive_negative() {
        // Negative factors are treated as inactive.
        let layer = EnhancementLayer::new(1920, 1080, -0.5);
        assert!(!layer.is_active());
    }

    #[test]
    fn test_enhancement_layer_fields() {
        let layer = EnhancementLayer::new(3840, 2160, 2.0);
        assert_eq!(layer.width, 3840);
        assert_eq!(layer.height, 2160);
        assert_eq!(layer.enhancement_factor, 2.0);
    }

    // --- ResolutionEnhancement ---

    #[test]
    fn test_scale_factor_x_2x() {
        let re = ResolutionEnhancement::new(1920, 1080, 3840, 2160);
        assert!((re.scale_factor_x() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale_factor_y_2x() {
        let re = ResolutionEnhancement::new(1920, 1080, 3840, 2160);
        assert!((re.scale_factor_y() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_upscale_ratio_2x() {
        let re = ResolutionEnhancement::new(1920, 1080, 3840, 2160);
        assert!((re.upscale_ratio() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_scale_factor_x_zero_base() {
        let re = ResolutionEnhancement::new(0, 1080, 3840, 2160);
        assert_eq!(re.scale_factor_x(), 0.0);
    }

    #[test]
    fn test_scale_factor_y_zero_base() {
        let re = ResolutionEnhancement::new(1920, 0, 3840, 2160);
        assert_eq!(re.scale_factor_y(), 0.0);
    }

    #[test]
    fn test_scale_factor_asymmetric() {
        let re = ResolutionEnhancement::new(1920, 1080, 3840, 1080);
        assert!((re.scale_factor_x() - 2.0).abs() < 1e-5);
        assert!((re.scale_factor_y() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_upscale_ratio_asymmetric() {
        let re = ResolutionEnhancement::new(1920, 1080, 3840, 1080);
        // geometric mean of 2.0 and 1.0 = sqrt(2.0)
        let expected = 2.0_f32.sqrt();
        assert!((re.upscale_ratio() - expected).abs() < 1e-4);
    }

    // --- EnhancementConfig ---

    #[test]
    fn test_enhancement_config_passthrough() {
        let cfg = EnhancementConfig::new(0.0, 0.0, 0.0);
        assert!(cfg.is_passthrough());
    }

    #[test]
    fn test_enhancement_config_not_passthrough_spatial() {
        let cfg = EnhancementConfig::new(0.5, 0.0, 0.0);
        assert!(!cfg.is_passthrough());
    }

    #[test]
    fn test_enhancement_config_not_passthrough_temporal() {
        let cfg = EnhancementConfig::new(0.0, 0.3, 0.0);
        assert!(!cfg.is_passthrough());
    }

    #[test]
    fn test_enhancement_config_not_passthrough_sharpness() {
        let cfg = EnhancementConfig::new(0.0, 0.0, 0.8);
        assert!(!cfg.is_passthrough());
    }

    #[test]
    fn test_enhancement_config_default_is_passthrough() {
        let cfg = EnhancementConfig::default();
        assert!(cfg.is_passthrough());
    }

    #[test]
    fn test_enhancement_config_fields() {
        let cfg = EnhancementConfig::new(1.0, 0.5, 0.25);
        assert_eq!(cfg.spatial_scale, 1.0);
        assert_eq!(cfg.temporal_filter, 0.5);
        assert_eq!(cfg.sharpness_boost, 0.25);
    }
}
