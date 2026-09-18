//! Deblocking filter for `OxiMedia` denoise crate.
//!
//! Reduces blocking artifacts introduced by DCT-based video codecs by
//! smoothing block boundaries based on neighbouring pixel differences.

#![allow(dead_code)]

/// Represents a detected block boundary between two pixel rows or columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockBoundary {
    /// Difference between the two pixels straddling the boundary.
    pub delta: f32,
    /// Position (row or column index) in the image.
    pub position: usize,
}

impl BlockBoundary {
    /// Create a new boundary descriptor.
    pub fn new(position: usize, delta: f32) -> Self {
        Self { delta, position }
    }

    /// True if the boundary is classified as a "strong" block edge.
    /// Strong boundaries have a larger delta, suggesting codec blocking.
    pub fn is_strong(&self, strong_threshold: f32) -> bool {
        self.delta.abs() >= strong_threshold
    }
}

/// Deblocking strength level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeblockStrength {
    /// Weak filtering: only smooth subtle boundaries.
    Weak,
    /// Normal filtering: moderate smoothing.
    Normal,
    /// Strong filtering: aggressive smoothing for heavily compressed content.
    Strong,
}

impl DeblockStrength {
    /// Alpha clipping threshold for this strength level.
    pub fn alpha(self) -> f32 {
        match self {
            Self::Weak => 0.05,
            Self::Normal => 0.15,
            Self::Strong => 0.30,
        }
    }

    /// Beta threshold (inner-region flatness).
    pub fn beta(self) -> f32 {
        match self {
            Self::Weak => 0.03,
            Self::Normal => 0.08,
            Self::Strong => 0.15,
        }
    }
}

/// Configuration for the deblocking filter.
#[derive(Clone, Debug)]
pub struct DeblockConfig {
    /// Block size in pixels (typically 8 or 16).
    pub block_size: usize,
    /// Deblocking strength.
    pub strength: DeblockStrength,
    /// Whether to filter horizontal edges (between rows).
    pub filter_horizontal: bool,
    /// Whether to filter vertical edges (between columns).
    pub filter_vertical: bool,
}

impl Default for DeblockConfig {
    fn default() -> Self {
        Self {
            block_size: 8,
            strength: DeblockStrength::Normal,
            filter_horizontal: true,
            filter_vertical: true,
        }
    }
}

impl DeblockConfig {
    /// True if strong-mode filtering is active.
    pub fn is_strong_mode(&self) -> bool {
        matches!(self.strength, DeblockStrength::Strong)
    }

    /// Alpha threshold for the configured strength.
    pub fn alpha(&self) -> f32 {
        self.strength.alpha()
    }
}

/// H.264-style deblocking filter applied to planar f32 pixel data.
pub struct DeblockFilter {
    config: DeblockConfig,
}

impl DeblockFilter {
    /// Create a new filter with the given configuration.
    pub fn new(config: DeblockConfig) -> Self {
        Self { config }
    }

    /// Filter horizontal block edges in a row-major pixel buffer.
    ///
    /// Smooths the boundary between rows at multiples of `block_size`.
    pub fn filter_horizontal(&self, pixels: &mut [f32], width: usize, height: usize) {
        if !self.config.filter_horizontal {
            return;
        }
        let alpha = self.config.alpha();
        let bs = self.config.block_size;

        for row in (bs..height).step_by(bs) {
            for col in 0..width {
                let p0 = pixels[(row - 1) * width + col];
                let q0 = pixels[row * width + col];
                let diff = q0 - p0;
                if diff.abs() < alpha {
                    let correction = diff * 0.25;
                    pixels[(row - 1) * width + col] += correction;
                    pixels[row * width + col] -= correction;
                }
            }
        }
    }

    /// Filter vertical block edges in a row-major pixel buffer.
    ///
    /// Smooths the boundary between columns at multiples of `block_size`.
    pub fn filter_vertical(&self, pixels: &mut [f32], width: usize, height: usize) {
        if !self.config.filter_vertical {
            return;
        }
        let alpha = self.config.alpha();
        let bs = self.config.block_size;

        for col in (bs..width).step_by(bs) {
            for row in 0..height {
                let p0 = pixels[row * width + col - 1];
                let q0 = pixels[row * width + col];
                let diff = q0 - p0;
                if diff.abs() < alpha {
                    let correction = diff * 0.25;
                    pixels[row * width + col - 1] += correction;
                    pixels[row * width + col] -= correction;
                }
            }
        }
    }

    /// Apply both horizontal and vertical deblocking.
    pub fn apply(&self, pixels: &mut [f32], width: usize, height: usize) {
        self.filter_horizontal(pixels, width, height);
        self.filter_vertical(pixels, width, height);
    }

    /// Configuration accessor.
    pub fn config(&self) -> &DeblockConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_boundary_is_strong_above() {
        let b = BlockBoundary::new(8, 0.5);
        assert!(b.is_strong(0.3));
    }

    #[test]
    fn test_block_boundary_is_strong_below() {
        let b = BlockBoundary::new(8, 0.1);
        assert!(!b.is_strong(0.3));
    }

    #[test]
    fn test_block_boundary_negative_delta_strong() {
        let b = BlockBoundary::new(16, -0.4);
        assert!(b.is_strong(0.3)); // abs value checked
    }

    #[test]
    fn test_deblock_strength_alpha_ordering() {
        assert!(DeblockStrength::Weak.alpha() < DeblockStrength::Normal.alpha());
        assert!(DeblockStrength::Normal.alpha() < DeblockStrength::Strong.alpha());
    }

    #[test]
    fn test_deblock_strength_beta_ordering() {
        assert!(DeblockStrength::Weak.beta() < DeblockStrength::Normal.beta());
        assert!(DeblockStrength::Normal.beta() < DeblockStrength::Strong.beta());
    }

    #[test]
    fn test_deblock_strength_alpha_weak() {
        assert!((DeblockStrength::Weak.alpha() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_config_is_strong_mode_false() {
        let cfg = DeblockConfig::default();
        assert!(!cfg.is_strong_mode());
    }

    #[test]
    fn test_config_is_strong_mode_true() {
        let cfg = DeblockConfig {
            strength: DeblockStrength::Strong,
            ..Default::default()
        };
        assert!(cfg.is_strong_mode());
    }

    #[test]
    fn test_filter_horizontal_reduces_boundary_artifact() {
        let cfg = DeblockConfig::default();
        let filter = DeblockFilter::new(cfg);
        let width = 8;
        let height = 16;
        let mut pixels = vec![0.0_f32; width * height];
        // Create a step discontinuity at row boundary y=8
        for col in 0..width {
            pixels[7 * width + col] = 0.0;
            pixels[8 * width + col] = 0.1; // within alpha range
        }
        let before_diff = pixels[8 * width] - pixels[7 * width];
        filter.filter_horizontal(&mut pixels, width, height);
        let after_diff = pixels[8 * width] - pixels[7 * width];
        assert!(after_diff.abs() < before_diff.abs());
    }

    #[test]
    fn test_filter_vertical_reduces_boundary_artifact() {
        let cfg = DeblockConfig::default();
        let filter = DeblockFilter::new(cfg);
        let width = 16;
        let height = 8;
        let mut pixels = vec![0.0_f32; width * height];
        // Discontinuity at col boundary x=8
        for row in 0..height {
            pixels[row * width + 7] = 0.0;
            pixels[row * width + 8] = 0.1;
        }
        let before_diff = pixels[8] - pixels[7];
        filter.filter_vertical(&mut pixels, width, height);
        let after_diff = pixels[8] - pixels[7];
        assert!(after_diff.abs() < before_diff.abs());
    }

    #[test]
    fn test_filter_apply_no_panic() {
        let cfg = DeblockConfig::default();
        let filter = DeblockFilter::new(cfg);
        let mut pixels = vec![0.5_f32; 16 * 16];
        filter.apply(&mut pixels, 16, 16);
        assert!(pixels.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_filter_skip_horizontal_when_disabled() {
        let cfg = DeblockConfig {
            filter_horizontal: false,
            ..Default::default()
        };
        let filter = DeblockFilter::new(cfg);
        let width = 8;
        let height = 16;
        let mut pixels = vec![0.0_f32; width * height];
        pixels[8 * width] = 0.1; // boundary artifact
        let before = pixels[8 * width];
        filter.filter_horizontal(&mut pixels, width, height);
        // Should be unchanged because horizontal is disabled
        assert!((pixels[8 * width] - before).abs() < 1e-10);
    }

    #[test]
    fn test_filter_large_delta_not_filtered() {
        // Delta above alpha should NOT be smoothed (content edge, not block artifact)
        let cfg = DeblockConfig::default(); // alpha = 0.15
        let filter = DeblockFilter::new(cfg);
        let width = 8;
        let height = 16;
        let mut pixels = vec![0.0_f32; width * height];
        // Large step: 0.8 >> alpha
        for col in 0..width {
            pixels[7 * width + col] = 0.0;
            pixels[8 * width + col] = 0.8;
        }
        let before_p = pixels[7 * width];
        let before_q = pixels[8 * width];
        filter.filter_horizontal(&mut pixels, width, height);
        // Large delta → no modification
        assert!((pixels[7 * width] - before_p).abs() < 1e-10);
        assert!((pixels[8 * width] - before_q).abs() < 1e-10);
    }

    #[test]
    fn test_config_accessor() {
        let cfg = DeblockConfig {
            block_size: 16,
            ..Default::default()
        };
        let filter = DeblockFilter::new(cfg);
        assert_eq!(filter.config().block_size, 16);
    }
}
