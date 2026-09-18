#![allow(dead_code)]
//! Drop shadow effect generation for broadcast graphics elements.
//!
//! Provides configurable drop shadow rendering including offset, blur radius,
//! color, opacity, and spread controls. Suitable for lower thirds, titles,
//! and other broadcast overlay elements that need visual separation from
//! the background.

/// RGBA color for shadow rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowColor {
    /// Red channel [0..255].
    pub r: u8,
    /// Green channel [0..255].
    pub g: u8,
    /// Blue channel [0..255].
    pub b: u8,
    /// Alpha channel [0..255].
    pub a: u8,
}

impl ShadowColor {
    /// Create a new shadow color.
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Solid black shadow.
    pub fn black() -> Self {
        Self::new(0, 0, 0, 255)
    }

    /// Semi-transparent black shadow (common default).
    pub fn default_shadow() -> Self {
        Self::new(0, 0, 0, 128)
    }

    /// Convert to normalized f64 RGBA tuple.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_normalized(&self) -> (f64, f64, f64, f64) {
        (
            self.r as f64 / 255.0,
            self.g as f64 / 255.0,
            self.b as f64 / 255.0,
            self.a as f64 / 255.0,
        )
    }

    /// Create from normalized f64 values (each in [0.0, 1.0]).
    pub fn from_normalized(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self {
            r: (r.clamp(0.0, 1.0) * 255.0) as u8,
            g: (g.clamp(0.0, 1.0) * 255.0) as u8,
            b: (b.clamp(0.0, 1.0) * 255.0) as u8,
            a: (a.clamp(0.0, 1.0) * 255.0) as u8,
        }
    }

    /// Pre-multiply alpha into the color channels.
    #[allow(clippy::cast_precision_loss)]
    pub fn premultiply(&self) -> (f64, f64, f64, f64) {
        let (r, g, b, a) = self.to_normalized();
        (r * a, g * a, b * a, a)
    }
}

impl Default for ShadowColor {
    fn default() -> Self {
        Self::default_shadow()
    }
}

/// Configuration for a drop shadow effect.
#[derive(Clone, Debug)]
pub struct DropShadowConfig {
    /// Horizontal offset in pixels (positive = right).
    pub offset_x: f64,
    /// Vertical offset in pixels (positive = down).
    pub offset_y: f64,
    /// Blur radius in pixels (standard deviation for Gaussian).
    pub blur_radius: f64,
    /// Spread radius in pixels (grows/shrinks the shadow shape).
    pub spread: f64,
    /// Shadow color.
    pub color: ShadowColor,
    /// Whether the shadow is inset (inside the element).
    pub inset: bool,
}

impl DropShadowConfig {
    /// Create a new drop shadow configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the offset.
    pub fn with_offset(mut self, x: f64, y: f64) -> Self {
        self.offset_x = x;
        self.offset_y = y;
        self
    }

    /// Set the blur radius.
    pub fn with_blur(mut self, radius: f64) -> Self {
        self.blur_radius = radius.max(0.0);
        self
    }

    /// Set the spread.
    pub fn with_spread(mut self, spread: f64) -> Self {
        self.spread = spread;
        self
    }

    /// Set the color.
    pub fn with_color(mut self, color: ShadowColor) -> Self {
        self.color = color;
        self
    }

    /// Set as inset shadow.
    pub fn with_inset(mut self, inset: bool) -> Self {
        self.inset = inset;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.blur_radius < 0.0 {
            return Err("Blur radius cannot be negative".to_string());
        }
        if self.blur_radius > 500.0 {
            return Err("Blur radius exceeds maximum of 500 pixels".to_string());
        }
        Ok(())
    }

    /// Compute the total extent of the shadow beyond the original element bounds.
    pub fn extent(&self) -> ShadowExtent {
        let blur_extent = self.blur_radius * 3.0; // 3-sigma covers ~99.7%
        let total = blur_extent + self.spread;
        ShadowExtent {
            left: (-self.offset_x + total).max(0.0),
            right: (self.offset_x + total).max(0.0),
            top: (-self.offset_y + total).max(0.0),
            bottom: (self.offset_y + total).max(0.0),
        }
    }
}

impl Default for DropShadowConfig {
    fn default() -> Self {
        Self {
            offset_x: 4.0,
            offset_y: 4.0,
            blur_radius: 8.0,
            spread: 0.0,
            color: ShadowColor::default(),
            inset: false,
        }
    }
}

/// The extent of a shadow beyond the element bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowExtent {
    /// Pixels extending to the left.
    pub left: f64,
    /// Pixels extending to the right.
    pub right: f64,
    /// Pixels extending upward.
    pub top: f64,
    /// Pixels extending downward.
    pub bottom: f64,
}

impl ShadowExtent {
    /// Total additional width.
    pub fn total_width(&self) -> f64 {
        self.left + self.right
    }

    /// Total additional height.
    pub fn total_height(&self) -> f64 {
        self.top + self.bottom
    }
}

/// A 1D Gaussian kernel for blur operations.
#[derive(Clone, Debug)]
pub struct GaussianKernel {
    /// Kernel weights (normalized, symmetric).
    weights: Vec<f64>,
    /// Radius (half-width, not counting center).
    radius: usize,
}

impl GaussianKernel {
    /// Create a Gaussian kernel with the given standard deviation.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(sigma: f64) -> Self {
        if sigma <= 0.0 {
            return Self {
                weights: vec![1.0],
                radius: 0,
            };
        }
        let radius = (sigma * 3.0).ceil() as usize;
        let size = 2 * radius + 1;
        let mut weights = Vec::with_capacity(size);
        let two_sigma_sq = 2.0 * sigma * sigma;
        let mut sum = 0.0;
        for i in 0..size {
            let x = i as f64 - radius as f64;
            let w = (-x * x / two_sigma_sq).exp();
            weights.push(w);
            sum += w;
        }
        // Normalize
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }
        Self { weights, radius }
    }

    /// Get the kernel weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the kernel radius.
    pub fn radius(&self) -> usize {
        self.radius
    }

    /// Get the kernel size.
    pub fn size(&self) -> usize {
        self.weights.len()
    }

    /// Apply the kernel to a 1D signal at position `center`.
    pub fn apply_1d(&self, signal: &[f64], center: usize) -> f64 {
        let mut result = 0.0;
        let len = signal.len();
        for (i, &w) in self.weights.iter().enumerate() {
            let idx = center as isize + i as isize - self.radius as isize;
            let clamped = idx.clamp(0, (len as isize) - 1) as usize;
            result += signal[clamped] * w;
        }
        result
    }
}

/// Generate a shadow alpha mask for a rectangular element.
///
/// Returns a 2D grid of alpha values [0.0, 1.0] representing the blurred
/// shadow for a rectangle of given dimensions.
#[allow(clippy::cast_precision_loss)]
pub fn generate_rect_shadow(
    width: usize,
    height: usize,
    config: &DropShadowConfig,
) -> Vec<Vec<f64>> {
    let extent = config.extent();
    let out_w = width + extent.total_width().ceil() as usize;
    let out_h = height + extent.total_height().ceil() as usize;

    // Create the source mask (rectangle at offset)
    let ox = (extent.left + config.offset_x) as isize;
    let oy = (extent.top + config.offset_y) as isize;

    let mut mask = vec![vec![0.0_f64; out_w]; out_h];
    for row in 0..height {
        for col in 0..width {
            let r = row as isize + oy;
            let c = col as isize + ox;
            if r >= 0 && (r as usize) < out_h && c >= 0 && (c as usize) < out_w {
                mask[r as usize][c as usize] = 1.0;
            }
        }
    }

    // Apply Gaussian blur (separable: horizontal then vertical)
    let kernel = GaussianKernel::new(config.blur_radius);
    // Horizontal pass
    let mut h_blur = vec![vec![0.0_f64; out_w]; out_h];
    for row in 0..out_h {
        for col in 0..out_w {
            h_blur[row][col] = kernel.apply_1d(&mask[row], col);
        }
    }
    // Vertical pass
    let mut result = vec![vec![0.0_f64; out_w]; out_h];
    for col in 0..out_w {
        let column: Vec<f64> = (0..out_h).map(|r| h_blur[r][col]).collect();
        for row in 0..out_h {
            result[row][col] = kernel.apply_1d(&column, row);
        }
    }

    // Apply shadow color alpha
    let (_, _, _, a) = config.color.to_normalized();
    for row in &mut result {
        for val in row.iter_mut() {
            *val *= a;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_color_black() {
        let c = ShadowColor::black();
        assert_eq!(c.r, 0);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_shadow_color_default() {
        let c = ShadowColor::default();
        assert_eq!(c.a, 128);
    }

    #[test]
    fn test_shadow_color_normalized_roundtrip() {
        let c = ShadowColor::new(128, 64, 32, 255);
        let (r, g, b, a) = c.to_normalized();
        let c2 = ShadowColor::from_normalized(r, g, b, a);
        assert_eq!(c.r, c2.r);
        assert_eq!(c.g, c2.g);
        assert_eq!(c.b, c2.b);
        assert_eq!(c.a, c2.a);
    }

    #[test]
    fn test_shadow_color_premultiply() {
        let c = ShadowColor::new(255, 0, 0, 128);
        let (r, _g, _b, a) = c.premultiply();
        // Alpha ~0.502, red ~0.502
        assert!(a > 0.49 && a < 0.51);
        assert!(r > 0.49 && r < 0.51);
    }

    #[test]
    fn test_config_defaults() {
        let config = DropShadowConfig::new();
        assert!((config.offset_x - 4.0).abs() < 1e-12);
        assert!((config.blur_radius - 8.0).abs() < 1e-12);
        assert!(!config.inset);
    }

    #[test]
    fn test_config_builder() {
        let config = DropShadowConfig::new()
            .with_offset(10.0, 12.0)
            .with_blur(16.0)
            .with_spread(2.0)
            .with_inset(true);
        assert!((config.offset_x - 10.0).abs() < 1e-12);
        assert!((config.offset_y - 12.0).abs() < 1e-12);
        assert!((config.blur_radius - 16.0).abs() < 1e-12);
        assert!((config.spread - 2.0).abs() < 1e-12);
        assert!(config.inset);
    }

    #[test]
    fn test_config_validate() {
        let good = DropShadowConfig::new();
        assert!(good.validate().is_ok());

        let bad = DropShadowConfig {
            blur_radius: 600.0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_shadow_extent() {
        let config = DropShadowConfig::new()
            .with_offset(4.0, 4.0)
            .with_blur(8.0)
            .with_spread(0.0);
        let ext = config.extent();
        assert!(ext.right > ext.left); // shadow goes right
        assert!(ext.bottom > ext.top); // shadow goes down
        assert!(ext.total_width() > 0.0);
        assert!(ext.total_height() > 0.0);
    }

    #[test]
    fn test_gaussian_kernel_zero_sigma() {
        let k = GaussianKernel::new(0.0);
        assert_eq!(k.size(), 1);
        assert!((k.weights()[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_kernel_normalized() {
        let k = GaussianKernel::new(5.0);
        let sum: f64 = k.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let k = GaussianKernel::new(3.0);
        let n = k.size();
        for i in 0..n / 2 {
            assert!((k.weights()[i] - k.weights()[n - 1 - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_generate_rect_shadow_nonzero() {
        let config = DropShadowConfig::new().with_blur(2.0);
        let shadow = generate_rect_shadow(10, 10, &config);
        assert!(!shadow.is_empty());
        // Center area should have non-zero alpha
        let mid_row = shadow.len() / 2;
        let mid_col = shadow[0].len() / 2;
        assert!(shadow[mid_row][mid_col] > 0.0);
    }

    #[test]
    fn test_kernel_apply_flat_signal() {
        let k = GaussianKernel::new(2.0);
        let signal = vec![1.0; 20];
        let result = k.apply_1d(&signal, 10);
        // Blurring a flat signal should stay ~1.0
        assert!((result - 1.0).abs() < 1e-9);
    }
}
