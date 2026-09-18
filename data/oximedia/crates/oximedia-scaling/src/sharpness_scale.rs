#![allow(dead_code)]
//! Sharpness-aware scaling with post-scale enhancement
//!
//! Provides scaling operations that analyze and preserve sharpness during
//! resize. Includes unsharp masking, edge-adaptive enhancement, and
//! perceptual sharpness metrics to ensure scaled output remains crisp.

use std::fmt;

/// Sharpness enhancement method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharpenMethod {
    /// Unsharp mask (classic Photoshop-style).
    UnsharpMask,
    /// Laplacian edge enhancement.
    Laplacian,
    /// Adaptive sharpening that avoids over-sharpening flat regions.
    Adaptive,
}

impl fmt::Display for SharpenMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::UnsharpMask => "unsharp-mask",
            Self::Laplacian => "laplacian",
            Self::Adaptive => "adaptive",
        };
        write!(f, "{s}")
    }
}

/// Configuration for sharpness-aware scaling.
#[derive(Debug, Clone)]
pub struct SharpnessScaleConfig {
    /// Source width.
    pub src_width: u32,
    /// Source height.
    pub src_height: u32,
    /// Destination width.
    pub dst_width: u32,
    /// Destination height.
    pub dst_height: u32,
    /// Sharpening method to apply after scaling.
    pub method: SharpenMethod,
    /// Sharpening strength (0.0 = none, 1.0 = full).
    pub strength: f64,
    /// Radius for the sharpening kernel (in pixels).
    pub radius: u32,
    /// Threshold below which sharpening is not applied (adaptive only).
    pub threshold: f64,
}

impl SharpnessScaleConfig {
    /// Create a new sharpness scale config with default parameters.
    pub fn new(src_width: u32, src_height: u32, dst_width: u32, dst_height: u32) -> Self {
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            method: SharpenMethod::UnsharpMask,
            strength: 0.5,
            radius: 1,
            threshold: 4.0,
        }
    }

    /// Set the sharpening method.
    pub fn with_method(mut self, method: SharpenMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the sharpening strength.
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 2.0);
        self
    }

    /// Set the kernel radius.
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius.max(1);
        self
    }

    /// Set the adaptive threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.max(0.0);
        self
    }

    /// Whether this is an upscale operation.
    pub fn is_upscale(&self) -> bool {
        self.dst_width > self.src_width || self.dst_height > self.src_height
    }
}

/// Compute a simple sharpness metric (average gradient magnitude).
///
/// Returns a value indicating overall sharpness. Higher = sharper.
#[allow(clippy::cast_precision_loss)]
pub fn sharpness_metric(data: &[u8], width: u32, height: u32) -> f64 {
    if width < 2 || height < 2 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut count = 0u64;

    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            let idx = (y * width + x) as usize;
            let right = (y * width + x + 1) as usize;
            let below = ((y + 1) * width + x) as usize;
            if below < data.len() && right < data.len() {
                let gx = (data[right] as f64 - data[idx] as f64).abs();
                let gy = (data[below] as f64 - data[idx] as f64).abs();
                sum += (gx * gx + gy * gy).sqrt();
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Apply unsharp mask to a grayscale buffer.
///
/// USM = original + strength * (original - blurred)
#[allow(clippy::cast_precision_loss)]
pub fn unsharp_mask(data: &[u8], width: u32, height: u32, strength: f64, radius: u32) -> Vec<u8> {
    let blurred = box_blur(data, width, height, radius);
    let mut out = vec![0u8; data.len()];
    for i in 0..data.len() {
        let orig = data[i] as f64;
        let blur = blurred[i] as f64;
        let val = orig + strength * (orig - blur);
        out[i] = val.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Apply a simple box blur for a given radius.
#[allow(clippy::cast_precision_loss)]
pub fn box_blur(data: &[u8], width: u32, height: u32, radius: u32) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    let r = radius as i32;
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for ky in -r..=r {
                for kx in -r..=r {
                    let ny = y + ky;
                    let nx = x + kx;
                    if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                        sum += data[(ny as u32 * width + nx as u32) as usize] as f64;
                        count += 1;
                    }
                }
            }
            out[(y as u32 * width + x as u32) as usize] = if count > 0 {
                (sum / count as f64).round().clamp(0.0, 255.0) as u8
            } else {
                0
            };
        }
    }
    out
}

/// Apply Laplacian edge enhancement.
#[allow(clippy::cast_precision_loss)]
pub fn laplacian_sharpen(data: &[u8], width: u32, height: u32, strength: f64) -> Vec<u8> {
    let mut out = vec![0u8; data.len()];
    for y in 1..(height.saturating_sub(1)) {
        for x in 1..(width.saturating_sub(1)) {
            let idx = |xx: u32, yy: u32| data[(yy * width + xx) as usize] as f64;
            let center = idx(x, y);
            let laplacian =
                -4.0 * center + idx(x - 1, y) + idx(x + 1, y) + idx(x, y - 1) + idx(x, y + 1);
            let val = center - strength * laplacian;
            out[(y * width + x) as usize] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    // Copy border pixels
    for x in 0..width {
        out[x as usize] = data[x as usize];
        let last_row = ((height - 1) * width + x) as usize;
        if last_row < data.len() {
            out[last_row] = data[last_row];
        }
    }
    for y in 0..height {
        out[(y * width) as usize] = data[(y * width) as usize];
        let last_col = (y * width + width - 1) as usize;
        if last_col < data.len() {
            out[last_col] = data[last_col];
        }
    }
    out
}

/// Apply adaptive sharpening that avoids flat regions.
#[allow(clippy::cast_precision_loss)]
pub fn adaptive_sharpen(
    data: &[u8],
    width: u32,
    height: u32,
    strength: f64,
    threshold: f64,
) -> Vec<u8> {
    let blurred = box_blur(data, width, height, 1);
    let mut out = vec![0u8; data.len()];
    for i in 0..data.len() {
        let orig = data[i] as f64;
        let blur = blurred[i] as f64;
        let diff = (orig - blur).abs();
        if diff > threshold {
            let val = orig + strength * (orig - blur);
            out[i] = val.round().clamp(0.0, 255.0) as u8;
        } else {
            out[i] = data[i];
        }
    }
    out
}

/// Apply sharpening according to the given config.
pub fn sharpen(data: &[u8], width: u32, height: u32, config: &SharpnessScaleConfig) -> Vec<u8> {
    match config.method {
        SharpenMethod::UnsharpMask => {
            unsharp_mask(data, width, height, config.strength, config.radius)
        }
        SharpenMethod::Laplacian => laplacian_sharpen(data, width, height, config.strength),
        SharpenMethod::Adaptive => {
            adaptive_sharpen(data, width, height, config.strength, config.threshold)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpen_method_display() {
        assert_eq!(SharpenMethod::UnsharpMask.to_string(), "unsharp-mask");
        assert_eq!(SharpenMethod::Laplacian.to_string(), "laplacian");
        assert_eq!(SharpenMethod::Adaptive.to_string(), "adaptive");
    }

    #[test]
    fn test_config_defaults() {
        let cfg = SharpnessScaleConfig::new(1920, 1080, 3840, 2160);
        assert_eq!(cfg.method, SharpenMethod::UnsharpMask);
        assert!((cfg.strength - 0.5).abs() < f64::EPSILON);
        assert_eq!(cfg.radius, 1);
    }

    #[test]
    fn test_config_builder() {
        let cfg = SharpnessScaleConfig::new(1920, 1080, 3840, 2160)
            .with_method(SharpenMethod::Adaptive)
            .with_strength(0.8)
            .with_radius(2)
            .with_threshold(10.0);
        assert_eq!(cfg.method, SharpenMethod::Adaptive);
        assert!((cfg.strength - 0.8).abs() < f64::EPSILON);
        assert_eq!(cfg.radius, 2);
        assert!((cfg.threshold - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_strength_clamped() {
        let cfg = SharpnessScaleConfig::new(100, 100, 200, 200).with_strength(5.0);
        assert!((cfg.strength - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_upscale() {
        let up = SharpnessScaleConfig::new(100, 100, 200, 200);
        assert!(up.is_upscale());
        let down = SharpnessScaleConfig::new(200, 200, 100, 100);
        assert!(!down.is_upscale());
    }

    #[test]
    fn test_sharpness_metric_uniform() {
        let data = vec![128u8; 16];
        let metric = sharpness_metric(&data, 4, 4);
        assert!((metric - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sharpness_metric_edge() {
        // Left half 0, right half 255
        let mut data = vec![0u8; 16];
        for y in 0..4u32 {
            for x in 2..4u32 {
                data[(y * 4 + x) as usize] = 255;
            }
        }
        let metric = sharpness_metric(&data, 4, 4);
        assert!(metric > 0.0);
    }

    #[test]
    fn test_box_blur_uniform() {
        let data = vec![100u8; 16];
        let blurred = box_blur(&data, 4, 4, 1);
        for &v in &blurred {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_unsharp_mask_zero_strength() {
        let data = vec![50u8; 16];
        let result = unsharp_mask(&data, 4, 4, 0.0, 1);
        assert_eq!(result, data);
    }

    #[test]
    fn test_laplacian_uniform() {
        // Uniform data → laplacian is 0 → output equals input
        let data = vec![128u8; 25]; // 5x5
        let result = laplacian_sharpen(&data, 5, 5, 1.0);
        for &v in &result {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn test_adaptive_below_threshold() {
        // Uniform data: diff is 0, below any threshold → unchanged
        let data = vec![80u8; 16];
        let result = adaptive_sharpen(&data, 4, 4, 1.0, 1.0);
        assert_eq!(result, data);
    }

    #[test]
    fn test_sharpen_dispatch_usm() {
        let data = vec![128u8; 16];
        let cfg = SharpnessScaleConfig::new(4, 4, 4, 4);
        let result = sharpen(&data, 4, 4, &cfg);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_sharpen_dispatch_laplacian() {
        let data = vec![128u8; 25];
        let cfg = SharpnessScaleConfig::new(5, 5, 5, 5).with_method(SharpenMethod::Laplacian);
        let result = sharpen(&data, 5, 5, &cfg);
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_sharpen_dispatch_adaptive() {
        let data = vec![128u8; 16];
        let cfg = SharpnessScaleConfig::new(4, 4, 4, 4).with_method(SharpenMethod::Adaptive);
        let result = sharpen(&data, 4, 4, &cfg);
        assert_eq!(result.len(), 16);
    }
}
