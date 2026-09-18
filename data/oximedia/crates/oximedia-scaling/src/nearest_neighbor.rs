//! Nearest-neighbor scaling for pixel-art and retro content.
//!
//! Provides integer-ratio and arbitrary-ratio nearest-neighbor scaling that
//! preserves hard pixel edges without interpolation. This is ideal for:
//! - Pixel art upscaling (e.g. 2x, 3x, 4x)
//! - Retro game content display
//! - Thumbnail generation where speed matters more than quality
//! - Preview rendering

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(dead_code)]

use std::fmt;

/// Configuration for nearest-neighbor scaling.
#[derive(Debug, Clone)]
pub struct NearestNeighborConfig {
    /// Target width in pixels.
    pub target_width: u32,
    /// Target height in pixels.
    pub target_height: u32,
    /// Whether to apply integer-only scaling (snaps to nearest integer ratio).
    pub integer_only: bool,
}

impl NearestNeighborConfig {
    /// Create a new configuration with target dimensions.
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            target_width,
            target_height,
            integer_only: false,
        }
    }

    /// Create a configuration for integer-ratio scaling (pixel-art mode).
    pub fn pixel_art(target_width: u32, target_height: u32) -> Self {
        Self {
            target_width,
            target_height,
            integer_only: true,
        }
    }

    /// Set integer-only mode.
    pub fn with_integer_only(mut self, integer_only: bool) -> Self {
        self.integer_only = integer_only;
        self
    }
}

impl fmt::Display for NearestNeighborConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NearestNeighbor({}x{}, integer={})",
            self.target_width, self.target_height, self.integer_only
        )
    }
}

/// Nearest-neighbor scaler engine.
#[derive(Debug)]
pub struct NearestNeighborScaler {
    config: NearestNeighborConfig,
}

impl NearestNeighborScaler {
    /// Create a new scaler with the given configuration.
    pub fn new(config: NearestNeighborConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &NearestNeighborConfig {
        &self.config
    }

    /// Compute effective target dimensions, applying integer-ratio snapping if enabled.
    ///
    /// When `integer_only` is true, the scale factor is rounded to the nearest
    /// integer (minimum 1). Both axes use the same integer factor to preserve
    /// square pixels.
    pub fn effective_dimensions(&self, src_width: u32, src_height: u32) -> (u32, u32) {
        if src_width == 0 || src_height == 0 {
            return (0, 0);
        }

        if self.config.integer_only {
            // Compute the best integer scale factor that fits both axes.
            let scale_x = (self.config.target_width as f64 / src_width as f64).round() as u32;
            let scale_y = (self.config.target_height as f64 / src_height as f64).round() as u32;
            // Use the minimum to ensure we don't exceed either target dimension.
            let scale = scale_x.min(scale_y).max(1);
            (src_width * scale, src_height * scale)
        } else {
            (self.config.target_width, self.config.target_height)
        }
    }

    /// Scale a single-channel (grayscale) image using nearest-neighbor sampling.
    ///
    /// Returns `None` if buffer size doesn't match dimensions or dimensions are zero.
    pub fn scale_gray(
        &self,
        pixels: &[u8],
        src_width: u32,
        src_height: u32,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let sw = src_width as usize;
        let sh = src_height as usize;
        if sw == 0 || sh == 0 || pixels.len() < sw * sh {
            return None;
        }

        let (dst_w, dst_h) = self.effective_dimensions(src_width, src_height);
        let dw = dst_w as usize;
        let dh = dst_h as usize;
        if dw == 0 || dh == 0 {
            return None;
        }

        let mut out = vec![0u8; dw * dh];
        for dy in 0..dh {
            // Half-pixel center offset: maps each output pixel center to the nearest
            // input pixel center using integer arithmetic.
            // sy = (dy * sh + sh/2) / dh  ≡  floor(dy/dh * sh + 0.5)
            let sy = (dy * sh + sh / 2) / dh;
            for dx in 0..dw {
                let sx = (dx * sw + sw / 2) / dw;
                out[dy * dw + dx] = pixels[sy * sw + sx];
            }
        }

        Some((out, dst_w, dst_h))
    }

    /// Scale a packed RGB (3 bytes per pixel) image using nearest-neighbor sampling.
    ///
    /// Returns `None` if buffer size doesn't match dimensions or dimensions are zero.
    pub fn scale_rgb(
        &self,
        pixels: &[u8],
        src_width: u32,
        src_height: u32,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let sw = src_width as usize;
        let sh = src_height as usize;
        if sw == 0 || sh == 0 || pixels.len() < sw * sh * 3 {
            return None;
        }

        let (dst_w, dst_h) = self.effective_dimensions(src_width, src_height);
        let dw = dst_w as usize;
        let dh = dst_h as usize;
        if dw == 0 || dh == 0 {
            return None;
        }

        let mut out = vec![0u8; dw * dh * 3];
        for dy in 0..dh {
            let sy = (dy * sh + sh / 2) / dh;
            for dx in 0..dw {
                let sx = (dx * sw + sw / 2) / dw;
                let src_base = (sy * sw + sx) * 3;
                let dst_base = (dy * dw + dx) * 3;
                out[dst_base] = pixels[src_base];
                out[dst_base + 1] = pixels[src_base + 1];
                out[dst_base + 2] = pixels[src_base + 2];
            }
        }

        Some((out, dst_w, dst_h))
    }

    /// Scale a packed RGBA (4 bytes per pixel) image using nearest-neighbor sampling.
    ///
    /// Returns `None` if buffer size doesn't match dimensions or dimensions are zero.
    pub fn scale_rgba(
        &self,
        pixels: &[u8],
        src_width: u32,
        src_height: u32,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let sw = src_width as usize;
        let sh = src_height as usize;
        if sw == 0 || sh == 0 || pixels.len() < sw * sh * 4 {
            return None;
        }

        let (dst_w, dst_h) = self.effective_dimensions(src_width, src_height);
        let dw = dst_w as usize;
        let dh = dst_h as usize;
        if dw == 0 || dh == 0 {
            return None;
        }

        let mut out = vec![0u8; dw * dh * 4];
        for dy in 0..dh {
            let sy = (dy * sh + sh / 2) / dh;
            for dx in 0..dw {
                let sx = (dx * sw + sw / 2) / dw;
                let src_base = (sy * sw + sx) * 4;
                let dst_base = (dy * dw + dx) * 4;
                out[dst_base] = pixels[src_base];
                out[dst_base + 1] = pixels[src_base + 1];
                out[dst_base + 2] = pixels[src_base + 2];
                out[dst_base + 3] = pixels[src_base + 3];
            }
        }

        Some((out, dst_w, dst_h))
    }
}

/// Convenience function: scale a grayscale image by an integer factor.
///
/// Each source pixel becomes a `factor × factor` block in the output.
/// Returns `None` if factor is 0 or dimensions are zero.
pub fn scale_integer(pixels: &[u8], width: u32, height: u32, factor: u32) -> Option<Vec<u8>> {
    if factor == 0 || width == 0 || height == 0 {
        return None;
    }
    let sw = width as usize;
    let sh = height as usize;
    if pixels.len() < sw * sh {
        return None;
    }

    let dw = sw * factor as usize;
    let dh = sh * factor as usize;
    let f = factor as usize;
    let mut out = vec![0u8; dw * dh];

    for sy in 0..sh {
        for sx in 0..sw {
            let val = pixels[sy * sw + sx];
            for fy in 0..f {
                let dy = sy * f + fy;
                for fx in 0..f {
                    let dx = sx * f + fx;
                    out[dy * dw + dx] = val;
                }
            }
        }
    }

    Some(out)
}

/// Convenience function: scale an RGB image by an integer factor.
///
/// Each source pixel becomes a `factor × factor` block in the output.
pub fn scale_integer_rgb(pixels: &[u8], width: u32, height: u32, factor: u32) -> Option<Vec<u8>> {
    if factor == 0 || width == 0 || height == 0 {
        return None;
    }
    let sw = width as usize;
    let sh = height as usize;
    if pixels.len() < sw * sh * 3 {
        return None;
    }

    let dw = sw * factor as usize;
    let dh = sh * factor as usize;
    let f = factor as usize;
    let mut out = vec![0u8; dw * dh * 3];

    for sy in 0..sh {
        for sx in 0..sw {
            let src_base = (sy * sw + sx) * 3;
            let r = pixels[src_base];
            let g = pixels[src_base + 1];
            let b = pixels[src_base + 2];
            for fy in 0..f {
                let dy = sy * f + fy;
                for fx in 0..f {
                    let dx = sx * f + fx;
                    let dst_base = (dy * dw + dx) * 3;
                    out[dst_base] = r;
                    out[dst_base + 1] = g;
                    out[dst_base + 2] = b;
                }
            }
        }
    }

    Some(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let cfg = NearestNeighborConfig::new(640, 480);
        assert_eq!(cfg.target_width, 640);
        assert_eq!(cfg.target_height, 480);
        assert!(!cfg.integer_only);
    }

    #[test]
    fn test_config_pixel_art() {
        let cfg = NearestNeighborConfig::pixel_art(640, 480);
        assert!(cfg.integer_only);
    }

    #[test]
    fn test_config_display() {
        let cfg = NearestNeighborConfig::new(320, 240);
        let s = cfg.to_string();
        assert!(s.contains("320"));
        assert!(s.contains("240"));
    }

    #[test]
    fn test_effective_dimensions_arbitrary() {
        let cfg = NearestNeighborConfig::new(640, 480);
        let scaler = NearestNeighborScaler::new(cfg);
        assert_eq!(scaler.effective_dimensions(320, 240), (640, 480));
    }

    #[test]
    fn test_effective_dimensions_integer_2x() {
        let cfg = NearestNeighborConfig::pixel_art(640, 480);
        let scaler = NearestNeighborScaler::new(cfg);
        // 320x240 -> 2x = 640x480
        assert_eq!(scaler.effective_dimensions(320, 240), (640, 480));
    }

    #[test]
    fn test_effective_dimensions_integer_3x() {
        let cfg = NearestNeighborConfig::pixel_art(900, 700);
        let scaler = NearestNeighborScaler::new(cfg);
        // 320x240 with target ~900x700 -> round(2.8)=3x, round(2.9)=3x -> min=3x -> 960x720
        assert_eq!(scaler.effective_dimensions(320, 240), (960, 720));
    }

    #[test]
    fn test_effective_dimensions_integer_snap_to_1x() {
        let cfg = NearestNeighborConfig::pixel_art(100, 80);
        let scaler = NearestNeighborScaler::new(cfg);
        // 320x240 with target 100x80 -> ratio < 1 -> clamp to 1x
        assert_eq!(scaler.effective_dimensions(320, 240), (320, 240));
    }

    #[test]
    fn test_effective_dimensions_zero_source() {
        let cfg = NearestNeighborConfig::new(640, 480);
        let scaler = NearestNeighborScaler::new(cfg);
        assert_eq!(scaler.effective_dimensions(0, 100), (0, 0));
    }

    #[test]
    fn test_scale_gray_2x() {
        // 2x2 source
        let pixels = vec![10, 20, 30, 40];
        let cfg = NearestNeighborConfig::new(4, 4);
        let scaler = NearestNeighborScaler::new(cfg);
        let result = scaler.scale_gray(&pixels, 2, 2);
        assert!(result.is_some());
        let (buf, w, h) = result.expect("scale_gray should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(buf.len(), 16);
        // Top-left 2x2 block should all be pixel[0]=10
        assert_eq!(buf[0], 10);
        assert_eq!(buf[1], 10);
        assert_eq!(buf[4], 10);
        assert_eq!(buf[5], 10);
    }

    #[test]
    fn test_scale_gray_downscale() {
        // 4x4 checkerboard
        let pixels = vec![
            0, 255, 0, 255, 255, 0, 255, 0, 0, 255, 0, 255, 255, 0, 255, 0,
        ];
        let cfg = NearestNeighborConfig::new(2, 2);
        let scaler = NearestNeighborScaler::new(cfg);
        let result = scaler.scale_gray(&pixels, 4, 4);
        assert!(result.is_some());
        let (buf, w, h) = result.expect("scale_gray should succeed");
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn test_scale_gray_invalid() {
        let cfg = NearestNeighborConfig::new(4, 4);
        let scaler = NearestNeighborScaler::new(cfg);
        // Buffer too small
        assert!(scaler.scale_gray(&[0u8; 3], 2, 2).is_none());
        // Zero dimensions
        assert!(scaler.scale_gray(&[], 0, 0).is_none());
    }

    #[test]
    fn test_scale_rgb_upscale() {
        // 2x2 RGB image: R, G, B, W
        let pixels = vec![
            255, 0, 0, 0, 255, 0, // row 0
            0, 0, 255, 255, 255, 255, // row 1
        ];
        let cfg = NearestNeighborConfig::new(4, 4);
        let scaler = NearestNeighborScaler::new(cfg);
        let result = scaler.scale_rgb(&pixels, 2, 2);
        assert!(result.is_some());
        let (buf, w, h) = result.expect("scale_rgb should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(buf.len(), 4 * 4 * 3);
        // Top-left pixel should be red
        assert_eq!(buf[0], 255);
        assert_eq!(buf[1], 0);
        assert_eq!(buf[2], 0);
    }

    #[test]
    fn test_scale_rgb_invalid_buffer() {
        let cfg = NearestNeighborConfig::new(4, 4);
        let scaler = NearestNeighborScaler::new(cfg);
        assert!(scaler.scale_rgb(&[0u8; 5], 2, 2).is_none());
    }

    #[test]
    fn test_scale_rgba() {
        let pixels = vec![
            255, 0, 0, 255, 0, 255, 0, 128, // row 0
            0, 0, 255, 64, 255, 255, 0, 255, // row 1
        ];
        let cfg = NearestNeighborConfig::new(4, 4);
        let scaler = NearestNeighborScaler::new(cfg);
        let result = scaler.scale_rgba(&pixels, 2, 2);
        assert!(result.is_some());
        let (buf, w, h) = result.expect("scale_rgba should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(buf.len(), 4 * 4 * 4);
        // Top-left pixel: red with full alpha
        assert_eq!(buf[0], 255);
        assert_eq!(buf[3], 255);
    }

    #[test]
    fn test_scale_rgba_invalid() {
        let cfg = NearestNeighborConfig::new(4, 4);
        let scaler = NearestNeighborScaler::new(cfg);
        assert!(scaler.scale_rgba(&[0u8; 10], 2, 2).is_none());
    }

    #[test]
    fn test_scale_integer_2x() {
        let pixels = vec![100, 200, 50, 150];
        let result = scale_integer(&pixels, 2, 2, 2);
        assert!(result.is_some());
        let out = result.expect("scale_integer should succeed");
        assert_eq!(out.len(), 4 * 4);
        // Each pixel duplicated in 2x2 block
        assert_eq!(out[0], 100);
        assert_eq!(out[1], 100);
        assert_eq!(out[4], 100);
        assert_eq!(out[5], 100);
        assert_eq!(out[2], 200);
        assert_eq!(out[3], 200);
    }

    #[test]
    fn test_scale_integer_1x() {
        let pixels = vec![42, 84, 126, 168];
        let result = scale_integer(&pixels, 2, 2, 1);
        assert!(result.is_some());
        let out = result.expect("scale_integer 1x should succeed");
        assert_eq!(out, pixels);
    }

    #[test]
    fn test_scale_integer_0x() {
        assert!(scale_integer(&[1, 2, 3, 4], 2, 2, 0).is_none());
    }

    #[test]
    fn test_scale_integer_zero_dims() {
        assert!(scale_integer(&[], 0, 0, 2).is_none());
    }

    #[test]
    fn test_scale_integer_rgb_2x() {
        let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let result = scale_integer_rgb(&pixels, 2, 2, 2);
        assert!(result.is_some());
        let out = result.expect("scale_integer_rgb should succeed");
        assert_eq!(out.len(), 4 * 4 * 3);
        // Top-left 2x2 block should be red
        assert_eq!(&out[0..3], &[255, 0, 0]);
        assert_eq!(&out[3..6], &[255, 0, 0]);
    }

    #[test]
    fn test_scale_integer_rgb_invalid() {
        assert!(scale_integer_rgb(&[0u8; 5], 2, 2, 2).is_none());
        assert!(scale_integer_rgb(&[], 0, 0, 2).is_none());
        assert!(scale_integer_rgb(&[0u8; 12], 2, 2, 0).is_none());
    }

    #[test]
    fn test_pixel_art_mode_preserves_exact_pixels() {
        // In pixel-art mode, every output pixel should exactly match a source pixel.
        let pixels = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
        let cfg = NearestNeighborConfig::pixel_art(9, 9);
        let scaler = NearestNeighborScaler::new(cfg);
        let result = scaler.scale_gray(&pixels, 3, 3);
        assert!(result.is_some());
        let (buf, _, _) = result.expect("pixel_art scale should succeed");
        // Every value in output should be one of the source values
        let src_set: std::collections::HashSet<u8> = pixels.iter().copied().collect();
        for &v in &buf {
            assert!(
                src_set.contains(&v),
                "output {v} is not a source pixel value"
            );
        }
    }

    #[test]
    fn test_identity_scale() {
        let pixels = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let cfg = NearestNeighborConfig::new(3, 3);
        let scaler = NearestNeighborScaler::new(cfg);
        let result = scaler.scale_gray(&pixels, 3, 3);
        assert!(result.is_some());
        let (buf, w, h) = result.expect("identity scale should succeed");
        assert_eq!(w, 3);
        assert_eq!(h, 3);
        assert_eq!(buf, pixels);
    }

    #[test]
    fn test_with_integer_only_builder() {
        let cfg = NearestNeighborConfig::new(640, 480).with_integer_only(true);
        assert!(cfg.integer_only);
    }
}
