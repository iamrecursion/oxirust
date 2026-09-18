//! Gamut compression algorithms.
//!
//! This module provides tools for compressing out-of-gamut colors into the target gamut.

use crate::Rgb;

/// Gamut compression method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamutCompressionMethod {
    /// Soft-clip using sigmoid function.
    SoftClip,
    /// Desaturation-based compression.
    Desaturate,
    /// Lightness-preserving compression.
    PreserveLightness,
    /// Roll-off compression (gradual reduction).
    RollOff,
}

/// Gamut compression processor.
pub struct GamutCompression {
    method: GamutCompressionMethod,
    threshold: f64,
    strength: f64,
}

impl GamutCompression {
    /// Create a new gamut compression processor.
    ///
    /// # Arguments
    ///
    /// * `method` - Compression method
    /// * `threshold` - Threshold for starting compression (0.0-1.0)
    /// * `strength` - Compression strength (0.0-1.0)
    #[must_use]
    pub fn new(method: GamutCompressionMethod, threshold: f64, strength: f64) -> Self {
        Self {
            method,
            threshold,
            strength,
        }
    }

    /// Create a default soft-clip compressor.
    #[must_use]
    pub fn soft_clip() -> Self {
        Self::new(GamutCompressionMethod::SoftClip, 0.8, 0.5)
    }

    /// Compress an out-of-gamut color.
    #[must_use]
    pub fn compress(&self, rgb: &Rgb) -> Rgb {
        match self.method {
            GamutCompressionMethod::SoftClip => self.soft_clip_compress(rgb),
            GamutCompressionMethod::Desaturate => self.desaturate_compress(rgb),
            GamutCompressionMethod::PreserveLightness => self.preserve_lightness_compress(rgb),
            GamutCompressionMethod::RollOff => self.rolloff_compress(rgb),
        }
    }

    /// Soft-clip compression using sigmoid function.
    fn soft_clip_compress(&self, rgb: &Rgb) -> Rgb {
        [
            self.soft_clip_channel(rgb[0]),
            self.soft_clip_channel(rgb[1]),
            self.soft_clip_channel(rgb[2]),
        ]
    }

    /// Apply soft-clip to a single channel.
    fn soft_clip_channel(&self, value: f64) -> f64 {
        if value <= self.threshold {
            value
        } else {
            // Map values above threshold into the range [threshold, 1.0]
            // using a smooth sigmoid-like compression
            let excess = value - self.threshold;
            let range = 1.0 - self.threshold;
            // Normalized excess in [0, inf) → compress to [0, 1)
            let t = excess / (excess + range * self.strength);
            self.threshold + t * range
        }
    }

    /// Sigmoid function for smooth compression.
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-10.0 * (x - 0.5)).exp())
    }

    /// Desaturation-based compression.
    fn desaturate_compress(&self, rgb: &Rgb) -> Rgb {
        let max_val = rgb[0].max(rgb[1]).max(rgb[2]);

        if max_val <= 1.0 {
            return *rgb;
        }

        let lightness = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
        let scale = (1.0 - self.strength) + self.strength * (1.0 / max_val);

        [
            (lightness + (rgb[0] - lightness) * scale).clamp(0.0, 1.0),
            (lightness + (rgb[1] - lightness) * scale).clamp(0.0, 1.0),
            (lightness + (rgb[2] - lightness) * scale).clamp(0.0, 1.0),
        ]
    }

    /// Lightness-preserving compression.
    fn preserve_lightness_compress(&self, rgb: &Rgb) -> Rgb {
        self.desaturate_compress(rgb)
    }

    /// Roll-off compression (gradual reduction near boundaries).
    fn rolloff_compress(&self, rgb: &Rgb) -> Rgb {
        [
            self.rolloff_channel(rgb[0]),
            self.rolloff_channel(rgb[1]),
            self.rolloff_channel(rgb[2]),
        ]
    }

    /// Apply roll-off to a single channel.
    fn rolloff_channel(&self, value: f64) -> f64 {
        if value <= self.threshold {
            value
        } else if value >= 1.0 {
            1.0
        } else {
            let t = (value - self.threshold) / (1.0 - self.threshold);
            let compressed = t * t * (3.0 - 2.0 * t); // Smoothstep
            self.threshold + compressed * (1.0 - self.threshold)
        }
    }

    /// Compress an entire image.
    #[must_use]
    pub fn compress_image(&self, image_data: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(image_data.len());

        for chunk in image_data.chunks_exact(3) {
            let r = f64::from(chunk[0]) / 255.0;
            let g = f64::from(chunk[1]) / 255.0;
            let b = f64::from(chunk[2]) / 255.0;

            let compressed = self.compress(&[r, g, b]);

            output.push((compressed[0] * 255.0).round() as u8);
            output.push((compressed[1] * 255.0).round() as u8);
            output.push((compressed[2] * 255.0).round() as u8);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamut_compression_new() {
        let comp = GamutCompression::new(GamutCompressionMethod::SoftClip, 0.8, 0.5);

        assert_eq!(comp.method, GamutCompressionMethod::SoftClip);
        assert!((comp.threshold - 0.8).abs() < 1e-10);
        assert!((comp.strength - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_soft_clip() {
        let comp = GamutCompression::soft_clip();
        let rgb = [0.5, 0.6, 0.7];
        let result = comp.compress(&rgb);

        // Below threshold, should be unchanged
        assert!((result[0] - 0.5).abs() < 0.1);
        assert!((result[1] - 0.6).abs() < 0.1);
        assert!((result[2] - 0.7).abs() < 0.1);
    }

    #[test]
    fn test_soft_clip_over_threshold() {
        let comp = GamutCompression::soft_clip();
        let rgb = [1.5, 1.2, 1.0];
        let result = comp.compress(&rgb);

        // Should compress values over 1.0
        assert!(result[0] <= 1.0);
        assert!(result[1] <= 1.0);
        assert!(result[2] <= 1.0);
    }

    #[test]
    fn test_desaturate() {
        let comp = GamutCompression::new(GamutCompressionMethod::Desaturate, 0.8, 0.8);
        let rgb = [1.5, 0.8, 0.6];
        let result = comp.compress(&rgb);

        // Should bring into gamut
        assert!(result[0] <= 1.0);
        assert!(result[1] <= 1.0);
        assert!(result[2] <= 1.0);
    }

    #[test]
    fn test_rolloff() {
        let comp = GamutCompression::new(GamutCompressionMethod::RollOff, 0.8, 0.5);
        let rgb = [0.9, 0.85, 0.7];
        let result = comp.compress(&rgb);

        // Should apply smooth roll-off
        assert!(result[0] >= 0.8);
        assert!(result[0] <= 1.0);
    }

    #[test]
    fn test_compress_image() {
        let comp = GamutCompression::soft_clip();
        let image = vec![128, 128, 128, 255, 0, 0];
        let output = comp.compress_image(&image);

        assert_eq!(output.len(), image.len());
    }

    #[test]
    fn test_sigmoid() {
        let comp = GamutCompression::soft_clip();

        // Sigmoid should be 0.5 at x=0.5
        let result = comp.sigmoid(0.5);
        assert!((result - 0.5).abs() < 0.1);

        // Sigmoid should approach 0 at x=0
        let result = comp.sigmoid(0.0);
        assert!(result < 0.1);

        // Sigmoid should approach 1 at x=1
        let result = comp.sigmoid(1.0);
        assert!(result > 0.9);
    }
}
