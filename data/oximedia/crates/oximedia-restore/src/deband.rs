//! Debanding for legacy digitized content.
//!
//! Removes color/luminance banding artifacts that appear in low-bit-depth or compressed sources.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Configuration for the debanding processor.
#[derive(Debug, Clone)]
pub struct DebandConfig {
    /// Detection threshold: pixels within this range of their neighbors are considered banded.
    pub threshold: f32,
    /// Spatial search radius in pixels.
    pub radius: usize,
    /// Strength of correction (0.0–1.0).
    pub strength: f32,
    /// Number of iterations to apply.
    pub iterations: usize,
    /// Whether to process each channel independently.
    pub per_channel: bool,
}

impl Default for DebandConfig {
    fn default() -> Self {
        Self {
            threshold: 0.02,
            radius: 4,
            strength: 0.8,
            iterations: 1,
            per_channel: false,
        }
    }
}

/// A detected banding region.
#[derive(Debug, Clone, Copy)]
pub struct BandRegion {
    /// X coordinate of the banding segment start.
    pub x: usize,
    /// Y coordinate of the row.
    pub y: usize,
    /// Length of the segment in pixels.
    pub length: usize,
    /// Estimated band level (0.0–1.0).
    pub level: f32,
}

/// Detect banding regions in a single-channel image.
///
/// Returns list of detected band segments.
pub fn detect_bands(
    pixels: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> Vec<BandRegion> {
    let mut bands = Vec::new();
    if width == 0 || height == 0 || pixels.len() < width * height {
        return bands;
    }

    // Scan each row looking for flat segments (banding)
    for y in 0..height {
        let row_start = y * width;
        let row = &pixels[row_start..row_start + width];

        let mut seg_start = 0;
        let mut seg_level = row[0];

        for x in 1..width {
            let diff = (row[x] - seg_level).abs();
            if diff > threshold {
                // End of flat segment
                let length = x - seg_start;
                if length >= 4 {
                    bands.push(BandRegion {
                        x: seg_start,
                        y,
                        length,
                        level: seg_level,
                    });
                }
                seg_start = x;
                seg_level = row[x];
            }
        }
        // Handle segment that runs to end of row
        let length = width - seg_start;
        if length >= 4 {
            bands.push(BandRegion {
                x: seg_start,
                y,
                length,
                level: seg_level,
            });
        }
    }

    bands
}

/// Debanding processor.
#[derive(Debug, Clone)]
pub struct Debander {
    config: DebandConfig,
}

impl Debander {
    /// Create a new debander.
    pub fn new(config: DebandConfig) -> Self {
        Self { config }
    }

    /// Process a single-channel image in-place.
    ///
    /// `pixels`: flat array of pixel values (0.0–1.0) in row-major order.
    pub fn process(&self, pixels: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut output = pixels.to_vec();
        for _ in 0..self.config.iterations {
            output = self.process_pass(&output, width, height);
        }
        output
    }

    /// One pass of debanding.
    fn process_pass(&self, pixels: &[f32], width: usize, height: usize) -> Vec<f32> {
        if width == 0 || height == 0 || pixels.len() < width * height {
            return pixels.to_vec();
        }

        let mut output = pixels.to_vec();
        let radius = self.config.radius;
        let threshold = self.config.threshold;
        let strength = self.config.strength;

        for y in 0..height {
            for x in 0..width {
                let center = pixels[y * width + x];

                // Sample neighbors within radius
                let x_lo = x.saturating_sub(radius);
                let x_hi = (x + radius + 1).min(width);
                let y_lo = y.saturating_sub(radius);
                let y_hi = (y + radius + 1).min(height);

                let mut sum = 0.0f32;
                let mut count = 0u32;

                for ny in y_lo..y_hi {
                    for nx in x_lo..x_hi {
                        let neighbor = pixels[ny * width + nx];
                        if (neighbor - center).abs() <= threshold {
                            sum += neighbor;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let avg = sum / count as f32;
                    // Blend towards average if the pixel looks banded
                    let corrected = center * (1.0 - strength) + avg * strength;
                    output[y * width + x] = corrected.clamp(0.0, 1.0);
                }
            }
        }

        output
    }

    /// Get the config.
    pub fn config(&self) -> &DebandConfig {
        &self.config
    }

    /// Estimate the banding severity of an image (0.0 = none, 1.0 = severe).
    pub fn banding_severity(&self, pixels: &[f32], width: usize, height: usize) -> f32 {
        if pixels.is_empty() || width == 0 || height == 0 {
            return 0.0;
        }

        let bands = detect_bands(pixels, width, height, self.config.threshold);
        let total_pixels = (width * height) as f32;
        let banded_pixels: f32 = bands.iter().map(|b| b.length as f32).sum();
        (banded_pixels / total_pixels).min(1.0)
    }
}

/// Dither pattern for adding controlled noise to break up banding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherPattern {
    /// Ordered Bayer 4x4 matrix dither.
    Bayer4x4,
    /// Ordered Bayer 8x8 matrix dither.
    Bayer8x8,
    /// White noise dither.
    WhiteNoise,
}

/// Bayer 4x4 dither matrix (normalized 0–1).
const BAYER4: [[f32; 4]; 4] = [
    [0.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0, 10.0 / 16.0],
    [12.0 / 16.0, 4.0 / 16.0, 14.0 / 16.0, 6.0 / 16.0],
    [3.0 / 16.0, 11.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0],
    [15.0 / 16.0, 7.0 / 16.0, 13.0 / 16.0, 5.0 / 16.0],
];

/// Apply ordered dither to an image.
pub fn apply_dither(
    pixels: &[f32],
    width: usize,
    height: usize,
    pattern: DitherPattern,
    amplitude: f32,
) -> Vec<f32> {
    let mut output = pixels.to_vec();
    if width == 0 || height == 0 || pixels.len() < width * height {
        return output;
    }

    for y in 0..height {
        for x in 0..width {
            let dither_val = match pattern {
                DitherPattern::Bayer4x4 => BAYER4[y % 4][x % 4] - 0.5,
                DitherPattern::Bayer8x8 => {
                    // Use double-period Bayer approximation
                    BAYER4[(y / 2) % 4][(x / 2) % 4] - 0.5
                }
                DitherPattern::WhiteNoise => {
                    // Deterministic pseudo-noise based on position
                    let seed = (x * 1664525 + y * 1013904223) & 0xFFFF;
                    seed as f32 / 65536.0 - 0.5
                }
            };
            let orig = pixels[y * width + x];
            output[y * width + x] = (orig + dither_val * amplitude).clamp(0.0, 1.0);
        }
    }

    output
}

/// Multi-channel debanding for RGB images.
#[derive(Debug, Clone)]
pub struct RgbDebander {
    debander: Debander,
}

impl RgbDebander {
    /// Create a new RGB debander.
    pub fn new(config: DebandConfig) -> Self {
        Self {
            debander: Debander::new(config),
        }
    }

    /// Process an RGB image.
    ///
    /// `pixels`: interleaved RGB pixels (length = width * height * 3).
    pub fn process_rgb(&self, pixels: &[f32], width: usize, height: usize) -> Vec<f32> {
        if pixels.len() < width * height * 3 {
            return pixels.to_vec();
        }

        // Deinterleave
        let mut r: Vec<f32> = Vec::with_capacity(width * height);
        let mut g: Vec<f32> = Vec::with_capacity(width * height);
        let mut b: Vec<f32> = Vec::with_capacity(width * height);

        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }

        // Process each channel
        let r_out = self.debander.process(&r, width, height);
        let g_out = self.debander.process(&g, width, height);
        let b_out = self.debander.process(&b, width, height);

        // Reinterleave
        let mut output = Vec::with_capacity(width * height * 3);
        for i in 0..width * height {
            output.push(r_out[i]);
            output.push(g_out[i]);
            output.push(b_out[i]);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deband_config_default() {
        let c = DebandConfig::default();
        assert!(c.threshold > 0.0);
        assert!(c.radius > 0);
        assert!(c.strength > 0.0);
    }

    #[test]
    fn test_detect_bands_empty() {
        let bands = detect_bands(&[], 0, 0, 0.02);
        assert!(bands.is_empty());
    }

    #[test]
    fn test_detect_bands_uniform_row() {
        // Uniform row: one long flat segment
        let pixels = vec![0.5f32; 20];
        let bands = detect_bands(&pixels, 20, 1, 0.02);
        assert!(!bands.is_empty());
        assert_eq!(bands[0].length, 20);
    }

    #[test]
    fn test_detect_bands_two_levels() {
        // Row with two distinct levels
        let mut pixels = vec![0.2f32; 10];
        pixels.extend(vec![0.8f32; 10]);
        let bands = detect_bands(&pixels, 20, 1, 0.1);
        assert_eq!(bands.len(), 2);
    }

    #[test]
    fn test_debander_passthrough_smooth() {
        let config = DebandConfig {
            strength: 0.5,
            ..Default::default()
        };
        let debander = Debander::new(config);
        let pixels = vec![0.5f32; 16];
        let result = debander.process(&pixels, 4, 4);
        for &v in &result {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_debander_clamps_output() {
        let config = DebandConfig::default();
        let debander = Debander::new(config);
        let pixels = vec![1.0f32; 16];
        let result = debander.process(&pixels, 4, 4);
        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_debander_wrong_size() {
        let config = DebandConfig::default();
        let debander = Debander::new(config);
        let pixels = vec![0.5f32; 5]; // too small for 4x4
        let result = debander.process(&pixels, 4, 4);
        assert_eq!(result, pixels); // returns original
    }

    #[test]
    fn test_banding_severity_uniform() {
        let config = DebandConfig::default();
        let debander = Debander::new(config);
        let pixels = vec![0.5f32; 100];
        let severity = debander.banding_severity(&pixels, 10, 10);
        // Uniform = full banding detected
        assert!(severity > 0.0);
    }

    #[test]
    fn test_apply_dither_bayer4x4() {
        let pixels = vec![0.5f32; 16];
        let result = apply_dither(&pixels, 4, 4, DitherPattern::Bayer4x4, 0.1);
        // Result should be different from input
        assert_ne!(result, pixels);
        // All values should be valid
        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_apply_dither_zero_amplitude() {
        let pixels = vec![0.5f32; 16];
        let result = apply_dither(&pixels, 4, 4, DitherPattern::Bayer4x4, 0.0);
        for (&orig, &res) in pixels.iter().zip(result.iter()) {
            assert!((orig - res).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_dither_white_noise() {
        let pixels = vec![0.5f32; 16];
        let result = apply_dither(&pixels, 4, 4, DitherPattern::WhiteNoise, 0.05);
        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_rgb_debander_output_size() {
        let config = DebandConfig::default();
        let debander = RgbDebander::new(config);
        let pixels = vec![0.5f32; 4 * 4 * 3];
        let result = debander.process_rgb(&pixels, 4, 4);
        assert_eq!(result.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_rgb_debander_uniform_image() {
        let config = DebandConfig {
            strength: 0.5,
            ..Default::default()
        };
        let debander = RgbDebander::new(config);
        let pixels = vec![0.5f32; 4 * 4 * 3];
        let result = debander.process_rgb(&pixels, 4, 4);
        for &v in &result {
            assert!((v - 0.5).abs() < 1e-4);
        }
    }
}
