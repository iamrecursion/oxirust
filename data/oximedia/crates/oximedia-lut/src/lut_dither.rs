#![allow(dead_code)]
//! LUT dithering utilities to reduce banding artifacts.
//!
//! When a LUT is applied to footage, quantization can cause visible banding
//! in smooth gradients. This module provides ordered (Bayer) dithering and
//! simple error-diffusion to break up these artifacts while preserving
//! perceptual quality.

/// Dither pattern type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DitherPattern {
    /// 2x2 ordered Bayer matrix.
    Bayer2x2,
    /// 4x4 ordered Bayer matrix.
    Bayer4x4,
    /// 8x8 ordered Bayer matrix.
    Bayer8x8,
}

/// Configuration for a dithering operation.
#[derive(Clone, Debug)]
pub struct DitherConfig {
    /// The dither pattern to use.
    pub pattern: DitherPattern,
    /// Dither strength (0.0 = none, 1.0 = full).
    pub strength: f64,
    /// Target bit depth for quantization (e.g. 8, 10, 12).
    pub bit_depth: u32,
}

impl DitherConfig {
    /// Create a new dither configuration.
    #[must_use]
    pub fn new(pattern: DitherPattern, strength: f64, bit_depth: u32) -> Self {
        Self {
            pattern,
            strength: strength.clamp(0.0, 1.0),
            bit_depth,
        }
    }

    /// Default configuration for 8-bit output with Bayer 4x4 pattern.
    #[must_use]
    pub fn default_8bit() -> Self {
        Self::new(DitherPattern::Bayer4x4, 1.0, 8)
    }

    /// Default configuration for 10-bit output with Bayer 4x4 pattern.
    #[must_use]
    pub fn default_10bit() -> Self {
        Self::new(DitherPattern::Bayer4x4, 1.0, 10)
    }
}

/// 2x2 Bayer threshold matrix (values in 0..4, normalized).
const BAYER_2X2: [[f64; 2]; 2] = [[0.0 / 4.0, 2.0 / 4.0], [3.0 / 4.0, 1.0 / 4.0]];

/// 4x4 Bayer threshold matrix.
const BAYER_4X4: [[f64; 4]; 4] = [
    [0.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0, 10.0 / 16.0],
    [12.0 / 16.0, 4.0 / 16.0, 14.0 / 16.0, 6.0 / 16.0],
    [3.0 / 16.0, 11.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0],
    [15.0 / 16.0, 7.0 / 16.0, 13.0 / 16.0, 5.0 / 16.0],
];

/// 8x8 Bayer threshold matrix.
const BAYER_8X8: [[f64; 8]; 8] = [
    [
        0.0 / 64.0,
        32.0 / 64.0,
        8.0 / 64.0,
        40.0 / 64.0,
        2.0 / 64.0,
        34.0 / 64.0,
        10.0 / 64.0,
        42.0 / 64.0,
    ],
    [
        48.0 / 64.0,
        16.0 / 64.0,
        56.0 / 64.0,
        24.0 / 64.0,
        50.0 / 64.0,
        18.0 / 64.0,
        58.0 / 64.0,
        26.0 / 64.0,
    ],
    [
        12.0 / 64.0,
        44.0 / 64.0,
        4.0 / 64.0,
        36.0 / 64.0,
        14.0 / 64.0,
        46.0 / 64.0,
        6.0 / 64.0,
        38.0 / 64.0,
    ],
    [
        60.0 / 64.0,
        28.0 / 64.0,
        52.0 / 64.0,
        20.0 / 64.0,
        62.0 / 64.0,
        30.0 / 64.0,
        54.0 / 64.0,
        22.0 / 64.0,
    ],
    [
        3.0 / 64.0,
        35.0 / 64.0,
        11.0 / 64.0,
        43.0 / 64.0,
        1.0 / 64.0,
        33.0 / 64.0,
        9.0 / 64.0,
        41.0 / 64.0,
    ],
    [
        51.0 / 64.0,
        19.0 / 64.0,
        59.0 / 64.0,
        27.0 / 64.0,
        49.0 / 64.0,
        17.0 / 64.0,
        57.0 / 64.0,
        25.0 / 64.0,
    ],
    [
        15.0 / 64.0,
        47.0 / 64.0,
        7.0 / 64.0,
        39.0 / 64.0,
        13.0 / 64.0,
        45.0 / 64.0,
        5.0 / 64.0,
        37.0 / 64.0,
    ],
    [
        63.0 / 64.0,
        31.0 / 64.0,
        55.0 / 64.0,
        23.0 / 64.0,
        61.0 / 64.0,
        29.0 / 64.0,
        53.0 / 64.0,
        21.0 / 64.0,
    ],
];

/// Get the Bayer threshold for a given pixel position and pattern.
///
/// The threshold is in the range [0.0, 1.0).
#[must_use]
pub fn bayer_threshold(x: usize, y: usize, pattern: DitherPattern) -> f64 {
    match pattern {
        DitherPattern::Bayer2x2 => BAYER_2X2[y % 2][x % 2],
        DitherPattern::Bayer4x4 => BAYER_4X4[y % 4][x % 4],
        DitherPattern::Bayer8x8 => BAYER_8X8[y % 8][x % 8],
    }
}

/// Quantize a normalized value [0.0, 1.0] to the given bit depth.
///
/// Returns the quantized value still in normalized [0.0, 1.0] range.
#[must_use]
pub fn quantize(value: f64, bit_depth: u32) -> f64 {
    let levels = ((1_u64 << bit_depth) - 1) as f64;
    (value * levels).round() / levels
}

/// Apply ordered dithering to a single normalized pixel value.
///
/// # Arguments
/// * `value` - Pixel value in [0.0, 1.0]
/// * `x` - Pixel x coordinate
/// * `y` - Pixel y coordinate
/// * `config` - Dither configuration
#[must_use]
pub fn dither_value(value: f64, x: usize, y: usize, config: &DitherConfig) -> f64 {
    let levels = ((1_u64 << config.bit_depth) - 1) as f64;
    let step = 1.0 / levels;
    let threshold = bayer_threshold(x, y, config.pattern);
    // Offset the value before quantization
    let offset = (threshold - 0.5) * step * config.strength;
    let dithered = (value + offset).clamp(0.0, 1.0);
    quantize(dithered, config.bit_depth)
}

/// Apply ordered dithering to an RGB triplet.
///
/// # Arguments
/// * `rgb` - RGB values in [0.0, 1.0]
/// * `x` - Pixel x coordinate
/// * `y` - Pixel y coordinate
/// * `config` - Dither configuration
#[must_use]
pub fn dither_rgb(rgb: &[f64; 3], x: usize, y: usize, config: &DitherConfig) -> [f64; 3] {
    [
        dither_value(rgb[0], x, y, config),
        dither_value(rgb[1], x, y, config),
        dither_value(rgb[2], x, y, config),
    ]
}

/// Apply ordered dithering to a row of RGB pixels.
///
/// # Arguments
/// * `row` - Slice of RGB pixels
/// * `y` - Row index
/// * `config` - Dither configuration
#[must_use]
pub fn dither_row(row: &[[f64; 3]], y: usize, config: &DitherConfig) -> Vec<[f64; 3]> {
    row.iter()
        .enumerate()
        .map(|(x, px)| dither_rgb(px, x, y, config))
        .collect()
}

/// Compute the peak signal-to-noise ratio between original and dithered values.
///
/// Both slices must have the same length.
///
/// Returns `f64::INFINITY` if the signals are identical.
#[must_use]
pub fn dither_psnr(original: &[f64], dithered: &[f64]) -> f64 {
    if original.len() != dithered.len() || original.is_empty() {
        return 0.0;
    }
    let mse: f64 = original
        .iter()
        .zip(dithered.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / original.len() as f64;
    if mse < 1e-15 {
        return f64::INFINITY;
    }
    10.0 * (1.0 / mse).log10()
}

/// Simple 1D Floyd-Steinberg error diffusion on a scanline.
///
/// Quantizes each value and propagates the error to the next sample.
#[must_use]
pub fn error_diffusion_1d(values: &[f64], bit_depth: u32) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut error = 0.0_f64;
    for &v in values {
        let adjusted = (v + error).clamp(0.0, 1.0);
        let quantized = quantize(adjusted, bit_depth);
        error = adjusted - quantized;
        result.push(quantized);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_8bit() {
        let q = quantize(0.5, 8);
        // 0.5 * 255 = 127.5 -> rounds to 128 -> 128/255
        let expected = 128.0 / 255.0;
        assert!((q - expected).abs() < 1e-10);
    }

    #[test]
    fn test_quantize_boundaries() {
        assert!((quantize(0.0, 8) - 0.0).abs() < 1e-12);
        assert!((quantize(1.0, 8) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantize_10bit() {
        let q = quantize(0.5, 10);
        let levels: f64 = 1023.0;
        let expected = (0.5 * levels).round() / levels;
        assert!((q - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bayer_threshold_2x2_range() {
        for y in 0..2 {
            for x in 0..2 {
                let t = bayer_threshold(x, y, DitherPattern::Bayer2x2);
                assert!(t >= 0.0);
                assert!(t < 1.0);
            }
        }
    }

    #[test]
    fn test_bayer_threshold_4x4_range() {
        for y in 0..4 {
            for x in 0..4 {
                let t = bayer_threshold(x, y, DitherPattern::Bayer4x4);
                assert!(t >= 0.0);
                assert!(t < 1.0);
            }
        }
    }

    #[test]
    fn test_bayer_threshold_wraps() {
        let t1 = bayer_threshold(0, 0, DitherPattern::Bayer4x4);
        let t2 = bayer_threshold(4, 4, DitherPattern::Bayer4x4);
        assert!((t1 - t2).abs() < 1e-15);
    }

    #[test]
    fn test_dither_value_clamps() {
        let config = DitherConfig::default_8bit();
        let result = dither_value(0.0, 0, 0, &config);
        assert!(result >= 0.0);
        let result2 = dither_value(1.0, 3, 3, &config);
        assert!(result2 <= 1.0);
    }

    #[test]
    fn test_dither_rgb_same_quantization() {
        let config = DitherConfig::new(DitherPattern::Bayer2x2, 0.0, 8);
        let rgb = [0.5, 0.5, 0.5];
        let result = dither_rgb(&rgb, 0, 0, &config);
        // With zero strength, all channels should get same quantization
        assert!((result[0] - result[1]).abs() < 1e-12);
        assert!((result[1] - result[2]).abs() < 1e-12);
    }

    #[test]
    fn test_dither_row_length() {
        let config = DitherConfig::default_8bit();
        let row = vec![[0.5, 0.5, 0.5]; 10];
        let result = dither_row(&row, 0, &config);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_dither_psnr_identical() {
        let a = vec![0.1, 0.5, 0.9];
        let psnr = dither_psnr(&a, &a);
        assert!(psnr.is_infinite());
    }

    #[test]
    fn test_dither_psnr_known() {
        let a = vec![0.5];
        let b = vec![0.6];
        let psnr = dither_psnr(&a, &b);
        // MSE = 0.01, PSNR = 10*log10(100) = 20
        assert!((psnr - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_diffusion_1d_boundary() {
        let vals = vec![0.0, 1.0];
        let result = error_diffusion_1d(&vals, 8);
        assert!((result[0] - 0.0).abs() < 1e-12);
        assert!((result[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dither_config_default_8bit() {
        let config = DitherConfig::default_8bit();
        assert_eq!(config.bit_depth, 8);
        assert!((config.strength - 1.0).abs() < 1e-12);
        assert_eq!(config.pattern, DitherPattern::Bayer4x4);
    }

    #[test]
    fn test_error_diffusion_1d_propagation() {
        // A ramp of values that don't land on quantization boundaries
        let vals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let result = error_diffusion_1d(&vals, 8);
        assert_eq!(result.len(), 10);
        // All results should be valid quantized values in [0, 1]
        for &v in &result {
            assert!(v >= 0.0);
            assert!(v <= 1.0);
        }
    }
}
