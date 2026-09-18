//! LUT preview, visualization, and statistical analysis.
//!
//! Provides gradient swatch rendering, waveform previews, and statistical
//! analysis of LUT properties such as contrast ratio and saturation change.

#![allow(dead_code)]

/// A gradient swatch descriptor for LUT preview generation.
#[derive(Clone, Debug)]
pub struct GradientSwatch {
    /// Width of the preview image in pixels.
    pub width: u32,
    /// Height of the preview image in pixels.
    pub height: u32,
    /// Start color (RGB, normalized 0.0–1.0).
    pub start_rgb: [f64; 3],
    /// End color (RGB, normalized 0.0–1.0).
    pub end_rgb: [f64; 3],
}

impl GradientSwatch {
    /// Create a new gradient swatch.
    #[must_use]
    pub fn new(width: u32, height: u32, start_rgb: [f64; 3], end_rgb: [f64; 3]) -> Self {
        Self {
            width,
            height,
            start_rgb,
            end_rgb,
        }
    }

    /// Create a standard grayscale ramp swatch.
    #[must_use]
    pub fn grayscale_ramp(width: u32, height: u32) -> Self {
        Self::new(width, height, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }
}

/// Generate a gradient preview image with a LUT applied.
///
/// Interpolates from `swatch.start_rgb` to `swatch.end_rgb` across the width,
/// applies `apply_lut` to each pixel, and returns raw RGB bytes.
///
/// # Arguments
///
/// * `swatch` - Gradient swatch configuration
/// * `apply_lut` - Closure mapping input RGB to output RGB
///
/// # Returns
///
/// Raw RGB bytes (3 bytes per pixel, row-major).
#[must_use]
pub fn generate_gradient_preview(
    swatch: &GradientSwatch,
    apply_lut: &dyn Fn([f64; 3]) -> [f64; 3],
) -> Vec<u8> {
    let w = swatch.width as usize;
    let h = swatch.height as usize;
    let mut pixels = Vec::with_capacity(w * h * 3);

    for _row in 0..h {
        for col in 0..w {
            let t = if w <= 1 {
                0.0
            } else {
                col as f64 / (w - 1) as f64
            };
            let input = [
                swatch.start_rgb[0] + t * (swatch.end_rgb[0] - swatch.start_rgb[0]),
                swatch.start_rgb[1] + t * (swatch.end_rgb[1] - swatch.start_rgb[1]),
                swatch.start_rgb[2] + t * (swatch.end_rgb[2] - swatch.start_rgb[2]),
            ];
            let output = apply_lut(input);
            for ch in 0..3 {
                let byte = (output[ch].clamp(0.0, 1.0) * 255.0).round() as u8;
                pixels.push(byte);
            }
        }
    }
    pixels
}

/// Waveform preview data (mono samples).
#[derive(Clone, Debug)]
pub struct WaveformPreview {
    /// Waveform sample values (normalized 0.0–1.0).
    pub samples: Vec<f64>,
}

impl WaveformPreview {
    /// Create a waveform preview from a slice of samples.
    #[must_use]
    pub fn new(samples: Vec<f64>) -> Self {
        Self { samples }
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns true if there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Compute the contrast ratio of a 1D LUT.
///
/// Measures the ratio of maximum output to minimum output over `steps` evenly-spaced
/// input samples. Returns infinity if the minimum is zero.
///
/// # Arguments
///
/// * `lut` - Closure mapping input (0.0–1.0) to output
/// * `steps` - Number of evaluation steps (must be >= 2)
#[must_use]
pub fn compute_lut_contrast_ratio(lut: &dyn Fn(f64) -> f64, steps: usize) -> f64 {
    if steps < 2 {
        return 1.0;
    }
    let mut min_out = f64::MAX;
    let mut max_out = f64::MIN;
    for i in 0..steps {
        let x = i as f64 / (steps - 1) as f64;
        let y = lut(x);
        if y < min_out {
            min_out = y;
        }
        if y > max_out {
            max_out = y;
        }
    }
    if min_out <= 0.0 {
        f64::INFINITY
    } else {
        max_out / min_out
    }
}

/// Compute the average saturation change introduced by a 3D LUT.
///
/// Evaluates the LUT at `steps` uniformly-sampled gray ramp inputs and
/// measures how much the output deviates from neutral gray (equal R, G, B).
///
/// # Arguments
///
/// * `lut` - Closure mapping input RGB to output RGB
/// * `steps` - Number of evaluation steps (must be >= 1)
#[must_use]
pub fn compute_lut_saturation_change(lut: &dyn Fn([f64; 3]) -> [f64; 3], steps: usize) -> f64 {
    if steps == 0 {
        return 0.0;
    }
    let mut total_deviation = 0.0;
    for i in 0..steps {
        let v = i as f64 / steps.max(1) as f64;
        let input = [v, v, v];
        let output = lut(input);
        let avg = (output[0] + output[1] + output[2]) / 3.0;
        let deviation = output.iter().map(|&c| (c - avg).abs()).sum::<f64>() / 3.0;
        total_deviation += deviation;
    }
    total_deviation / steps as f64
}

/// Statistical summary of a 1D LUT.
#[derive(Clone, Debug)]
pub struct LutStats {
    /// Minimum output value.
    pub min_output: f64,
    /// Maximum output value.
    pub max_output: f64,
    /// Average gain (mean output / mean input).
    pub avg_gain: f64,
    /// Contrast ratio (`max_output` / `min_output`, or inf if min is 0).
    pub contrast_ratio: f64,
}

/// Analyze a 1D LUT and return statistical summary.
///
/// # Arguments
///
/// * `values` - LUT table values (indexed 0 to N-1, representing inputs 0.0 to 1.0)
#[must_use]
pub fn analyze_1d_lut(values: &[f64]) -> LutStats {
    if values.is_empty() {
        return LutStats {
            min_output: 0.0,
            max_output: 0.0,
            avg_gain: 1.0,
            contrast_ratio: 1.0,
        };
    }

    let n = values.len();
    let mut min_out = f64::MAX;
    let mut max_out = f64::MIN;
    let mut sum_out = 0.0;
    let mut sum_in = 0.0;

    for (i, &v) in values.iter().enumerate() {
        let input = i as f64 / (n - 1).max(1) as f64;
        if v < min_out {
            min_out = v;
        }
        if v > max_out {
            max_out = v;
        }
        sum_out += v;
        sum_in += input;
    }

    let mean_in = sum_in / n as f64;
    let mean_out = sum_out / n as f64;
    let avg_gain = if mean_in == 0.0 {
        1.0
    } else {
        mean_out / mean_in
    };
    let contrast_ratio = if min_out <= 0.0 {
        f64::INFINITY
    } else {
        max_out / min_out
    };

    LutStats {
        min_output: min_out,
        max_output: max_out,
        avg_gain,
        contrast_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_preview_pixel_count() {
        let swatch = GradientSwatch::grayscale_ramp(10, 4);
        let preview = generate_gradient_preview(&swatch, &|rgb| rgb);
        assert_eq!(preview.len(), 10 * 4 * 3);
    }

    #[test]
    fn test_gradient_preview_identity_lut() {
        let swatch = GradientSwatch::grayscale_ramp(2, 1);
        let preview = generate_gradient_preview(&swatch, &|rgb| rgb);
        // First pixel: black
        assert_eq!(preview[0], 0);
        assert_eq!(preview[1], 0);
        assert_eq!(preview[2], 0);
        // Last pixel: white
        assert_eq!(preview[3], 255);
        assert_eq!(preview[4], 255);
        assert_eq!(preview[5], 255);
    }

    #[test]
    fn test_gradient_preview_invert_lut() {
        let swatch = GradientSwatch::grayscale_ramp(2, 1);
        let preview =
            generate_gradient_preview(&swatch, &|rgb| [1.0 - rgb[0], 1.0 - rgb[1], 1.0 - rgb[2]]);
        // First pixel: white (inverted black)
        assert_eq!(preview[0], 255);
        // Last pixel: black (inverted white)
        assert_eq!(preview[3], 0);
    }

    #[test]
    fn test_gradient_preview_clamps_output() {
        let swatch = GradientSwatch::grayscale_ramp(4, 1);
        // LUT returns out-of-range values
        let preview = generate_gradient_preview(&swatch, &|_| [2.0, -1.0, 0.5]);
        assert_eq!(preview[0], 255); // clamped to 1.0 -> 255
        assert_eq!(preview[1], 0); // clamped to 0.0 -> 0
        assert_eq!(preview[2], 128); // 0.5 -> 128
    }

    #[test]
    fn test_compute_lut_contrast_ratio_identity() {
        // Identity LUT: contrast_ratio = max/min, but min=0 -> infinity
        let ratio = compute_lut_contrast_ratio(&|x| x, 100);
        // min output is 0.0 -> infinity
        assert!(ratio.is_infinite());
    }

    #[test]
    fn test_compute_lut_contrast_ratio_offset() {
        // LUT adds 0.5 offset: min=0.5, max=1.5 clamped conceptually
        let ratio = compute_lut_contrast_ratio(&|x| 0.5 + x * 0.5, 100);
        // min_out = 0.5, max_out = 1.0, ratio = 2.0
        assert!((ratio - 2.0).abs() < 0.05);
    }

    #[test]
    fn test_compute_lut_contrast_ratio_one_step() {
        let ratio = compute_lut_contrast_ratio(&|_| 0.5, 1);
        assert_eq!(ratio, 1.0); // fallback for steps < 2
    }

    #[test]
    fn test_compute_lut_saturation_change_identity() {
        // Identity LUT on gray ramp: no saturation change
        let change = compute_lut_saturation_change(&|rgb| rgb, 100);
        assert!(change.abs() < 1e-10);
    }

    #[test]
    fn test_compute_lut_saturation_change_tint() {
        // LUT adds red tint: boosts R channel
        let change = compute_lut_saturation_change(&|rgb| [rgb[0] * 1.2, rgb[1], rgb[2]], 100);
        assert!(change > 0.0);
    }

    #[test]
    fn test_analyze_1d_lut_identity() {
        let size = 256;
        let values: Vec<f64> = (0..size).map(|i| i as f64 / (size - 1) as f64).collect();
        let stats = analyze_1d_lut(&values);
        assert!((stats.min_output - 0.0).abs() < 1e-6);
        assert!((stats.max_output - 1.0).abs() < 1e-6);
        assert!((stats.avg_gain - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_analyze_1d_lut_boosted() {
        // LUT that maps [0, 0.5, 1] -> [0, 0.75, 1.5 clamped] approx
        let values = vec![0.0, 0.75, 1.0];
        let stats = analyze_1d_lut(&values);
        assert!((stats.min_output - 0.0).abs() < 1e-6);
        assert!((stats.max_output - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_1d_lut_empty() {
        let stats = analyze_1d_lut(&[]);
        assert_eq!(stats.avg_gain, 1.0);
        assert_eq!(stats.contrast_ratio, 1.0);
    }

    #[test]
    fn test_analyze_1d_lut_constant() {
        let values = vec![0.5, 0.5, 0.5, 0.5];
        let stats = analyze_1d_lut(&values);
        assert!((stats.min_output - 0.5).abs() < 1e-6);
        assert!((stats.max_output - 0.5).abs() < 1e-6);
        assert_eq!(stats.contrast_ratio, 1.0);
    }

    #[test]
    fn test_waveform_preview_len() {
        let wp = WaveformPreview::new(vec![0.1, 0.5, 0.9]);
        assert_eq!(wp.len(), 3);
        assert!(!wp.is_empty());
    }

    #[test]
    fn test_waveform_preview_empty() {
        let wp = WaveformPreview::new(vec![]);
        assert!(wp.is_empty());
    }

    #[test]
    fn test_gradient_swatch_new() {
        let swatch = GradientSwatch::new(100, 50, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(swatch.width, 100);
        assert_eq!(swatch.height, 50);
    }
}
