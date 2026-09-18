//! GPU-accelerated histogram equalization with prefix-sum CDF mapping.
//!
//! # Algorithm
//!
//! 1. **Compute histogram** – count pixel occurrences for each intensity bin
//!    (0..255) over the luminance channel.
//! 2. **Prefix-sum equalization mapping** – derive the CDF and create a
//!    monotone mapping table `[u8; 256]`.
//! 3. **Apply tone curve** – map every pixel through the equalization table.
//!
//! Both luma-only and per-channel RGBA equalization are supported via
//! [`HistogramEqualizerConfig`].
//!
//! The implementation uses rayon for CPU-parallel execution.  A future GPU
//! compute-shader path can be dropped in behind the same public API.

use crate::{GpuDevice, Result};
use rayon::prelude::*;

use super::utils;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Controls which channels are equalized and how the mapping is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqualizationMode {
    /// Equalize the luma (Y from BT.601 conversion) and transfer to RGB.
    ///
    /// This preserves colour ratios while improving contrast.
    LumaOnly,
    /// Equalize each of R, G, B independently.
    ///
    /// This maximises per-channel contrast but can shift colours.
    PerChannel,
}

impl Default for EqualizationMode {
    fn default() -> Self {
        Self::LumaOnly
    }
}

/// Configuration for [`HistogramEqualizer`].
#[derive(Debug, Clone)]
pub struct HistogramEqualizerConfig {
    /// Which channels to equalize.
    pub mode: EqualizationMode,
    /// Clip limit for contrast-limited AHE (0.0 = no clipping, 1.0 = full clip).
    ///
    /// Values in (0.0, 1.0) implement a simplified CLAHE-like contrast limit.
    /// A value of `0.0` gives standard histogram equalization.
    pub clip_limit: f32,
}

impl Default for HistogramEqualizerConfig {
    fn default() -> Self {
        Self {
            mode: EqualizationMode::default(),
            clip_limit: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HistogramEqualizer
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-accelerated (CPU-fallback) histogram equalizer.
pub struct HistogramEqualizer {
    config: HistogramEqualizerConfig,
}

impl HistogramEqualizer {
    /// Create a new equalizer with the given configuration.
    #[must_use]
    pub fn new(config: HistogramEqualizerConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration (luma-only, no clip limit).
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(HistogramEqualizerConfig::default())
    }

    /// Equalize the histogram of an RGBA image.
    ///
    /// * `input` / `output` – packed RGBA, 4 bytes per pixel.
    /// * `device` – reserved for GPU dispatch; CPU path is used automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or buffers are too small.
    pub fn equalize(
        &self,
        _device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.equalize_cpu(input, output, width, height)
    }

    /// CPU-only variant — usable without a GPU device.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or buffers are too small.
    pub fn equalize_cpu(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;
        utils::validate_buffer_size(input, width, height, 4)?;
        utils::validate_buffer_size(output, width, height, 4)?;

        match self.config.mode {
            EqualizationMode::LumaOnly => self.equalize_luma(input, output, width, height),
            EqualizationMode::PerChannel => self.equalize_per_channel(input, output, width, height),
        }
    }

    // ── luma-only equalization ────────────────────────────────────────────────

    fn equalize_luma(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let n_pixels = (width * height) as usize;

        // Step 1: build luma histogram.
        let mut hist = [0u64; 256];
        for px in input.chunks_exact(4) {
            let y = luma_bt601(px[0], px[1], px[2]);
            hist[y as usize] += 1;
        }

        // Step 2: apply clip limit.
        let hist = self.apply_clip_limit(hist, n_pixels);

        // Step 3: compute CDF mapping.
        let lut = build_equalization_lut(&hist, n_pixels);

        // Step 4: apply mapping — scale all channels proportionally.
        output
            .par_chunks_exact_mut(4)
            .zip(input.par_chunks_exact(4))
            .for_each(|(out, inn)| {
                let y_orig = luma_bt601(inn[0], inn[1], inn[2]);
                let y_eq = lut[y_orig as usize];

                if y_orig == 0 {
                    out[0] = 0;
                    out[1] = 0;
                    out[2] = 0;
                } else {
                    // Scale RGB proportionally to the new luma.
                    let scale = f32::from(y_eq) / f32::from(y_orig);
                    out[0] = (f32::from(inn[0]) * scale).clamp(0.0, 255.0).round() as u8;
                    out[1] = (f32::from(inn[1]) * scale).clamp(0.0, 255.0).round() as u8;
                    out[2] = (f32::from(inn[2]) * scale).clamp(0.0, 255.0).round() as u8;
                }
                out[3] = inn[3]; // pass-through alpha
            });

        Ok(())
    }

    // ── per-channel equalization ──────────────────────────────────────────────

    fn equalize_per_channel(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let n_pixels = (width * height) as usize;

        // Build per-channel histograms.
        let mut hist_r = [0u64; 256];
        let mut hist_g = [0u64; 256];
        let mut hist_b = [0u64; 256];

        for px in input.chunks_exact(4) {
            hist_r[px[0] as usize] += 1;
            hist_g[px[1] as usize] += 1;
            hist_b[px[2] as usize] += 1;
        }

        let hist_r = self.apply_clip_limit(hist_r, n_pixels);
        let hist_g = self.apply_clip_limit(hist_g, n_pixels);
        let hist_b = self.apply_clip_limit(hist_b, n_pixels);

        let lut_r = build_equalization_lut(&hist_r, n_pixels);
        let lut_g = build_equalization_lut(&hist_g, n_pixels);
        let lut_b = build_equalization_lut(&hist_b, n_pixels);

        output
            .par_chunks_exact_mut(4)
            .zip(input.par_chunks_exact(4))
            .for_each(|(out, inn)| {
                out[0] = lut_r[inn[0] as usize];
                out[1] = lut_g[inn[1] as usize];
                out[2] = lut_b[inn[2] as usize];
                out[3] = inn[3];
            });

        Ok(())
    }

    // ── clip-limit redistribution ─────────────────────────────────────────────

    fn apply_clip_limit(&self, mut hist: [u64; 256], n_pixels: usize) -> [u64; 256] {
        let clip = self.config.clip_limit;
        if clip <= 0.0 {
            return hist;
        }

        let max_count = (clip.clamp(0.0, 1.0) * n_pixels as f32 / 256.0).round() as u64;
        if max_count == 0 {
            return hist;
        }

        // Accumulate clipped pixels and redistribute uniformly.
        let mut clipped_total = 0u64;
        for bin in &mut hist {
            if *bin > max_count {
                clipped_total += *bin - max_count;
                *bin = max_count;
            }
        }

        let redistribute = clipped_total / 256;
        let remainder = (clipped_total % 256) as usize;
        for bin in &mut hist {
            *bin += redistribute;
        }
        for bin in hist.iter_mut().take(remainder) {
            *bin += 1;
        }

        hist
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the BT.601 luma value for an RGB pixel (integer approximation).
#[inline(always)]
fn luma_bt601(r: u8, g: u8, b: u8) -> u8 {
    let y = 0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b);
    y.clamp(0.0, 255.0).round() as u8
}

/// Build a 256-entry equalization LUT from a histogram.
///
/// Uses the classic CDF-based formula:
/// `lut[v] = round((cdf[v] - cdf_min) / (n - cdf_min) * 255)`
fn build_equalization_lut(hist: &[u64; 256], n_pixels: usize) -> [u8; 256] {
    // Compute CDF.
    let mut cdf = [0u64; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Find minimum non-zero CDF value.
    let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
    let denom = (n_pixels as u64).saturating_sub(cdf_min);

    let mut lut = [0u8; 256];
    for (i, lut_v) in lut.iter_mut().enumerate() {
        if cdf[i] == 0 {
            // Bin is empty (before the first populated bin).
            *lut_v = 0;
        } else if denom == 0 {
            // All pixels share the same intensity level — map to maximum.
            *lut_v = 255;
        } else if cdf[i] <= cdf_min {
            // This bin holds only pixels at the minimum CDF level.
            // Standard formula maps it to 0 unless it is also the maximum bin,
            // in which case it should map to 255 (degenerate single-bin case).
            if cdf[i] == cdf[255] {
                *lut_v = 255;
            } else {
                *lut_v = 0;
            }
        } else {
            let num = cdf[i] - cdf_min;
            *lut_v = ((num as f64 / denom as f64) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    lut
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_rgba(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v, v, v, 255u8].repeat((w * h) as usize)
    }

    fn gradient_rgba(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h))
            .flat_map(|i| {
                let v = ((i * 255) / (w * h - 1).max(1)) as u8;
                [v, v, v, 255u8]
            })
            .collect()
    }

    // ── build_equalization_lut ────────────────────────────────────────────────

    #[test]
    fn test_lut_uniform_histogram() {
        // Uniform histogram → output should span full [0, 255].
        let hist = [4u64; 256];
        let lut = build_equalization_lut(&hist, 1024);
        assert_eq!(lut[0], 0, "first bin maps to 0");
        assert_eq!(lut[255], 255, "last bin maps to 255");
        // Monotone.
        for i in 1..256 {
            assert!(lut[i] >= lut[i - 1], "LUT must be monotone at {i}");
        }
    }

    #[test]
    fn test_lut_single_value_histogram() {
        // All pixels have the same value → after equalisation they all go to 255.
        let mut hist = [0u64; 256];
        hist[128] = 100;
        let lut = build_equalization_lut(&hist, 100);
        assert_eq!(lut[128], 255, "single-value bin maps to 255");
    }

    // ── luma_bt601 ────────────────────────────────────────────────────────────

    #[test]
    fn test_luma_pure_red() {
        let y = luma_bt601(255, 0, 0);
        assert_eq!(y, 76, "BT.601 luma of red ≈ 76");
    }

    #[test]
    fn test_luma_pure_green() {
        let y = luma_bt601(0, 255, 0);
        assert_eq!(y, 150, "BT.601 luma of green ≈ 150");
    }

    #[test]
    fn test_luma_white() {
        let y = luma_bt601(255, 255, 255);
        assert_eq!(y, 255, "luma of white = 255");
    }

    #[test]
    fn test_luma_black() {
        let y = luma_bt601(0, 0, 0);
        assert_eq!(y, 0, "luma of black = 0");
    }

    // ── HistogramEqualizer::equalize_cpu ─────────────────────────────────────

    #[test]
    fn test_equalize_constant_image_luma() {
        let w = 8u32;
        let h = 8u32;
        let input = gray_rgba(w, h, 100);
        let mut output = vec![0u8; (w * h * 4) as usize];
        let eq = HistogramEqualizer::default_config();
        eq.equalize_cpu(&input, &mut output, w, h)
            .expect("equalize constant image");
        // Constant image → all pixels equalise to 255.
        for i in 0..(w * h) as usize {
            assert_eq!(output[i * 4 + 3], 255, "alpha must be preserved");
        }
    }

    #[test]
    fn test_equalize_gradient_luma_monotone() {
        let w = 16u32;
        let h = 16u32;
        let input = gradient_rgba(w, h);
        let mut output = vec![0u8; (w * h * 4) as usize];
        let eq = HistogramEqualizer::default_config();
        eq.equalize_cpu(&input, &mut output, w, h)
            .expect("equalize gradient");

        // Output lumas must be non-decreasing (monotone).
        let mut prev_y = 0u8;
        for i in 0..(w * h) as usize {
            let y = luma_bt601(output[i * 4], output[i * 4 + 1], output[i * 4 + 2]);
            assert!(
                y >= prev_y,
                "output luma must be non-decreasing: prev={prev_y}, cur={y}"
            );
            prev_y = y;
        }
    }

    #[test]
    fn test_equalize_per_channel() {
        let w = 8u32;
        let h = 8u32;
        let input = gradient_rgba(w, h);
        let mut output = vec![0u8; (w * h * 4) as usize];
        let eq = HistogramEqualizer::new(HistogramEqualizerConfig {
            mode: EqualizationMode::PerChannel,
            clip_limit: 0.0,
        });
        eq.equalize_cpu(&input, &mut output, w, h)
            .expect("equalize per channel");
        // First pixel should be 0 (min of gradient), last should be 255 (max).
        let n = (w * h) as usize;
        assert_eq!(output[0], 0, "first pixel red = 0 after per-channel eq");
        assert_eq!(
            output[(n - 1) * 4],
            255,
            "last pixel red = 255 after per-channel eq"
        );
    }

    #[test]
    fn test_equalize_alpha_passthrough_luma() {
        let w = 4u32;
        let h = 4u32;
        let input: Vec<u8> = (0..w * h * 4)
            .map(|i| if i % 4 == 3 { 200u8 } else { 128 })
            .collect();
        let mut output = vec![0u8; (w * h * 4) as usize];
        HistogramEqualizer::default_config()
            .equalize_cpu(&input, &mut output, w, h)
            .expect("equalize alpha passthrough luma");
        for i in 0..(w * h) as usize {
            assert_eq!(output[i * 4 + 3], 200, "alpha must pass through");
        }
    }

    #[test]
    fn test_equalize_alpha_passthrough_per_channel() {
        let w = 4u32;
        let h = 4u32;
        let input: Vec<u8> = (0..w * h * 4)
            .map(|i| if i % 4 == 3 { 77u8 } else { 100 })
            .collect();
        let mut output = vec![0u8; (w * h * 4) as usize];
        HistogramEqualizer::new(HistogramEqualizerConfig {
            mode: EqualizationMode::PerChannel,
            clip_limit: 0.0,
        })
        .equalize_cpu(&input, &mut output, w, h)
        .expect("equalize alpha passthrough per channel");
        for i in 0..(w * h) as usize {
            assert_eq!(output[i * 4 + 3], 77, "alpha must pass through");
        }
    }

    #[test]
    fn test_equalize_invalid_dimensions() {
        let input = vec![0u8; 64];
        let mut output = vec![0u8; 64];
        let result = HistogramEqualizer::default_config().equalize_cpu(&input, &mut output, 0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_equalize_buffer_too_small() {
        let input = vec![0u8; 4]; // too small for 4×4
        let mut output = vec![0u8; 64];
        let result = HistogramEqualizer::default_config().equalize_cpu(&input, &mut output, 4, 4);
        assert!(result.is_err());
    }

    // ── clip limit ────────────────────────────────────────────────────────────

    #[test]
    fn test_clip_limit_preserves_total() {
        let eq = HistogramEqualizer::new(HistogramEqualizerConfig {
            mode: EqualizationMode::LumaOnly,
            clip_limit: 0.3,
        });
        let mut hist = [0u64; 256];
        hist[100] = 500;
        hist[150] = 300;
        let n = 800usize;
        let clipped = eq.apply_clip_limit(hist, n);
        let total: u64 = clipped.iter().sum();
        assert_eq!(total, n as u64, "clip limit must preserve pixel count");
    }

    #[test]
    fn test_clip_limit_zero_no_change() {
        let eq = HistogramEqualizer::new(HistogramEqualizerConfig {
            mode: EqualizationMode::LumaOnly,
            clip_limit: 0.0,
        });
        let hist = [10u64; 256];
        let result = eq.apply_clip_limit(hist, 2560);
        assert_eq!(hist, result, "zero clip limit must not change histogram");
    }
}
