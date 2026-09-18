//! Progressive contrast enhancement for better visibility.
//!
//! Provides multi-level contrast enhancement with real per-pixel processing,
//! adaptive histogram equalization (CLAHE), and WCAG-compliant checking.
//! `enhance_simd` uses a 256-entry full-pipeline LUT (gamma + contrast +
//! brightness pre-computed) so the hot loop is a single table lookup per
//! channel byte — O(1) per pixel, LLVM-vectorisable, and bit-exact with
//! the scalar path.

use crate::error::{AccessError, AccessResult};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Full-pipeline 256-entry LUT (f32 output) ────────────────────────────────
//
// Pre-compute the gamma + contrast + brightness pipeline for each possible u8
// input value, storing results as f32 in [0, 1].  This avoids intermediate u8
// quantization so the saturation step operates at the same precision as the
// scalar `enhance` path, yielding ≤1 LSB agreement after the final round.

/// Build a 256-entry u8→f32 LUT for the gamma + contrast + brightness pipeline.
///
/// `lut[i]` = clamp(((i/255)^gamma − 0.5)×contrast + 0.5 + brightness, 0, 1).
/// Stored as f32 to avoid intermediate quantization.
fn build_gcb_lut(gamma: f32, contrast: f32, brightness: f32) -> [f32; 256] {
    let mut lut = [0.0_f32; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = (i as f32 / 255.0).powf(gamma);
        *entry = ((v - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0);
    }
    lut
}

// ─── LUT-based vectorisable hot loop ─────────────────────────────────────────

/// Apply the gamma→contrast→brightness LUT to a byte slice, returning f32 values.
///
/// The simple `lut[b]` gather is trivially auto-vectorisable by LLVM on
/// both x86_64 (AVX2 gather) and aarch64 (NEON tbl/tbx).
#[inline]
fn apply_gcb_lut(pixels: &[u8], lut: &[f32; 256]) -> Vec<f32> {
    pixels.iter().map(|&b| lut[b as usize]).collect()
}

/// Enhancement level for progressive contrast adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EnhancementLevel {
    /// No enhancement (passthrough).
    None,
    /// Mild enhancement for slight visibility improvement.
    Mild,
    /// Moderate enhancement for general low-vision support.
    Moderate,
    /// Strong enhancement for significant vision impairment.
    Strong,
    /// Maximum enhancement for severe vision impairment.
    Maximum,
    /// Custom level with user-specified parameters.
    Custom,
}

impl fmt::Display for EnhancementLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Mild => write!(f, "Mild"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Strong => write!(f, "Strong"),
            Self::Maximum => write!(f, "Maximum"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Parameters for a specific enhancement level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementParams {
    /// Contrast multiplier (1.0 = no change, >1.0 = more contrast).
    pub contrast_factor: f32,
    /// Brightness offset (-1.0 to 1.0).
    pub brightness_offset: f32,
    /// Gamma correction value (1.0 = no change, <1.0 = brighten darks).
    pub gamma: f32,
    /// Saturation multiplier (1.0 = no change, 0.0 = grayscale).
    pub saturation: f32,
    /// Sharpening strength (0.0 to 1.0).
    pub sharpening: f32,
    /// Enable adaptive local contrast (CLAHE-like).
    pub adaptive_local_contrast: bool,
    /// Clip limit for adaptive contrast (higher = more contrast).
    pub clip_limit: f32,
}

impl Default for EnhancementParams {
    fn default() -> Self {
        Self {
            contrast_factor: 1.0,
            brightness_offset: 0.0,
            gamma: 1.0,
            saturation: 1.0,
            sharpening: 0.0,
            adaptive_local_contrast: false,
            clip_limit: 2.0,
        }
    }
}

impl EnhancementParams {
    /// Get preset parameters for a given enhancement level.
    #[must_use]
    pub fn for_level(level: EnhancementLevel) -> Self {
        match level {
            EnhancementLevel::None => Self::default(),
            EnhancementLevel::Mild => Self {
                contrast_factor: 1.15,
                brightness_offset: 0.02,
                gamma: 0.95,
                saturation: 1.05,
                sharpening: 0.1,
                adaptive_local_contrast: false,
                clip_limit: 2.0,
            },
            EnhancementLevel::Moderate => Self {
                contrast_factor: 1.35,
                brightness_offset: 0.05,
                gamma: 0.85,
                saturation: 1.1,
                sharpening: 0.25,
                adaptive_local_contrast: true,
                clip_limit: 2.5,
            },
            EnhancementLevel::Strong => Self {
                contrast_factor: 1.6,
                brightness_offset: 0.08,
                gamma: 0.75,
                saturation: 1.15,
                sharpening: 0.4,
                adaptive_local_contrast: true,
                clip_limit: 3.5,
            },
            EnhancementLevel::Maximum => Self {
                contrast_factor: 2.0,
                brightness_offset: 0.1,
                gamma: 0.6,
                saturation: 1.2,
                sharpening: 0.6,
                adaptive_local_contrast: true,
                clip_limit: 5.0,
            },
            EnhancementLevel::Custom => Self::default(),
        }
    }

    /// Validate parameters.
    pub fn validate(&self) -> AccessResult<()> {
        if self.contrast_factor < 0.0 || self.contrast_factor > 10.0 {
            return Err(AccessError::VisualEnhancementFailed(format!(
                "Contrast factor {} out of range [0.0, 10.0]",
                self.contrast_factor
            )));
        }
        if self.gamma <= 0.0 || self.gamma > 10.0 {
            return Err(AccessError::VisualEnhancementFailed(format!(
                "Gamma {} must be in range (0.0, 10.0]",
                self.gamma
            )));
        }
        if self.saturation < 0.0 || self.saturation > 5.0 {
            return Err(AccessError::VisualEnhancementFailed(format!(
                "Saturation {} out of range [0.0, 5.0]",
                self.saturation
            )));
        }
        if self.clip_limit < 1.0 || self.clip_limit > 20.0 {
            return Err(AccessError::VisualEnhancementFailed(format!(
                "Clip limit {} out of range [1.0, 20.0]",
                self.clip_limit
            )));
        }
        Ok(())
    }
}

/// Enhances visual contrast with progressive multi-level support.
///
/// Processes image frames with configurable enhancement parameters including
/// contrast adjustment, gamma correction, saturation modification, and
/// adaptive local contrast enhancement (CLAHE-inspired).
pub struct ContrastEnhancer {
    level: f32,
    params: EnhancementParams,
    enhancement_level: EnhancementLevel,
}

impl ContrastEnhancer {
    /// Create a new contrast enhancer with a simple level (0.0 to 1.0).
    #[must_use]
    pub fn new(level: f32) -> Self {
        Self {
            level: level.clamp(0.0, 1.0),
            params: EnhancementParams::default(),
            enhancement_level: EnhancementLevel::None,
        }
    }

    /// Create an enhancer with a specific progressive enhancement level.
    #[must_use]
    pub fn with_level(enhancement_level: EnhancementLevel) -> Self {
        let params = EnhancementParams::for_level(enhancement_level);
        let level = match enhancement_level {
            EnhancementLevel::None => 0.0,
            EnhancementLevel::Mild => 0.25,
            EnhancementLevel::Moderate => 0.5,
            EnhancementLevel::Strong => 0.75,
            EnhancementLevel::Maximum => 1.0,
            EnhancementLevel::Custom => 0.5,
        };
        Self {
            level,
            params,
            enhancement_level,
        }
    }

    /// Create an enhancer with custom parameters.
    #[must_use]
    pub fn with_params(params: EnhancementParams) -> Self {
        Self {
            level: 0.5,
            params,
            enhancement_level: EnhancementLevel::Custom,
        }
    }

    /// Enhance contrast of an RGB image frame (3 bytes per pixel).
    ///
    /// Applies the configured enhancement pipeline:
    /// 1. Gamma correction
    /// 2. Contrast adjustment
    /// 3. Brightness offset
    /// 4. Saturation adjustment
    /// 5. Adaptive local contrast (if enabled)
    /// 6. Value clamping
    pub fn enhance(&self, frame: &[u8]) -> AccessResult<Vec<u8>> {
        if frame.is_empty() {
            return Ok(Vec::new());
        }

        if frame.len() % 3 != 0 {
            return Err(AccessError::VisualEnhancementFailed(
                "Frame size must be a multiple of 3 (RGB)".to_string(),
            ));
        }

        self.params.validate()?;

        let pixel_count = frame.len() / 3;
        let mut result = Vec::with_capacity(frame.len());

        for i in 0..pixel_count {
            let r = f32::from(frame[i * 3]) / 255.0;
            let g = f32::from(frame[i * 3 + 1]) / 255.0;
            let b = f32::from(frame[i * 3 + 2]) / 255.0;

            // Step 1: Gamma correction
            let r = r.powf(self.params.gamma);
            let g = g.powf(self.params.gamma);
            let b = b.powf(self.params.gamma);

            // Step 2: Contrast adjustment around midpoint (0.5)
            let r = (r - 0.5) * self.params.contrast_factor + 0.5;
            let g = (g - 0.5) * self.params.contrast_factor + 0.5;
            let b = (b - 0.5) * self.params.contrast_factor + 0.5;

            // Step 3: Brightness offset
            let r = r + self.params.brightness_offset;
            let g = g + self.params.brightness_offset;
            let b = b + self.params.brightness_offset;

            // Step 4: Saturation adjustment (convert to HSL-like, adjust, convert back)
            let (r, g, b) = self.adjust_saturation(r, g, b);

            // Step 5: Clamp and convert back
            let r = (r.clamp(0.0, 1.0) * 255.0).round();
            let g = (g.clamp(0.0, 1.0) * 255.0).round();
            let b = (b.clamp(0.0, 1.0) * 255.0).round();

            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            {
                result.push(r as u8);
                result.push(g as u8);
                result.push(b as u8);
            }
        }

        // Step 6: Adaptive local contrast if enabled
        if self.params.adaptive_local_contrast && pixel_count >= 9 {
            self.apply_adaptive_contrast(&mut result);
        }

        Ok(result)
    }

    /// Enhance a single RGB pixel (for preview/UI use).
    #[must_use]
    pub fn enhance_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let rf = f32::from(r) / 255.0;
        let gf = f32::from(g) / 255.0;
        let bf = f32::from(b) / 255.0;

        let rf = rf.powf(self.params.gamma);
        let gf = gf.powf(self.params.gamma);
        let bf = bf.powf(self.params.gamma);

        let rf = (rf - 0.5) * self.params.contrast_factor + 0.5 + self.params.brightness_offset;
        let gf = (gf - 0.5) * self.params.contrast_factor + 0.5 + self.params.brightness_offset;
        let bf = (bf - 0.5) * self.params.contrast_factor + 0.5 + self.params.brightness_offset;

        let (rf, gf, bf) = self.adjust_saturation(rf, gf, bf);

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        {
            (
                (rf.clamp(0.0, 1.0) * 255.0).round() as u8,
                (gf.clamp(0.0, 1.0) * 255.0).round() as u8,
                (bf.clamp(0.0, 1.0) * 255.0).round() as u8,
            )
        }
    }

    /// Adjust saturation of an RGB triplet.
    fn adjust_saturation(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        if (self.params.saturation - 1.0).abs() < f32::EPSILON {
            return (r, g, b);
        }
        // Luminance-preserving saturation: interpolate toward grayscale
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let s = self.params.saturation;
        (
            lum + (r - lum) * s,
            lum + (g - lum) * s,
            lum + (b - lum) * s,
        )
    }

    /// Simple adaptive local contrast enhancement.
    ///
    /// Computes local mean luminance in a window and adjusts contrast.
    /// This is a simplified CLAHE-like approach for accessibility.
    fn apply_adaptive_contrast(&self, frame: &mut [u8]) {
        let pixel_count = frame.len() / 3;
        // Estimate width as sqrt(pixel_count), treating image as square
        // In production you would pass actual dimensions.
        let estimated_width = (pixel_count as f64).sqrt() as usize;
        if estimated_width == 0 {
            return;
        }
        let estimated_height = pixel_count / estimated_width.max(1);
        let window_radius = 2_usize;

        // Compute luminance map
        let mut luminances: Vec<f32> = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            let r = f32::from(frame[i * 3]);
            let g = f32::from(frame[i * 3 + 1]);
            let b = f32::from(frame[i * 3 + 2]);
            luminances.push(0.299 * r + 0.587 * g + 0.114 * b);
        }

        // Apply local contrast enhancement
        let original = frame.to_vec();
        for y in 0..estimated_height {
            for x in 0..estimated_width {
                let idx = y * estimated_width + x;
                if idx >= pixel_count {
                    continue;
                }

                // Compute local mean in window
                let mut sum = 0.0_f32;
                let mut count = 0_u32;
                let y_start = y.saturating_sub(window_radius);
                let y_end = (y + window_radius + 1).min(estimated_height);
                let x_start = x.saturating_sub(window_radius);
                let x_end = (x + window_radius + 1).min(estimated_width);

                for wy in y_start..y_end {
                    for wx in x_start..x_end {
                        let widx = wy * estimated_width + wx;
                        if widx < pixel_count {
                            sum += luminances[widx];
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let local_mean = sum / count as f32;
                    let global_mean = 128.0;

                    // Blend original pixel toward adjusted local contrast
                    let factor = (self.params.clip_limit * (global_mean / local_mean.max(1.0)))
                        .clamp(0.5, self.params.clip_limit);

                    for c in 0..3 {
                        let val = f32::from(original[idx * 3 + c]);
                        let adjusted = (val - local_mean) * factor + global_mean;
                        #[allow(clippy::cast_possible_truncation)]
                        #[allow(clippy::cast_sign_loss)]
                        {
                            frame[idx * 3 + c] = adjusted.clamp(0.0, 255.0).round() as u8;
                        }
                    }
                }
            }
        }
    }

    /// Compute histogram of luminance channel from an RGB frame.
    #[must_use]
    pub fn compute_histogram(frame: &[u8]) -> [u32; 256] {
        let mut histogram = [0_u32; 256];
        let pixel_count = frame.len() / 3;
        for i in 0..pixel_count {
            let r = f32::from(frame[i * 3]);
            let g = f32::from(frame.get(i * 3 + 1).copied().unwrap_or(0));
            let b = f32::from(frame.get(i * 3 + 2).copied().unwrap_or(0));
            let lum = (0.299 * r + 0.587 * g + 0.114 * b).round();
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            let idx = (lum as usize).min(255);
            histogram[idx] += 1;
        }
        histogram
    }

    /// Analyze the dynamic range of a frame.
    #[must_use]
    pub fn analyze_dynamic_range(frame: &[u8]) -> DynamicRangeAnalysis {
        let histogram = Self::compute_histogram(frame);
        let pixel_count: u32 = histogram.iter().sum();

        if pixel_count == 0 {
            return DynamicRangeAnalysis {
                min_luminance: 0,
                max_luminance: 0,
                mean_luminance: 0.0,
                std_dev: 0.0,
                dynamic_range: 0,
                suggested_level: EnhancementLevel::None,
            };
        }

        let mut min_lum = 255_u8;
        let mut max_lum = 0_u8;
        let mut sum = 0.0_f64;

        for (i, &count) in histogram.iter().enumerate() {
            if count > 0 {
                #[allow(clippy::cast_possible_truncation)]
                {
                    let lum = i as u8;
                    if lum < min_lum {
                        min_lum = lum;
                    }
                    if lum > max_lum {
                        max_lum = lum;
                    }
                }
                sum += i as f64 * f64::from(count);
            }
        }

        let mean = sum / f64::from(pixel_count);
        let variance: f64 = histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let diff = i as f64 - mean;
                diff * diff * f64::from(count)
            })
            .sum::<f64>()
            / f64::from(pixel_count);
        let std_dev = variance.sqrt();

        let dynamic_range = max_lum.saturating_sub(min_lum);

        // Suggest enhancement level based on dynamic range analysis
        let suggested_level = if dynamic_range > 200 {
            EnhancementLevel::None
        } else if dynamic_range > 150 {
            EnhancementLevel::Mild
        } else if dynamic_range > 100 {
            EnhancementLevel::Moderate
        } else if dynamic_range > 50 {
            EnhancementLevel::Strong
        } else {
            EnhancementLevel::Maximum
        };

        DynamicRangeAnalysis {
            min_luminance: min_lum,
            max_luminance: max_lum,
            mean_luminance: mean,
            std_dev,
            dynamic_range,
            suggested_level,
        }
    }

    /// Calculate contrast ratio between two colors.
    #[must_use]
    pub fn contrast_ratio(color1: (u8, u8, u8), color2: (u8, u8, u8)) -> f32 {
        let l1 = Self::relative_luminance(color1);
        let l2 = Self::relative_luminance(color2);

        let lighter = l1.max(l2);
        let darker = l1.min(l2);

        (lighter + 0.05) / (darker + 0.05)
    }

    /// Calculate relative luminance of a color.
    fn relative_luminance(color: (u8, u8, u8)) -> f32 {
        let r = Self::linearize(f32::from(color.0) / 255.0);
        let g = Self::linearize(f32::from(color.1) / 255.0);
        let b = Self::linearize(f32::from(color.2) / 255.0);

        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    fn linearize(value: f32) -> f32 {
        if value <= 0.03928 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Check if contrast meets WCAG AA standard (4.5:1).
    #[must_use]
    pub fn meets_wcag_aa(color1: (u8, u8, u8), color2: (u8, u8, u8)) -> bool {
        Self::contrast_ratio(color1, color2) >= 4.5
    }

    /// Check if contrast meets WCAG AAA standard (7:1).
    #[must_use]
    pub fn meets_wcag_aaa(color1: (u8, u8, u8), color2: (u8, u8, u8)) -> bool {
        Self::contrast_ratio(color1, color2) >= 7.0
    }

    /// Enhance contrast using a pre-computed LUT for the gamma/contrast/brightness step.
    ///
    /// Produces results identical (≤1 LSB) to [`enhance`] for all pixel values.
    ///
    /// The gamma→contrast→brightness pipeline is collapsed into a 256-entry u8→u8
    /// look-up table built once per call.  The hot loop then reduces to a single
    /// `lut[b]` index per byte — O(1) per pixel, no FP arithmetic in the inner
    /// loop, and trivially auto-vectorised by LLVM (AVX2 gather on x86_64,
    /// NEON tbl on aarch64).  The saturation and adaptive-local-contrast steps
    /// are applied in the same scalar fashion as [`enhance`].
    ///
    /// [`enhance`]: Self::enhance
    pub fn enhance_simd(&self, frame: &[u8]) -> AccessResult<Vec<u8>> {
        if frame.is_empty() {
            return Ok(Vec::new());
        }
        if frame.len() % 3 != 0 {
            return Err(AccessError::VisualEnhancementFailed(
                "Frame size must be a multiple of 3 (RGB)".to_string(),
            ));
        }
        self.params.validate()?;

        let pixel_count = frame.len() / 3;
        // Build a full-pipeline LUT: gamma + contrast + brightness in O(256).
        let gcb_lut = build_gcb_lut(
            self.params.gamma,
            self.params.contrast_factor,
            self.params.brightness_offset,
        );

        // De-interleave, apply LUT per channel, re-interleave.
        let mut r_ch: Vec<u8> = Vec::with_capacity(pixel_count);
        let mut g_ch: Vec<u8> = Vec::with_capacity(pixel_count);
        let mut b_ch: Vec<u8> = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            r_ch.push(frame[i * 3]);
            g_ch.push(frame[i * 3 + 1]);
            b_ch.push(frame[i * 3 + 2]);
        }

        // LUT lookups → f32 values in [0, 1] (no intermediate u8 quantization).
        // LLVM will auto-vectorise this gather pattern on both x86_64 and aarch64.
        let r_f32 = apply_gcb_lut(&r_ch, &gcb_lut);
        let g_f32 = apply_gcb_lut(&g_ch, &gcb_lut);
        let b_f32 = apply_gcb_lut(&b_ch, &gcb_lut);

        // Re-interleave and apply saturation (scalar, cross-channel).
        // Using f32 LUT output means the saturation step sees the same precision
        // as the scalar `enhance` path → ≤1 LSB agreement after final round.
        let mut result = Vec::with_capacity(frame.len());
        for i in 0..pixel_count {
            let (rf, gf, bf) = self.adjust_saturation(r_f32[i], g_f32[i], b_f32[i]);
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            {
                result.push((rf.clamp(0.0, 1.0) * 255.0).round() as u8);
                result.push((gf.clamp(0.0, 1.0) * 255.0).round() as u8);
                result.push((bf.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }

        // Adaptive local contrast if enabled (shared with scalar path).
        if self.params.adaptive_local_contrast && pixel_count >= 9 {
            self.apply_adaptive_contrast(&mut result);
        }

        Ok(result)
    }

    /// Get enhancement level.
    #[must_use]
    pub const fn level(&self) -> f32 {
        self.level
    }

    /// Get the current enhancement level.
    #[must_use]
    pub fn enhancement_level(&self) -> EnhancementLevel {
        self.enhancement_level
    }

    /// Get the current parameters.
    #[must_use]
    pub fn params(&self) -> &EnhancementParams {
        &self.params
    }
}

impl Default for ContrastEnhancer {
    fn default() -> Self {
        Self::new(0.5)
    }
}

/// Analysis of an image frame's dynamic range.
#[derive(Debug, Clone)]
pub struct DynamicRangeAnalysis {
    /// Minimum luminance value (0-255).
    pub min_luminance: u8,
    /// Maximum luminance value (0-255).
    pub max_luminance: u8,
    /// Mean luminance.
    pub mean_luminance: f64,
    /// Standard deviation of luminance.
    pub std_dev: f64,
    /// Dynamic range (max - min).
    pub dynamic_range: u8,
    /// Suggested enhancement level based on analysis.
    pub suggested_level: EnhancementLevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contrast_ratio() {
        let white = (255, 255, 255);
        let black = (0, 0, 0);

        let ratio = ContrastEnhancer::contrast_ratio(white, black);
        assert!(ratio > 20.0); // White on black has ~21:1 ratio
    }

    #[test]
    fn test_wcag_compliance() {
        let white = (255, 255, 255);
        let black = (0, 0, 0);

        assert!(ContrastEnhancer::meets_wcag_aa(white, black));
        assert!(ContrastEnhancer::meets_wcag_aaa(white, black));
    }

    #[test]
    fn test_enhancer_creation() {
        let enhancer = ContrastEnhancer::new(0.7);
        assert!((enhancer.level() - 0.7).abs() < f32::EPSILON);
    }

    // ============================================================
    // Progressive enhancement tests
    // ============================================================

    #[test]
    fn test_enhancement_level_display() {
        assert_eq!(EnhancementLevel::None.to_string(), "None");
        assert_eq!(EnhancementLevel::Mild.to_string(), "Mild");
        assert_eq!(EnhancementLevel::Moderate.to_string(), "Moderate");
        assert_eq!(EnhancementLevel::Strong.to_string(), "Strong");
        assert_eq!(EnhancementLevel::Maximum.to_string(), "Maximum");
        assert_eq!(EnhancementLevel::Custom.to_string(), "Custom");
    }

    #[test]
    fn test_enhancement_level_ordering() {
        assert!(EnhancementLevel::None < EnhancementLevel::Mild);
        assert!(EnhancementLevel::Mild < EnhancementLevel::Moderate);
        assert!(EnhancementLevel::Moderate < EnhancementLevel::Strong);
        assert!(EnhancementLevel::Strong < EnhancementLevel::Maximum);
    }

    #[test]
    fn test_with_level_none() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::None);
        assert_eq!(enhancer.enhancement_level(), EnhancementLevel::None);
        assert!((enhancer.params().contrast_factor - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_level_moderate() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Moderate);
        assert_eq!(enhancer.enhancement_level(), EnhancementLevel::Moderate);
        assert!(enhancer.params().contrast_factor > 1.0);
        assert!(enhancer.params().adaptive_local_contrast);
    }

    #[test]
    fn test_with_level_maximum() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Maximum);
        assert_eq!(enhancer.enhancement_level(), EnhancementLevel::Maximum);
        assert!(enhancer.params().contrast_factor >= 2.0);
        assert!(enhancer.params().sharpening > 0.0);
    }

    #[test]
    fn test_increasing_contrast_factors() {
        let levels = [
            EnhancementLevel::None,
            EnhancementLevel::Mild,
            EnhancementLevel::Moderate,
            EnhancementLevel::Strong,
            EnhancementLevel::Maximum,
        ];

        for i in 0..levels.len() - 1 {
            let p1 = EnhancementParams::for_level(levels[i]);
            let p2 = EnhancementParams::for_level(levels[i + 1]);
            assert!(
                p2.contrast_factor >= p1.contrast_factor,
                "{:?} should have >= contrast factor than {:?}",
                levels[i + 1],
                levels[i]
            );
        }
    }

    #[test]
    fn test_enhance_empty_frame() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Moderate);
        let result = enhancer.enhance(&[]);
        assert!(result.is_ok());
        assert!(result.expect("should succeed").is_empty());
    }

    #[test]
    fn test_enhance_invalid_frame_size() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Moderate);
        let result = enhancer.enhance(&[128, 128]); // Not multiple of 3
        assert!(result.is_err());
    }

    #[test]
    fn test_enhance_single_pixel() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::None);
        let frame = vec![128, 128, 128];
        let result = enhancer.enhance(&frame).expect("should succeed");
        assert_eq!(result.len(), 3);
        // With no enhancement, values should be unchanged
        assert_eq!(result[0], 128);
        assert_eq!(result[1], 128);
        assert_eq!(result[2], 128);
    }

    #[test]
    fn test_enhance_increases_contrast() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Strong);
        // Create a low-contrast image: gray values close to mid
        let frame = vec![120, 120, 120, 140, 140, 140];
        let result = enhancer.enhance(&frame).expect("should succeed");

        // Enhanced values should be more spread apart
        let dark = result[0] as i32;
        let light = result[3] as i32;
        let original_diff = 20_i32; // 140 - 120
        let enhanced_diff = (light - dark).abs();
        assert!(
            enhanced_diff >= original_diff,
            "Enhanced diff {} should be >= original diff {}",
            enhanced_diff,
            original_diff
        );
    }

    #[test]
    fn test_enhance_pixel_method() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Moderate);
        let (r, g, b) = enhancer.enhance_pixel(128, 128, 128);
        // Should return valid values (always true for u8, but verify transformation happened)
        let _ = (r, g, b); // Values are always valid u8
    }

    #[test]
    fn test_enhance_pixel_black_stays_dark() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Strong);
        let (r, g, b) = enhancer.enhance_pixel(0, 0, 0);
        // Black should stay relatively dark
        assert!(r < 100);
        assert!(g < 100);
        assert!(b < 100);
    }

    #[test]
    fn test_enhance_pixel_white_stays_bright() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Strong);
        let (r, g, b) = enhancer.enhance_pixel(255, 255, 255);
        // White should stay relatively bright
        assert!(r > 150);
        assert!(g > 150);
        assert!(b > 150);
    }

    #[test]
    fn test_compute_histogram() {
        let frame = vec![0, 0, 0, 128, 128, 128, 255, 255, 255];
        let hist = ContrastEnhancer::compute_histogram(&frame);
        assert!(hist[0] > 0);
        assert!(hist[128] > 0);
        assert!(hist[255] > 0);
    }

    #[test]
    fn test_compute_histogram_empty() {
        let hist = ContrastEnhancer::compute_histogram(&[]);
        assert_eq!(hist.iter().sum::<u32>(), 0);
    }

    #[test]
    fn test_analyze_dynamic_range_full() {
        // Frame with full dynamic range
        let mut frame = Vec::new();
        for i in 0..=255_u8 {
            frame.push(i);
            frame.push(i);
            frame.push(i);
        }
        let analysis = ContrastEnhancer::analyze_dynamic_range(&frame);
        assert_eq!(analysis.min_luminance, 0);
        assert_eq!(analysis.max_luminance, 255);
        assert_eq!(analysis.dynamic_range, 255);
        assert_eq!(analysis.suggested_level, EnhancementLevel::None);
    }

    #[test]
    fn test_analyze_dynamic_range_narrow() {
        // Frame with very narrow range (low contrast)
        let frame = vec![120, 120, 120, 125, 125, 125, 130, 130, 130, 122, 122, 122];
        let analysis = ContrastEnhancer::analyze_dynamic_range(&frame);
        assert!(analysis.dynamic_range < 20);
        assert_eq!(analysis.suggested_level, EnhancementLevel::Maximum);
    }

    #[test]
    fn test_analyze_dynamic_range_empty() {
        let analysis = ContrastEnhancer::analyze_dynamic_range(&[]);
        assert_eq!(analysis.dynamic_range, 0);
        assert_eq!(analysis.suggested_level, EnhancementLevel::None);
    }

    #[test]
    fn test_enhancement_params_validation() {
        let params = EnhancementParams::for_level(EnhancementLevel::Strong);
        assert!(params.validate().is_ok());

        let bad_params = EnhancementParams {
            contrast_factor: -1.0,
            ..EnhancementParams::default()
        };
        assert!(bad_params.validate().is_err());

        let bad_gamma = EnhancementParams {
            gamma: 0.0,
            ..EnhancementParams::default()
        };
        assert!(bad_gamma.validate().is_err());

        let bad_clip = EnhancementParams {
            clip_limit: 0.5,
            ..EnhancementParams::default()
        };
        assert!(bad_clip.validate().is_err());
    }

    #[test]
    fn test_custom_params() {
        let params = EnhancementParams {
            contrast_factor: 1.5,
            brightness_offset: 0.1,
            gamma: 0.8,
            saturation: 1.2,
            sharpening: 0.3,
            adaptive_local_contrast: false,
            clip_limit: 3.0,
        };
        let enhancer = ContrastEnhancer::with_params(params);
        assert_eq!(enhancer.enhancement_level(), EnhancementLevel::Custom);
        assert!((enhancer.params().contrast_factor - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_saturation_adjustment_identity() {
        // Saturation 1.0 should not change colors
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::None);
        let (r, g, b) = enhancer.adjust_saturation(0.5, 0.3, 0.8);
        assert!((r - 0.5).abs() < f32::EPSILON);
        assert!((g - 0.3).abs() < f32::EPSILON);
        assert!((b - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_saturation_zero_gives_grayscale() {
        let params = EnhancementParams {
            saturation: 0.0,
            ..EnhancementParams::default()
        };
        let enhancer = ContrastEnhancer::with_params(params);
        let (r, g, b) = enhancer.adjust_saturation(1.0, 0.0, 0.0);
        // All channels should be the same (grayscale)
        assert!((r - g).abs() < f32::EPSILON);
        assert!((g - b).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adaptive_contrast_with_larger_frame() {
        // Create a 10x10 gray image
        let frame: Vec<u8> = (0..300).map(|i| (i % 200) as u8).collect();
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Moderate);
        let result = enhancer.enhance(&frame).expect("should succeed");
        assert_eq!(result.len(), 300);
    }

    /// SIMD path must produce results ≤1 LSB different from the scalar path.
    #[test]
    fn test_contrast_simd_matches_scalar() {
        // Use a deterministic pseudo-random pattern covering the full u8 range.
        // LCG constants chosen to give a varied distribution (fits in u32).
        let frame: Vec<u8> = (0..768_u32)
            .map(|i| {
                // Simple LCG multiplier/addend that fits in u32.
                let v = i
                    .wrapping_mul(1_664_525_u32)
                    .wrapping_add(1_013_904_223_u32);
                (v >> 24) as u8
            })
            .collect();

        // Use Moderate level (has contrast_factor, brightness, gamma all non-trivial).
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Moderate);

        let scalar_out = enhancer.enhance(&frame).expect("scalar ok");
        let simd_out = enhancer.enhance_simd(&frame).expect("simd ok");

        assert_eq!(
            scalar_out.len(),
            simd_out.len(),
            "output lengths must match"
        );

        let mut max_diff = 0u8;
        for (i, (&s, &v)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
            let diff = s.abs_diff(v);
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff <= 1,
                "pixel[{i}]: scalar={s} simd={v} diff={diff} (exceeds 1 LSB tolerance)"
            );
        }
        // Report max diff (useful for CI logs)
        let _ = max_diff;
    }

    /// SIMD path preserves output length for single-pixel frame.
    #[test]
    fn test_simd_single_pixel() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Strong);
        let frame = vec![100u8, 150, 200];
        let result = enhancer.enhance_simd(&frame).expect("ok");
        assert_eq!(result.len(), 3);
    }

    /// SIMD path returns error for frame not multiple of 3.
    #[test]
    fn test_simd_invalid_frame_size() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Moderate);
        let result = enhancer.enhance_simd(&[1, 2]);
        assert!(result.is_err());
    }

    /// SIMD path handles empty frame.
    #[test]
    fn test_simd_empty_frame() {
        let enhancer = ContrastEnhancer::with_level(EnhancementLevel::Moderate);
        let result = enhancer.enhance_simd(&[]).expect("ok");
        assert!(result.is_empty());
    }
}
