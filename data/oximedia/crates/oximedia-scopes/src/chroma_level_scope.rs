//! Dedicated Cb/Cr chroma level scope with broadcast-legal limits.
//!
//! This module provides a specialised chrominance level analyser that displays
//! the Cb (blue-difference) and Cr (red-difference) components of a video
//! frame as separate waveform traces, overlaid with ITU-R BT.601/709/2020
//! broadcast legal range markers.
//!
//! # Broadcast Legal Ranges (8-bit)
//!
//! For standard-range (narrow-range) encoding:
//! - Chroma channels: 16–240 (nominal range), outside indicates illegal chroma
//!
//! For full-range encoding:
//! - Chroma channels: 0–255 (all values legal)
//!
//! # Display
//!
//! Each scope is rendered as a column-by-column histogram (parade style) with:
//! - Cyan trace for Cb
//! - Red trace for Cr
//! - Dashed horizontal lines at legal boundaries (16 and 240 for 8-bit)
//! - Optional out-of-legal-range pixel highlighting

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// The colour space used to define legal chroma ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaStandard {
    /// ITU-R BT.601 (SD).
    Bt601,
    /// ITU-R BT.709 (HD).
    Bt709,
    /// ITU-R BT.2020 (UHD/HDR).
    Bt2020,
    /// Full range (0–255 legal for 8-bit).
    FullRange,
}

impl ChromaStandard {
    /// Returns `(chroma_min, chroma_max)` for 8-bit (0–255) encoding.
    #[must_use]
    pub fn legal_range_8bit(self) -> (u8, u8) {
        match self {
            // All three narrow-range standards use the same 8-bit chroma bounds
            Self::Bt601 | Self::Bt709 | Self::Bt2020 => (16, 240),
            Self::FullRange => (0, 255),
        }
    }

    /// Returns `(chroma_min, chroma_max)` normalised to 0.0–1.0.
    #[must_use]
    pub fn legal_range_norm(self) -> (f32, f32) {
        let (lo, hi) = self.legal_range_8bit();
        (lo as f32 / 255.0, hi as f32 / 255.0)
    }
}

/// Configuration for the chroma level scope.
#[derive(Debug, Clone)]
pub struct ChromaLevelConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Colour standard for legal range markers.
    pub standard: ChromaStandard,
    /// Whether to display Cb and Cr side-by-side (parade) or overlaid.
    pub parade_mode: bool,
    /// Whether to highlight out-of-legal-range pixels in the waveform.
    pub highlight_illegal: bool,
    /// RGBA colour for the Cb trace.
    pub cb_color: [u8; 4],
    /// RGBA colour for the Cr trace.
    pub cr_color: [u8; 4],
    /// RGBA colour for legal limit lines.
    pub limit_color: [u8; 4],
    /// RGBA colour for illegal pixel highlights.
    pub illegal_color: [u8; 4],
    /// Background RGBA colour.
    pub background_color: [u8; 4],
}

impl Default for ChromaLevelConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 256,
            standard: ChromaStandard::Bt709,
            parade_mode: true,
            highlight_illegal: true,
            cb_color: [0, 200, 255, 220],      // cyan
            cr_color: [255, 80, 80, 220],      // red
            limit_color: [255, 200, 0, 180],   // yellow
            illegal_color: [255, 0, 100, 255], // hot pink
            background_color: [10, 10, 12, 255],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Per-frame chroma analysis results.
#[derive(Debug, Clone)]
pub struct ChromaAnalysis {
    /// Average Cb value (0–255).
    pub avg_cb: f32,
    /// Average Cr value (0–255).
    pub avg_cr: f32,
    /// Maximum Cb value.
    pub max_cb: u8,
    /// Maximum Cr value.
    pub max_cr: u8,
    /// Minimum Cb value.
    pub min_cb: u8,
    /// Minimum Cr value.
    pub min_cr: u8,
    /// Pixel count exceeding the legal Cb upper limit.
    pub illegal_cb_high: u64,
    /// Pixel count below the legal Cb lower limit.
    pub illegal_cb_low: u64,
    /// Pixel count exceeding the legal Cr upper limit.
    pub illegal_cr_high: u64,
    /// Pixel count below the legal Cr lower limit.
    pub illegal_cr_low: u64,
    /// Total pixels analysed.
    pub total_pixels: u64,
}

impl ChromaAnalysis {
    /// Returns `true` if there are any out-of-legal-range pixels.
    #[must_use]
    pub fn has_illegal_chroma(&self) -> bool {
        self.illegal_cb_high > 0
            || self.illegal_cb_low > 0
            || self.illegal_cr_high > 0
            || self.illegal_cr_low > 0
    }

    /// Returns the percentage of pixels (0–100) that are outside legal chroma range.
    #[must_use]
    pub fn illegal_pct(&self) -> f32 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        let illegal =
            self.illegal_cb_high + self.illegal_cb_low + self.illegal_cr_high + self.illegal_cr_low;
        (illegal as f32 / self.total_pixels as f32) * 100.0
    }
}

/// Analyse the Cb/Cr chroma channels of an RGB24 frame.
///
/// # Errors
///
/// Returns an error if the frame buffer is too small.
pub fn analyze_chroma_levels(
    frame: &[u8],
    width: u32,
    height: u32,
    standard: ChromaStandard,
) -> OxiResult<ChromaAnalysis> {
    let num_pixels = (width as usize) * (height as usize);
    let expected = num_pixels * 3;
    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame too small: need {expected}, got {}",
            frame.len()
        )));
    }
    if num_pixels == 0 {
        return Ok(ChromaAnalysis {
            avg_cb: 128.0,
            avg_cr: 128.0,
            max_cb: 128,
            max_cr: 128,
            min_cb: 128,
            min_cr: 128,
            illegal_cb_high: 0,
            illegal_cb_low: 0,
            illegal_cr_high: 0,
            illegal_cr_low: 0,
            total_pixels: 0,
        });
    }

    let (chroma_min, chroma_max) = standard.legal_range_8bit();
    let mut sum_cb = 0u64;
    let mut sum_cr = 0u64;
    let mut max_cb = 0u8;
    let mut max_cr = 0u8;
    let mut min_cb = 255u8;
    let mut min_cr = 255u8;
    let mut illegal_cb_high = 0u64;
    let mut illegal_cb_low = 0u64;
    let mut illegal_cr_high = 0u64;
    let mut illegal_cr_low = 0u64;

    for i in 0..num_pixels {
        let r = frame[i * 3];
        let g = frame[i * 3 + 1];
        let b = frame[i * 3 + 2];

        // BT.709 RGB→YCbCr (full range)
        let (cb, cr) = rgb_to_cbcr_bt709(r, g, b);

        sum_cb += u64::from(cb);
        sum_cr += u64::from(cr);
        max_cb = max_cb.max(cb);
        max_cr = max_cr.max(cr);
        min_cb = min_cb.min(cb);
        min_cr = min_cr.min(cr);

        if cb > chroma_max {
            illegal_cb_high += 1;
        } else if cb < chroma_min {
            illegal_cb_low += 1;
        }
        if cr > chroma_max {
            illegal_cr_high += 1;
        } else if cr < chroma_min {
            illegal_cr_low += 1;
        }
    }

    Ok(ChromaAnalysis {
        avg_cb: sum_cb as f32 / num_pixels as f32,
        avg_cr: sum_cr as f32 / num_pixels as f32,
        max_cb,
        max_cr,
        min_cb,
        min_cr,
        illegal_cb_high,
        illegal_cb_low,
        illegal_cr_high,
        illegal_cr_low,
        total_pixels: num_pixels as u64,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Render the Cb/Cr chroma level scope as an RGBA image.
///
/// # Errors
///
/// Returns an error if the frame buffer is too small.
pub fn render_chroma_level_scope(
    frame: &[u8],
    frame_w: u32,
    frame_h: u32,
    config: &ChromaLevelConfig,
) -> OxiResult<Vec<u8>> {
    let num_pixels = (frame_w as usize) * (frame_h as usize);
    let expected = num_pixels * 3;
    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame too small: need {expected}, got {}",
            frame.len()
        )));
    }

    let out_w = config.width as usize;
    let out_h = config.height as usize;
    let mut pixels = vec![0u8; out_w * out_h * 4];

    // Background
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.copy_from_slice(&config.background_color);
    }

    if num_pixels == 0 {
        return Ok(pixels);
    }

    let (lo, hi) = config.standard.legal_range_8bit();

    if config.parade_mode {
        // Left half = Cb, right half = Cr
        let section_w = out_w / 2;
        render_chroma_section(
            frame,
            frame_w,
            frame_h,
            &mut pixels,
            out_w,
            out_h,
            0,
            section_w,
            lo,
            hi,
            config,
            false, // Cb
        );
        render_chroma_section(
            frame,
            frame_w,
            frame_h,
            &mut pixels,
            out_w,
            out_h,
            section_w,
            section_w,
            lo,
            hi,
            config,
            true, // Cr
        );
        // Section divider
        for py in 0..out_h {
            let idx = (py * out_w + section_w) * 4;
            if idx + 3 < pixels.len() {
                pixels[idx] = 80;
                pixels[idx + 1] = 80;
                pixels[idx + 2] = 80;
                pixels[idx + 3] = 255;
            }
        }
    } else {
        // Overlay both channels
        render_chroma_section(
            frame,
            frame_w,
            frame_h,
            &mut pixels,
            out_w,
            out_h,
            0,
            out_w,
            lo,
            hi,
            config,
            false,
        );
        render_chroma_section(
            frame,
            frame_w,
            frame_h,
            &mut pixels,
            out_w,
            out_h,
            0,
            out_w,
            lo,
            hi,
            config,
            true,
        );
    }

    // Legal limit horizontal lines
    let lo_y = out_h - 1 - ((lo as usize * out_h) / 255).min(out_h - 1);
    let hi_y = out_h - 1 - ((hi as usize * out_h) / 255).min(out_h - 1);
    let limit_color = config.limit_color;
    for px in 0..out_w {
        blend_pixel_slice(&mut pixels, out_w, px, lo_y, limit_color);
        blend_pixel_slice(&mut pixels, out_w, px, hi_y, limit_color);
    }

    Ok(pixels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Render a single Cb or Cr column histogram section.
#[allow(clippy::too_many_arguments)]
fn render_chroma_section(
    frame: &[u8],
    frame_w: u32,
    frame_h: u32,
    pixels: &mut [u8],
    out_w: usize,
    out_h: usize,
    offset_x: usize,
    section_w: usize,
    _lo: u8,
    _hi: u8,
    config: &ChromaLevelConfig,
    use_cr: bool,
) {
    let num_pixels = (frame_w as usize) * (frame_h as usize);
    // Build per-column histograms (section_w columns × 256 bins)
    let mut col_hist = vec![[0u32; 256]; section_w.max(1)];

    for i in 0..num_pixels {
        let r = frame[i * 3];
        let g = frame[i * 3 + 1];
        let b = frame[i * 3 + 2];
        let (cb, cr) = rgb_to_cbcr_bt709(r, g, b);
        let chroma_val = if use_cr { cr } else { cb };

        // Map frame pixel x to section column
        let frame_x = i % frame_w as usize;
        let col = (frame_x * section_w) / (frame_w as usize).max(1);
        if col < section_w {
            col_hist[col][chroma_val as usize] += 1;
        }
    }

    // Find max for brightness normalisation
    let max_val = col_hist
        .iter()
        .flat_map(|c| c.iter().copied())
        .max()
        .unwrap_or(1);

    let trace_color = if use_cr {
        config.cr_color
    } else {
        config.cb_color
    };

    for (col, histogram) in col_hist.iter().enumerate() {
        let px = offset_x + col;
        if px >= out_w {
            break;
        }
        for (val, &count) in histogram.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let brightness = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
            let py = out_h - 1 - ((val * out_h) / 256).min(out_h - 1);
            let mut color = trace_color;
            color[3] = brightness;
            blend_pixel_slice(pixels, out_w, px, py, color);
        }
    }
}

/// BT.709 full-range RGB → (Cb, Cr) in 8-bit integer.
fn rgb_to_cbcr_bt709(r: u8, g: u8, b: u8) -> (u8, u8) {
    let rf = f32::from(r);
    let gf = f32::from(g);
    let bf = f32::from(b);
    // Standard BT.709 matrix for full-range (0–255 → 0–255, 128 = neutral)
    let cb = (-0.1687 * rf - 0.3313 * gf + 0.5 * bf + 128.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    let cr = (0.5 * rf - 0.4187 * gf - 0.0813 * bf + 128.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    (cb, cr)
}

fn blend_pixel_slice(pixels: &mut [u8], w: usize, x: usize, y: usize, color: [u8; 4]) {
    let idx = (y * w + x) * 4;
    if idx + 3 >= pixels.len() {
        return;
    }
    let a = color[3] as f32 / 255.0;
    let ia = 1.0 - a;
    pixels[idx] = (color[0] as f32 * a + pixels[idx] as f32 * ia) as u8;
    pixels[idx + 1] = (color[1] as f32 * a + pixels[idx + 1] as f32 * ia) as u8;
    pixels[idx + 2] = (color[2] as f32 * a + pixels[idx + 2] as f32 * ia) as u8;
    pixels[idx + 3] = 255;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn grey_frame(v: u8, w: u32, h: u32) -> Vec<u8> {
        vec![v; (w * h * 3) as usize]
    }

    #[test]
    fn test_chroma_standard_legal_range_bt709() {
        let (lo, hi) = ChromaStandard::Bt709.legal_range_8bit();
        assert_eq!(lo, 16);
        assert_eq!(hi, 240);
    }

    #[test]
    fn test_chroma_standard_legal_range_full() {
        let (lo, hi) = ChromaStandard::FullRange.legal_range_8bit();
        assert_eq!(lo, 0);
        assert_eq!(hi, 255);
    }

    #[test]
    fn test_chroma_standard_norm_range() {
        let (lo, hi) = ChromaStandard::Bt709.legal_range_norm();
        assert!(lo > 0.0);
        assert!(hi < 1.0);
    }

    #[test]
    fn test_analyze_chroma_mid_grey_is_legal() {
        // 128,128,128 → Cb=128, Cr=128 → within 16..240 → legal
        let frame = grey_frame(128, 4, 4);
        let result = analyze_chroma_levels(&frame, 4, 4, ChromaStandard::Bt709);
        assert!(result.is_ok());
        let a = result.expect("should succeed");
        assert!(!a.has_illegal_chroma());
        assert_eq!(a.total_pixels, 16);
    }

    #[test]
    fn test_analyze_chroma_zero_pixels() {
        let frame = vec![];
        let result = analyze_chroma_levels(&frame, 0, 0, ChromaStandard::Bt709);
        assert!(result.is_ok());
        let a = result.expect("should succeed");
        assert_eq!(a.total_pixels, 0);
    }

    #[test]
    fn test_analyze_chroma_frame_too_small() {
        let frame = vec![0u8; 10];
        let result = analyze_chroma_levels(&frame, 10, 10, ChromaStandard::Bt709);
        assert!(result.is_err());
    }

    #[test]
    fn test_chroma_analysis_illegal_pct_zero_pixels() {
        let a = ChromaAnalysis {
            avg_cb: 128.0,
            avg_cr: 128.0,
            max_cb: 128,
            max_cr: 128,
            min_cb: 128,
            min_cr: 128,
            illegal_cb_high: 0,
            illegal_cb_low: 0,
            illegal_cr_high: 0,
            illegal_cr_low: 0,
            total_pixels: 0,
        };
        assert_eq!(a.illegal_pct(), 0.0);
    }

    #[test]
    fn test_render_chroma_level_scope_parade_mode() {
        let frame = grey_frame(128, 8, 8);
        let cfg = ChromaLevelConfig {
            width: 64,
            height: 32,
            parade_mode: true,
            ..Default::default()
        };
        let result = render_chroma_level_scope(&frame, 8, 8, &cfg);
        assert!(result.is_ok());
        let data = result.expect("should succeed");
        assert_eq!(data.len(), 64 * 32 * 4);
    }

    #[test]
    fn test_render_chroma_level_scope_overlay_mode() {
        let frame = grey_frame(64, 8, 8);
        let cfg = ChromaLevelConfig {
            width: 64,
            height: 32,
            parade_mode: false,
            ..Default::default()
        };
        let result = render_chroma_level_scope(&frame, 8, 8, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_render_chroma_level_scope_frame_too_small() {
        let frame = vec![0u8; 10];
        let cfg = ChromaLevelConfig::default();
        let result = render_chroma_level_scope(&frame, 10, 10, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_rgb_to_cbcr_neutral_grey() {
        let (cb, cr) = rgb_to_cbcr_bt709(128, 128, 128);
        // Should be near 128
        assert!((cb as i32 - 128).abs() <= 2);
        assert!((cr as i32 - 128).abs() <= 2);
    }

    #[test]
    fn test_chroma_level_config_default() {
        let cfg = ChromaLevelConfig::default();
        assert_eq!(cfg.standard, ChromaStandard::Bt709);
        assert!(cfg.parade_mode);
        assert!(cfg.highlight_illegal);
    }
}
