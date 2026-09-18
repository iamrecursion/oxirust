//! Exposure histogram with Ansel Adams Zone System overlay for cinematography.
//!
//! This module extends the standard luminance histogram with an Ansel Adams
//! Zone System overlay.  The Zone System divides the tonal range into eleven
//! zones (0–X):
//!
//! | Zone | IRE    | Description              |
//! |------|--------|--------------------------|
//! | 0    | 0      | Pure black               |
//! | I    | 5      | Near black, no texture   |
//! | II   | 11     | Darkest textured shadow   |
//! | III  | 17–22  | Dark shadow with texture |
//! | IV   | 29–38  | Dark subject             |
//! | V    | 50     | Middle gray (18% gray)   |
//! | VI   | 62     | Light skin tones         |
//! | VII  | 76     | Light subjects, highlights|
//! | VIII | 87     | Textured white           |
//! | IX   | 94     | Near white, no texture   |
//! | X    | 100    | Specular white           |
//!
//! The histogram is rendered with each zone coloured distinctly, enabling
//! cinematographers to assess exposure at a glance.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Zone System
// ─────────────────────────────────────────────────────────────────────────────

/// An Ansel Adams tonal zone with its boundaries and display colour.
#[derive(Debug, Clone, Copy)]
pub struct Zone {
    /// Zone number (0–10).
    pub number: u8,
    /// IRE lower bound (0–100).
    pub ire_low: f32,
    /// IRE upper bound (0–100).
    pub ire_high: f32,
    /// RGBA display colour for this zone.
    pub color: [u8; 4],
    /// Short description.
    pub description: &'static str,
}

impl Zone {
    /// Returns the 11 standard Ansel Adams zones.
    #[must_use]
    pub fn all() -> [Self; 11] {
        [
            Self {
                number: 0,
                ire_low: 0.0,
                ire_high: 2.0,
                color: [10, 10, 10, 255],
                description: "Pure black",
            },
            Self {
                number: 1,
                ire_low: 2.0,
                ire_high: 7.0,
                color: [30, 30, 30, 255],
                description: "Near black",
            },
            Self {
                number: 2,
                ire_low: 7.0,
                ire_high: 13.0,
                color: [55, 40, 55, 255],
                description: "Darkest textured shadow",
            },
            Self {
                number: 3,
                ire_low: 13.0,
                ire_high: 25.0,
                color: [60, 80, 120, 255],
                description: "Dark shadow with texture",
            },
            Self {
                number: 4,
                ire_low: 25.0,
                ire_high: 40.0,
                color: [40, 110, 160, 255],
                description: "Dark subject",
            },
            Self {
                number: 5,
                ire_low: 40.0,
                ire_high: 60.0,
                color: [50, 160, 80, 255],
                description: "Middle gray",
            },
            Self {
                number: 6,
                ire_low: 60.0,
                ire_high: 72.0,
                color: [180, 180, 60, 255],
                description: "Light skin tones",
            },
            Self {
                number: 7,
                ire_low: 72.0,
                ire_high: 82.0,
                color: [220, 140, 40, 255],
                description: "Light subjects",
            },
            Self {
                number: 8,
                ire_low: 82.0,
                ire_high: 92.0,
                color: [230, 90, 60, 255],
                description: "Textured white",
            },
            Self {
                number: 9,
                ire_low: 92.0,
                ire_high: 97.0,
                color: [220, 50, 50, 255],
                description: "Near white",
            },
            Self {
                number: 10,
                ire_low: 97.0,
                ire_high: 100.0,
                color: [250, 250, 250, 255],
                description: "Specular white",
            },
        ]
    }

    /// Returns the zone number for a given normalised luma value (0.0–1.0).
    #[must_use]
    pub fn classify(luma_norm: f32) -> u8 {
        let ire = (luma_norm * 100.0).clamp(0.0, 100.0);
        for z in Self::all().iter().rev() {
            if ire >= z.ire_low {
                return z.number;
            }
        }
        0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Scale mode for the histogram vertical axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramScale {
    /// Linear count.
    Linear,
    /// Natural logarithm of count.
    Logarithmic,
    /// Square root of count (good visual balance).
    Sqrt,
}

/// Configuration for `ExposureHistogram`.
#[derive(Debug, Clone)]
pub struct ExposureHistogramConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Number of histogram bins (typically 256 for 8-bit).
    pub bins: usize,
    /// Whether to draw the Zone System colour bands.
    pub show_zones: bool,
    /// Opacity of the zone bands (0 = invisible, 255 = opaque).
    pub zone_opacity: u8,
    /// Vertical scale mode.
    pub scale: HistogramScale,
    /// Whether to draw the 18%-grey reference line.
    pub show_middle_grey: bool,
    /// Whether to draw broadcast legal lines (16 and 235 for 8-bit).
    pub show_legal_limits: bool,
    /// Background RGBA colour.
    pub background_color: [u8; 4],
    /// Histogram bar RGBA colour.
    pub bar_color: [u8; 4],
}

impl Default for ExposureHistogramConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 256,
            bins: 256,
            show_zones: true,
            zone_opacity: 60,
            scale: HistogramScale::Sqrt,
            show_middle_grey: true,
            show_legal_limits: true,
            background_color: [10, 10, 12, 255],
            bar_color: [200, 200, 200, 200],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Histogram data
// ─────────────────────────────────────────────────────────────────────────────

/// Per-zone pixel count distribution.
#[derive(Debug, Clone)]
pub struct ZoneDistribution {
    /// Pixel counts for each of the 11 zones.
    pub zone_counts: [u64; 11],
    /// Total pixels analysed.
    pub total_pixels: u64,
}

impl ZoneDistribution {
    /// Returns the zone with the highest pixel count.
    #[must_use]
    pub fn dominant_zone(&self) -> u8 {
        self.zone_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map_or(5, |(idx, _)| idx as u8)
    }

    /// Returns the fraction (0–1) of pixels in zone `z`.
    #[must_use]
    pub fn zone_fraction(&self, zone: u8) -> f32 {
        if self.total_pixels == 0 || zone > 10 {
            return 0.0;
        }
        self.zone_counts[zone as usize] as f32 / self.total_pixels as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Histogram renderer
// ─────────────────────────────────────────────────────────────────────────────

/// Build a luma histogram from an RGB24 frame and classify pixels into zones.
///
/// # Arguments
///
/// * `frame` — RGB24 pixel data, length must be `width * height * 3`.
/// * `width` / `height` — frame dimensions.
///
/// # Errors
///
/// Returns an error if the frame buffer is too small.
#[must_use]
#[allow(clippy::integer_division)]
pub fn build_zone_histogram(
    frame: &[u8],
    width: u32,
    height: u32,
) -> OxiResult<(Vec<u64>, ZoneDistribution)> {
    let num_pixels = (width as usize) * (height as usize);
    let expected = num_pixels * 3;
    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame too small: need {expected}, got {}",
            frame.len()
        )));
    }

    let mut hist = vec![0u64; 256];
    let mut zone_counts = [0u64; 11];

    for i in 0..num_pixels {
        let r = frame[i * 3] as f32;
        let g = frame[i * 3 + 1] as f32;
        let b = frame[i * 3 + 2] as f32;
        // BT.709 luma
        let y_f = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let y = y_f.round() as usize;
        let y_clamped = y.min(255);
        hist[y_clamped] += 1;
        let zone = Zone::classify(y_f / 255.0);
        zone_counts[zone as usize] += 1;
    }

    let dist = ZoneDistribution {
        zone_counts,
        total_pixels: num_pixels as u64,
    };
    Ok((hist, dist))
}

/// Render the exposure histogram with Zone System overlay as an RGBA image.
///
/// # Errors
///
/// Propagates errors from `build_zone_histogram`.
pub fn render_exposure_histogram(
    frame: &[u8],
    frame_w: u32,
    frame_h: u32,
    config: &ExposureHistogramConfig,
) -> OxiResult<Vec<u8>> {
    let (hist, _dist) = build_zone_histogram(frame, frame_w, frame_h)?;

    let w = config.width as usize;
    let h = config.height as usize;
    let mut pixels = vec![0u8; w * h * 4];

    // Background
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.copy_from_slice(&config.background_color);
    }

    // Zone bands
    if config.show_zones {
        let zones = Zone::all();
        for zone in &zones {
            let x_lo = ((zone.ire_low / 100.0 * w as f32).round() as usize).min(w);
            let x_hi = ((zone.ire_high / 100.0 * w as f32).round() as usize).min(w);
            let mut col = zone.color;
            col[3] = config.zone_opacity;
            for py in 0..h {
                for px in x_lo..x_hi {
                    blend_pixel_slice(&mut pixels, w, px, py, col);
                }
            }
        }
    }

    // Histogram bars
    let bins = config.bins.min(256).max(1);
    // Find max value for scaling
    let bin_size = 256 / bins;
    let mut binned: Vec<u64> = vec![0; bins];
    for (i, &count) in hist.iter().enumerate() {
        binned[i / bin_size.max(1)] += count;
    }
    let max_val = binned.iter().copied().max().unwrap_or(1);

    for (bin_idx, &count) in binned.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let norm_height = match config.scale {
            HistogramScale::Linear => count as f32 / max_val as f32,
            HistogramScale::Logarithmic => (count as f32 + 1.0).ln() / (max_val as f32 + 1.0).ln(),
            HistogramScale::Sqrt => (count as f32 / max_val as f32).sqrt(),
        };
        let bar_h = (norm_height * h as f32).round() as usize;
        let x_lo = (bin_idx * w) / bins;
        let x_hi = ((bin_idx + 1) * w) / bins;

        for px in x_lo..x_hi {
            for py in (h - bar_h)..h {
                blend_pixel_slice(&mut pixels, w, px, py, config.bar_color);
            }
        }
    }

    // Middle grey reference (Y = 128 for 8-bit, IRE ≈ 50)
    if config.show_middle_grey {
        let mg_x = (128 * w) / 256;
        let mg_color = [80u8, 200, 80, 220];
        for py in 0..h {
            blend_pixel_slice(&mut pixels, w, mg_x, py, mg_color);
        }
    }

    // Broadcast legal limits (16 and 235 for 8-bit)
    if config.show_legal_limits {
        let lo_x = (16 * w) / 256;
        let hi_x = (235 * w) / 256;
        let limit_color = [220u8, 60, 60, 200];
        for py in 0..h {
            blend_pixel_slice(&mut pixels, w, lo_x, py, limit_color);
            blend_pixel_slice(&mut pixels, w, hi_x, py, limit_color);
        }
    }

    Ok(pixels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

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
    fn test_zone_classify_black() {
        assert_eq!(Zone::classify(0.0), 0);
    }

    #[test]
    fn test_zone_classify_middle_grey() {
        // IRE 50 → zone V (5)
        assert_eq!(Zone::classify(0.5), 5);
    }

    #[test]
    fn test_zone_classify_white() {
        assert_eq!(Zone::classify(1.0), 10);
    }

    #[test]
    fn test_zone_all_has_11_zones() {
        assert_eq!(Zone::all().len(), 11);
    }

    #[test]
    fn test_zone_numbers_sequential() {
        for (i, z) in Zone::all().iter().enumerate() {
            assert_eq!(z.number as usize, i);
        }
    }

    #[test]
    fn test_build_zone_histogram_all_black() {
        let frame = grey_frame(0, 4, 4);
        let (hist, dist) = build_zone_histogram(&frame, 4, 4).expect("should succeed");
        assert_eq!(hist[0], 16);
        assert!(hist[1..].iter().all(|&c| c == 0));
        assert_eq!(dist.zone_counts[0], 16);
        assert_eq!(dist.total_pixels, 16);
    }

    #[test]
    fn test_build_zone_histogram_all_white() {
        let frame = grey_frame(255, 4, 4);
        let (hist, dist) = build_zone_histogram(&frame, 4, 4).expect("should succeed");
        assert_eq!(hist[255], 16);
        assert_eq!(dist.zone_counts[10], 16);
    }

    #[test]
    fn test_build_zone_histogram_middle_grey() {
        let frame = grey_frame(128, 4, 4);
        let (_hist, dist) = build_zone_histogram(&frame, 4, 4).expect("should succeed");
        // Zone 5 = middle grey
        assert!(dist.zone_counts[5] > 0);
    }

    #[test]
    fn test_build_zone_histogram_frame_too_small() {
        let frame = vec![0u8; 10];
        let result = build_zone_histogram(&frame, 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_zone_distribution_dominant_zone() {
        let mut dist = ZoneDistribution {
            zone_counts: [0; 11],
            total_pixels: 100,
        };
        dist.zone_counts[5] = 80;
        assert_eq!(dist.dominant_zone(), 5);
    }

    #[test]
    fn test_zone_distribution_fraction() {
        let mut dist = ZoneDistribution {
            zone_counts: [0; 11],
            total_pixels: 200,
        };
        dist.zone_counts[3] = 50;
        assert!((dist.zone_fraction(3) - 0.25).abs() < 1e-5);
        assert_eq!(dist.zone_fraction(11), 0.0); // out of range
    }

    #[test]
    fn test_render_exposure_histogram_produces_correct_size() {
        let frame = grey_frame(128, 8, 8);
        let cfg = ExposureHistogramConfig {
            width: 64,
            height: 32,
            ..Default::default()
        };
        let result = render_exposure_histogram(&frame, 8, 8, &cfg);
        assert!(result.is_ok());
        let data = result.expect("should succeed");
        assert_eq!(data.len(), 64 * 32 * 4);
    }

    #[test]
    fn test_render_exposure_histogram_with_zones_disabled() {
        let frame = grey_frame(100, 8, 8);
        let cfg = ExposureHistogramConfig {
            width: 32,
            height: 32,
            show_zones: false,
            show_middle_grey: false,
            show_legal_limits: false,
            ..Default::default()
        };
        let result = render_exposure_histogram(&frame, 8, 8, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_render_exposure_histogram_log_scale() {
        let frame = grey_frame(64, 8, 8);
        let cfg = ExposureHistogramConfig {
            width: 64,
            height: 32,
            scale: HistogramScale::Logarithmic,
            ..Default::default()
        };
        let result = render_exposure_histogram(&frame, 8, 8, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_render_exposure_histogram_linear_scale() {
        let frame = grey_frame(200, 8, 8);
        let cfg = ExposureHistogramConfig {
            width: 64,
            height: 32,
            scale: HistogramScale::Linear,
            ..Default::default()
        };
        let result = render_exposure_histogram(&frame, 8, 8, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exposure_histogram_config_default() {
        let cfg = ExposureHistogramConfig::default();
        assert_eq!(cfg.bins, 256);
        assert!(cfg.show_zones);
        assert_eq!(cfg.scale, HistogramScale::Sqrt);
    }
}
