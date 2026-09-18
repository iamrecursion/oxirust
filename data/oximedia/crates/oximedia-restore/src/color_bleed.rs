#![allow(dead_code)]
//! Colour bleed (chroma bleeding) detection and correction.
//!
//! Analogue video and early digital formats often exhibit colour bleed —
//! chrominance leaking into adjacent luminance regions.  This module provides
//! tools to detect bleed artefacts and correct them by constraining chroma
//! transitions to match the underlying luminance edge structure.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Colour space for internal processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColourSpace {
    /// Standard sRGB.
    Srgb,
    /// YCbCr (BT.601).
    YCbCr601,
    /// YCbCr (BT.709).
    YCbCr709,
}

/// A pixel in (Y, Cb, Cr) form.
#[derive(Debug, Clone, Copy)]
pub struct YCbCrPixel {
    /// Luma component.
    pub y: f64,
    /// Blue-difference chroma.
    pub cb: f64,
    /// Red-difference chroma.
    pub cr: f64,
}

/// A pixel in (R, G, B) form (0..1 range).
#[derive(Debug, Clone, Copy)]
pub struct RgbPixel {
    /// Red channel.
    pub r: f64,
    /// Green channel.
    pub g: f64,
    /// Blue channel.
    pub b: f64,
}

/// Configuration for colour-bleed correction.
#[derive(Debug, Clone)]
pub struct BleedCorrectionConfig {
    /// Kernel radius for the bilateral chroma filter.
    pub kernel_radius: usize,
    /// Spatial sigma for the bilateral filter.
    pub sigma_spatial: f64,
    /// Range sigma for the bilateral filter.
    pub sigma_range: f64,
    /// Threshold for declaring a chroma edge as bleeding.
    pub bleed_threshold: f64,
    /// Maximum chroma shift per pixel (limits correction strength).
    pub max_correction: f64,
}

/// Detected bleed region.
#[derive(Debug, Clone)]
pub struct BleedRegion {
    /// Starting x position.
    pub x: usize,
    /// Starting y position.
    pub y_pos: usize,
    /// Width of the affected region.
    pub width: usize,
    /// Height of the affected region.
    pub height: usize,
    /// Severity metric (0..1).
    pub severity: f64,
}

/// Summary of a bleed correction pass.
#[derive(Debug, Clone)]
pub struct BleedCorrectionReport {
    /// Number of pixels corrected.
    pub corrected_pixels: usize,
    /// Total pixels analysed.
    pub total_pixels: usize,
    /// Average correction magnitude.
    pub avg_correction: f64,
    /// Number of detected bleed regions.
    pub region_count: usize,
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl Default for BleedCorrectionConfig {
    fn default() -> Self {
        Self {
            kernel_radius: 3,
            sigma_spatial: 2.0,
            sigma_range: 0.1,
            bleed_threshold: 0.08,
            max_correction: 0.15,
        }
    }
}

impl RgbPixel {
    /// Create a new RGB pixel.
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Convert to YCbCr using BT.709 matrix.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_ycbcr_709(&self) -> YCbCrPixel {
        let y = 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b;
        let cb = (self.b - y) / 1.8556;
        let cr = (self.r - y) / 1.5748;
        YCbCrPixel { y, cb, cr }
    }
}

impl YCbCrPixel {
    /// Create a new YCbCr pixel.
    pub fn new(y: f64, cb: f64, cr: f64) -> Self {
        Self { y, cb, cr }
    }

    /// Convert back to RGB using BT.709.
    pub fn to_rgb_709(&self) -> RgbPixel {
        let r = self.y + 1.5748 * self.cr;
        let g = self.y - 0.1873 * self.cb - 0.4681 * self.cr;
        let b = self.y + 1.8556 * self.cb;
        RgbPixel {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Chroma magnitude.
    pub fn chroma_magnitude(&self) -> f64 {
        (self.cb * self.cb + self.cr * self.cr).sqrt()
    }
}

/// Detect bleed regions in a YCbCr image buffer.
#[allow(clippy::cast_precision_loss)]
pub fn detect_bleed_regions(
    ycbcr: &[YCbCrPixel],
    width: usize,
    height: usize,
    threshold: f64,
) -> Vec<BleedRegion> {
    let mut regions = Vec::new();
    if ycbcr.is_empty() || width == 0 || height == 0 {
        return regions;
    }

    // Simple horizontal gradient check
    for row in 0..height {
        let mut run_start: Option<usize> = None;
        for col in 1..width.min(ycbcr.len() / height.max(1)) {
            let idx = row * width + col;
            let prev = row * width + col - 1;
            if idx >= ycbcr.len() || prev >= ycbcr.len() {
                continue;
            }
            let dcb = (ycbcr[idx].cb - ycbcr[prev].cb).abs();
            let dcr = (ycbcr[idx].cr - ycbcr[prev].cr).abs();
            let dy = (ycbcr[idx].y - ycbcr[prev].y).abs();

            // Bleed: large chroma gradient with small luma gradient
            if (dcb > threshold || dcr > threshold) && dy < threshold * 0.5 {
                if run_start.is_none() {
                    run_start = Some(col - 1);
                }
            } else if let Some(start) = run_start {
                let w = col - start;
                if w >= 2 {
                    regions.push(BleedRegion {
                        x: start,
                        y_pos: row,
                        width: w,
                        height: 1,
                        severity: (dcb.max(dcr) / threshold).min(1.0),
                    });
                }
                run_start = None;
            }
        }
    }
    regions
}

/// Apply bilateral chroma filter to reduce colour bleed.
#[allow(clippy::cast_precision_loss)]
pub fn correct_bleed(
    ycbcr: &[YCbCrPixel],
    width: usize,
    height: usize,
    config: &BleedCorrectionConfig,
) -> (Vec<YCbCrPixel>, BleedCorrectionReport) {
    let total = ycbcr.len();
    if total == 0 || width == 0 || height == 0 {
        return (
            Vec::new(),
            BleedCorrectionReport {
                corrected_pixels: 0,
                total_pixels: 0,
                avg_correction: 0.0,
                region_count: 0,
            },
        );
    }

    let mut output: Vec<YCbCrPixel> = ycbcr.to_vec();
    let mut corrected = 0usize;
    let mut total_correction = 0.0f64;
    let r = config.kernel_radius;

    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if idx >= total {
                continue;
            }
            let center = ycbcr[idx];
            let mut w_sum = 0.0f64;
            let mut cb_sum = 0.0f64;
            let mut cr_sum = 0.0f64;

            let row_start = row.saturating_sub(r);
            let row_end = (row + r + 1).min(height);
            let col_start = col.saturating_sub(r);
            let col_end = (col + r + 1).min(width);

            for nr in row_start..row_end {
                for nc in col_start..col_end {
                    let nidx = nr * width + nc;
                    if nidx >= total {
                        continue;
                    }
                    let n = ycbcr[nidx];
                    let dr = (nr as f64 - row as f64).powi(2) + (nc as f64 - col as f64).powi(2);
                    let spatial_w =
                        (-dr / (2.0 * config.sigma_spatial * config.sigma_spatial)).exp();
                    let dy = (n.y - center.y).abs();
                    let range_w =
                        (-dy * dy / (2.0 * config.sigma_range * config.sigma_range)).exp();
                    let w = spatial_w * range_w;
                    w_sum += w;
                    cb_sum += w * n.cb;
                    cr_sum += w * n.cr;
                }
            }

            if w_sum > 0.0 {
                let new_cb = cb_sum / w_sum;
                let new_cr = cr_sum / w_sum;
                let dcb = (new_cb - center.cb).abs();
                let dcr = (new_cr - center.cr).abs();
                if dcb > 1e-6 || dcr > 1e-6 {
                    let limited_cb = center.cb
                        + (new_cb - center.cb).clamp(-config.max_correction, config.max_correction);
                    let limited_cr = center.cr
                        + (new_cr - center.cr).clamp(-config.max_correction, config.max_correction);
                    output[idx] = YCbCrPixel {
                        y: center.y,
                        cb: limited_cb,
                        cr: limited_cr,
                    };
                    corrected += 1;
                    total_correction += dcb + dcr;
                }
            }
        }
    }

    let avg_correction = if corrected > 0 {
        total_correction / corrected as f64
    } else {
        0.0
    };
    let regions = detect_bleed_regions(ycbcr, width, height, config.bleed_threshold);

    (
        output,
        BleedCorrectionReport {
            corrected_pixels: corrected,
            total_pixels: total,
            avg_correction,
            region_count: regions.len(),
        },
    )
}

/// Compute chroma difference map between two YCbCr buffers.
pub fn chroma_diff_map(a: &[YCbCrPixel], b: &[YCbCrPixel]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(pa, pb)| {
            let dcb = pa.cb - pb.cb;
            let dcr = pa.cr - pb.cr;
            (dcb * dcb + dcr * dcr).sqrt()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_ycbcr_black() {
        let p = RgbPixel::new(0.0, 0.0, 0.0).to_ycbcr_709();
        assert!(p.y.abs() < 1e-9);
        assert!(p.cb.abs() < 1e-9);
        assert!(p.cr.abs() < 1e-9);
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let p = RgbPixel::new(1.0, 1.0, 1.0).to_ycbcr_709();
        assert!((p.y - 1.0).abs() < 1e-6);
        assert!(p.cb.abs() < 1e-6);
        assert!(p.cr.abs() < 1e-6);
    }

    #[test]
    fn test_ycbcr_roundtrip() {
        let orig = RgbPixel::new(0.5, 0.3, 0.7);
        let ycc = orig.to_ycbcr_709();
        let back = ycc.to_rgb_709();
        assert!((orig.r - back.r).abs() < 0.05);
        assert!((orig.g - back.g).abs() < 0.05);
        assert!((orig.b - back.b).abs() < 0.05);
    }

    #[test]
    fn test_chroma_magnitude() {
        let p = YCbCrPixel::new(0.5, 0.3, 0.4);
        assert!((p.chroma_magnitude() - 0.5) < 0.01);
    }

    #[test]
    fn test_default_config() {
        let c = BleedCorrectionConfig::default();
        assert_eq!(c.kernel_radius, 3);
        assert!(c.sigma_spatial > 0.0);
    }

    #[test]
    fn test_detect_bleed_empty() {
        let regions = detect_bleed_regions(&[], 0, 0, 0.08);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_detect_bleed_uniform() {
        let px = vec![YCbCrPixel::new(0.5, 0.0, 0.0); 16];
        let regions = detect_bleed_regions(&px, 4, 4, 0.08);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_correct_bleed_empty() {
        let (out, report) = correct_bleed(&[], 0, 0, &BleedCorrectionConfig::default());
        assert!(out.is_empty());
        assert_eq!(report.total_pixels, 0);
    }

    #[test]
    fn test_correct_bleed_preserves_length() {
        let px = vec![YCbCrPixel::new(0.5, 0.1, -0.1); 9];
        let (out, report) = correct_bleed(&px, 3, 3, &BleedCorrectionConfig::default());
        assert_eq!(out.len(), 9);
        assert_eq!(report.total_pixels, 9);
    }

    #[test]
    fn test_correct_bleed_uniform_no_change() {
        let px = vec![YCbCrPixel::new(0.5, 0.2, 0.2); 25];
        let (out, _) = correct_bleed(&px, 5, 5, &BleedCorrectionConfig::default());
        for (o, p) in out.iter().zip(px.iter()) {
            assert!((o.cb - p.cb).abs() < 1e-6);
            assert!((o.cr - p.cr).abs() < 1e-6);
        }
    }

    #[test]
    fn test_chroma_diff_map_identical() {
        let px = vec![YCbCrPixel::new(0.5, 0.1, 0.1); 10];
        let diff = chroma_diff_map(&px, &px);
        assert_eq!(diff.len(), 10);
        for d in &diff {
            assert!(d.abs() < 1e-12);
        }
    }

    #[test]
    fn test_chroma_diff_map_different() {
        let a = vec![YCbCrPixel::new(0.5, 0.0, 0.0)];
        let b = vec![YCbCrPixel::new(0.5, 0.3, 0.4)];
        let diff = chroma_diff_map(&a, &b);
        assert!(diff[0] > 0.49);
    }

    #[test]
    fn test_bleed_region_severity_clamped() {
        // Create a sharp chroma step
        let mut px = vec![YCbCrPixel::new(0.5, 0.0, 0.0); 10];
        px[5] = YCbCrPixel::new(0.5, 0.5, 0.5);
        px[6] = YCbCrPixel::new(0.5, 0.5, 0.5);
        let regions = detect_bleed_regions(&px, 10, 1, 0.08);
        for r in &regions {
            assert!(r.severity <= 1.0);
        }
    }

    #[test]
    fn test_rgb_pixel_new() {
        let p = RgbPixel::new(0.1, 0.2, 0.3);
        assert!((p.r - 0.1).abs() < 1e-9);
        assert!((p.g - 0.2).abs() < 1e-9);
        assert!((p.b - 0.3).abs() < 1e-9);
    }
}
