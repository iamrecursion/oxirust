#![allow(dead_code)]
//! Exposure metering for video frames.
//!
//! This module provides exposure analysis tools similar to a camera's built-in
//! light meter. It supports spot, center-weighted, evaluative (matrix), and
//! average metering modes. The exposure value (EV) is computed from the
//! measured luminance, and over/under-exposure warnings are generated.

use std::f64;

/// Metering mode used to weight different regions of the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeteringMode {
    /// Average (whole frame equally weighted).
    Average,
    /// Center-weighted (center of frame weighted more heavily).
    CenterWeighted,
    /// Spot (small central region only).
    Spot,
    /// Evaluative / matrix (multi-zone intelligent metering).
    Evaluative,
}

/// Configuration for the exposure meter.
#[derive(Debug, Clone)]
pub struct ExposureMeterConfig {
    /// Metering mode.
    pub mode: MeteringMode,
    /// Spot meter radius as fraction of frame diagonal (0.0 - 0.5).
    pub spot_radius: f64,
    /// Over-exposure threshold (0 - 255 for 8-bit, luma).
    pub over_threshold: f64,
    /// Under-exposure threshold (0 - 255 for 8-bit, luma).
    pub under_threshold: f64,
    /// Number of evaluation zones per axis (for evaluative mode).
    pub eval_zones: u32,
}

impl Default for ExposureMeterConfig {
    fn default() -> Self {
        Self {
            mode: MeteringMode::CenterWeighted,
            spot_radius: 0.05,
            over_threshold: 235.0,
            under_threshold: 16.0,
            eval_zones: 4,
        }
    }
}

/// Exposure metering result.
#[derive(Debug, Clone)]
pub struct ExposureReading {
    /// Measured average luminance (0.0 - 255.0 scale).
    pub avg_luminance: f64,
    /// Estimated exposure value (EV) relative to 18% gray.
    pub ev_estimate: f64,
    /// Fraction of pixels that are over-exposed (0.0 - 1.0).
    pub over_exposed_ratio: f64,
    /// Fraction of pixels that are under-exposed (0.0 - 1.0).
    pub under_exposed_ratio: f64,
    /// Exposure recommendation: negative means overexposed, positive means underexposed.
    pub correction_stops: f64,
    /// Number of pixels metered.
    pub metered_pixels: u64,
    /// Per-zone luminance values (for evaluative mode).
    pub zone_luminances: Vec<f64>,
}

/// Compute Rec.709 luma from 8-bit RGB.
#[must_use]
fn rgb_to_luma(r: u8, g: u8, b: u8) -> f64 {
    0.2126 * f64::from(r) + 0.7152 * f64::from(g) + 0.0722 * f64::from(b)
}

/// Compute the weight for center-weighted metering based on distance from center.
#[must_use]
fn center_weight(x: u32, y: u32, width: u32, height: u32) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let cx = width as f64 / 2.0;
    #[allow(clippy::cast_precision_loss)]
    let cy = height as f64 / 2.0;
    #[allow(clippy::cast_precision_loss)]
    let dx = (x as f64 - cx) / cx;
    #[allow(clippy::cast_precision_loss)]
    let dy = (y as f64 - cy) / cy;
    let dist = (dx * dx + dy * dy).sqrt();
    // Gaussian-like falloff
    (-2.0 * dist * dist).exp()
}

/// Compute whether a pixel is inside the spot metering circle.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn in_spot(x: u32, y: u32, width: u32, height: u32, radius_frac: f64) -> bool {
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let diag = ((width as f64).powi(2) + (height as f64).powi(2)).sqrt();
    let radius = radius_frac * diag;
    let dx = x as f64 - cx;
    let dy = y as f64 - cy;
    (dx * dx + dy * dy).sqrt() <= radius
}

/// Estimate EV from average luminance (0-255 scale).
///
/// Maps 18% gray (~46 on 0-255 scale) to EV 0.
#[must_use]
fn luminance_to_ev(avg_luma: f64) -> f64 {
    if avg_luma <= 0.0 {
        return -10.0;
    }
    // 18% gray corresponds to ~46 on 0-255 scale
    let gray_18 = 255.0 * 0.18;
    (avg_luma / gray_18).log2()
}

/// Meter a frame using the given configuration.
///
/// * `frame` — RGB24 pixel data
/// * `width` — frame width
/// * `height` — frame height
/// * `config` — metering configuration
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn meter_exposure(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ExposureMeterConfig,
) -> ExposureReading {
    let num_pixels = (width as usize) * (height as usize);
    let expected = num_pixels * 3;

    if frame.len() < expected || num_pixels == 0 {
        return ExposureReading {
            avg_luminance: 0.0,
            ev_estimate: -10.0,
            over_exposed_ratio: 0.0,
            under_exposed_ratio: 0.0,
            correction_stops: 0.0,
            metered_pixels: 0,
            zone_luminances: Vec::new(),
        };
    }

    match config.mode {
        MeteringMode::Average => meter_average(frame, width, height, config),
        MeteringMode::CenterWeighted => meter_center_weighted(frame, width, height, config),
        MeteringMode::Spot => meter_spot(frame, width, height, config),
        MeteringMode::Evaluative => meter_evaluative(frame, width, height, config),
    }
}

/// Average metering: all pixels weighted equally.
#[allow(clippy::cast_precision_loss)]
fn meter_average(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ExposureMeterConfig,
) -> ExposureReading {
    let n = (width as usize) * (height as usize);
    let mut sum = 0.0_f64;
    let mut over = 0_u64;
    let mut under = 0_u64;

    for i in 0..n {
        let luma = rgb_to_luma(frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]);
        sum += luma;
        if luma >= config.over_threshold {
            over += 1;
        }
        if luma <= config.under_threshold {
            under += 1;
        }
    }

    let avg = sum / n as f64;
    let ev = luminance_to_ev(avg);
    let metered = n as u64;

    ExposureReading {
        avg_luminance: avg,
        ev_estimate: ev,
        over_exposed_ratio: over as f64 / metered as f64,
        under_exposed_ratio: under as f64 / metered as f64,
        correction_stops: -ev,
        metered_pixels: metered,
        zone_luminances: Vec::new(),
    }
}

/// Center-weighted metering.
#[allow(clippy::cast_precision_loss)]
fn meter_center_weighted(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ExposureMeterConfig,
) -> ExposureReading {
    let n = (width as usize) * (height as usize);
    let mut weighted_sum = 0.0_f64;
    let mut weight_total = 0.0_f64;
    let mut over = 0_u64;
    let mut under = 0_u64;

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let luma = rgb_to_luma(frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]);
            let w = center_weight(x, y, width, height);
            weighted_sum += luma * w;
            weight_total += w;
            if luma >= config.over_threshold {
                over += 1;
            }
            if luma <= config.under_threshold {
                under += 1;
            }
        }
    }

    let avg = if weight_total > 0.0 {
        weighted_sum / weight_total
    } else {
        0.0
    };
    let ev = luminance_to_ev(avg);
    let metered = n as u64;

    ExposureReading {
        avg_luminance: avg,
        ev_estimate: ev,
        over_exposed_ratio: over as f64 / metered as f64,
        under_exposed_ratio: under as f64 / metered as f64,
        correction_stops: -ev,
        metered_pixels: metered,
        zone_luminances: Vec::new(),
    }
}

/// Spot metering: only a small central circle.
#[allow(clippy::cast_precision_loss)]
fn meter_spot(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ExposureMeterConfig,
) -> ExposureReading {
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    let mut over = 0_u64;
    let mut under = 0_u64;

    for y in 0..height {
        for x in 0..width {
            if !in_spot(x, y, width, height, config.spot_radius) {
                continue;
            }
            let i = (y * width + x) as usize;
            let luma = rgb_to_luma(frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]);
            sum += luma;
            count += 1;
            if luma >= config.over_threshold {
                over += 1;
            }
            if luma <= config.under_threshold {
                under += 1;
            }
        }
    }

    let avg = if count > 0 { sum / count as f64 } else { 0.0 };
    let ev = luminance_to_ev(avg);
    let denom = if count > 0 { count } else { 1 };

    ExposureReading {
        avg_luminance: avg,
        ev_estimate: ev,
        over_exposed_ratio: over as f64 / denom as f64,
        under_exposed_ratio: under as f64 / denom as f64,
        correction_stops: -ev,
        metered_pixels: count,
        zone_luminances: Vec::new(),
    }
}

/// Evaluative (matrix) metering: divides frame into zones.
#[allow(clippy::cast_precision_loss)]
fn meter_evaluative(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ExposureMeterConfig,
) -> ExposureReading {
    let total_px = (width as usize) * (height as usize);
    let zones = config.eval_zones.max(1);
    let zone_w = width / zones;
    let zone_h = height / zones;
    let zone_count = (zones * zones) as usize;

    let mut zone_sums = vec![0.0_f64; zone_count];
    let mut zone_counts = vec![0_u64; zone_count];
    let mut over = 0_u64;
    let mut under = 0_u64;
    let mut total_sum = 0.0_f64;

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let luma = rgb_to_luma(frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]);
            total_sum += luma;

            let zx = (x / zone_w.max(1)).min(zones - 1) as usize;
            let zy = (y / zone_h.max(1)).min(zones - 1) as usize;
            let zi = zy * zones as usize + zx;
            zone_sums[zi] += luma;
            zone_counts[zi] += 1;

            if luma >= config.over_threshold {
                over += 1;
            }
            if luma <= config.under_threshold {
                under += 1;
            }
        }
    }

    let zone_luminances: Vec<f64> = zone_sums
        .iter()
        .zip(zone_counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    let avg = total_sum / total_px as f64;
    let ev = luminance_to_ev(avg);
    let metered = total_px as u64;

    ExposureReading {
        avg_luminance: avg,
        ev_estimate: ev,
        over_exposed_ratio: over as f64 / metered as f64,
        under_exposed_ratio: under as f64 / metered as f64,
        correction_stops: -ev,
        metered_pixels: metered,
        zone_luminances,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(n * 3);
        for _ in 0..n {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        data
    }

    #[test]
    fn test_rgb_to_luma_black() {
        assert!((rgb_to_luma(0, 0, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rgb_to_luma_white() {
        let y = rgb_to_luma(255, 255, 255);
        assert!((y - 255.0).abs() < 0.01);
    }

    #[test]
    fn test_luminance_to_ev_18_gray() {
        let gray = 255.0 * 0.18;
        let ev = luminance_to_ev(gray);
        assert!(ev.abs() < 0.01);
    }

    #[test]
    fn test_luminance_to_ev_zero() {
        let ev = luminance_to_ev(0.0);
        assert!(ev < -5.0);
    }

    #[test]
    fn test_center_weight_at_center() {
        let w = center_weight(50, 50, 100, 100);
        assert!((w - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_center_weight_at_corner() {
        let w = center_weight(0, 0, 100, 100);
        assert!(w < 0.5);
    }

    #[test]
    fn test_in_spot_center() {
        assert!(in_spot(50, 50, 100, 100, 0.1));
    }

    #[test]
    fn test_in_spot_corner() {
        assert!(!in_spot(0, 0, 100, 100, 0.05));
    }

    #[test]
    fn test_meter_average_black() {
        let frame = make_frame(10, 10, 0, 0, 0);
        let config = ExposureMeterConfig {
            mode: MeteringMode::Average,
            ..Default::default()
        };
        let reading = meter_exposure(&frame, 10, 10, &config);
        assert!(reading.avg_luminance.abs() < f64::EPSILON);
        assert_eq!(reading.metered_pixels, 100);
    }

    #[test]
    fn test_meter_average_midgray() {
        let frame = make_frame(10, 10, 128, 128, 128);
        let config = ExposureMeterConfig {
            mode: MeteringMode::Average,
            ..Default::default()
        };
        let reading = meter_exposure(&frame, 10, 10, &config);
        assert!((reading.avg_luminance - 128.0).abs() < 1.0);
    }

    #[test]
    fn test_meter_center_weighted() {
        let frame = make_frame(20, 20, 100, 100, 100);
        let config = ExposureMeterConfig {
            mode: MeteringMode::CenterWeighted,
            ..Default::default()
        };
        let reading = meter_exposure(&frame, 20, 20, &config);
        assert!((reading.avg_luminance - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_meter_spot() {
        let frame = make_frame(100, 100, 200, 200, 200);
        let config = ExposureMeterConfig {
            mode: MeteringMode::Spot,
            spot_radius: 0.1,
            ..Default::default()
        };
        let reading = meter_exposure(&frame, 100, 100, &config);
        assert!(reading.metered_pixels < 10000);
        assert!((reading.avg_luminance - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_meter_evaluative() {
        let frame = make_frame(16, 16, 128, 128, 128);
        let config = ExposureMeterConfig {
            mode: MeteringMode::Evaluative,
            eval_zones: 4,
            ..Default::default()
        };
        let reading = meter_exposure(&frame, 16, 16, &config);
        assert_eq!(reading.zone_luminances.len(), 16);
        for &z in &reading.zone_luminances {
            assert!((z - 128.0).abs() < 1.0);
        }
    }

    #[test]
    fn test_meter_empty_frame() {
        let config = ExposureMeterConfig::default();
        let reading = meter_exposure(&[], 10, 10, &config);
        assert_eq!(reading.metered_pixels, 0);
    }

    #[test]
    fn test_over_exposure_detection() {
        let frame = make_frame(4, 4, 250, 250, 250);
        let config = ExposureMeterConfig {
            mode: MeteringMode::Average,
            over_threshold: 235.0,
            ..Default::default()
        };
        let reading = meter_exposure(&frame, 4, 4, &config);
        assert!((reading.over_exposed_ratio - 1.0).abs() < f64::EPSILON);
    }
}

// ============================================================================
// Enhanced Exposure Metering: ExposureMeter / new MeteringMode variants
// ============================================================================

/// Extended metering mode with highlight/shadow bias options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeteringModeEx {
    /// Multi-zone weighted average (9×9 grid, perimeter zones at 50% weight).
    Evaluative,
    /// Gaussian-weighted toward the image center (σ = 0.3 × min(w,h)).
    CenterWeighted,
    /// 5% central circle only.
    Spot,
    /// Bias toward highlights: weight = luma².
    Highlight,
    /// Bias toward shadows: weight = (1 − luma)².
    Shadow,
}

/// Camera-style exposure meter.
pub struct ExposureMeter {
    /// Active metering mode.
    pub mode: MeteringModeEx,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// Exposure reading produced by [`ExposureMeter::measure`].
#[derive(Debug, Clone)]
pub struct ExposureReadingEx {
    /// Exposure value (EV); 0 = 18% grey correctly exposed.
    pub ev: f32,
    /// Stops from correct exposure (negative = under, positive = over).
    pub stops_from_correct: f32,
    /// Gain adjustment in dB needed to reach correct exposure.
    pub suggested_gain_db: f32,
    /// Mean luminance of the metered area (linear 0–1).
    pub luminance_avg: f32,
    /// Peak luminance in the metered area.
    pub luminance_max: f32,
    /// Minimum luminance in the metered area.
    pub luminance_min: f32,
    /// Fraction of pixels brighter than 0.95.
    pub clipping_pct: f32,
    /// Fraction of pixels darker than 0.02.
    pub crushed_pct: f32,
}

impl ExposureMeter {
    /// Creates a new meter with the specified mode and frame dimensions.
    #[must_use]
    pub fn new(mode: MeteringModeEx, width: u32, height: u32) -> Self {
        Self {
            mode,
            width,
            height,
        }
    }

    /// Measures exposure from a normalised luma slice (values 0.0–1.0,
    /// length == `width × height`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn measure(&self, luma: &[f32]) -> ExposureReadingEx {
        let n = (self.width as usize) * (self.height as usize);
        if luma.len() < n || n == 0 {
            return Self::zero_reading();
        }

        // Global statistics needed for all modes
        let mut luma_max = 0.0_f32;
        let mut luma_min = 1.0_f32;
        let mut clipping = 0u64;
        let mut crushed = 0u64;
        for &v in &luma[..n] {
            if v > luma_max {
                luma_max = v;
            }
            if v < luma_min {
                luma_min = v;
            }
            if v > 0.95 {
                clipping += 1;
            }
            if v < 0.02 {
                crushed += 1;
            }
        }

        let avg = match self.mode {
            MeteringModeEx::Evaluative => self.average_evaluative(luma, n),
            MeteringModeEx::CenterWeighted => self.average_center_weighted(luma),
            MeteringModeEx::Spot => self.average_spot(luma),
            MeteringModeEx::Highlight => Self::average_highlight(luma, n),
            MeteringModeEx::Shadow => Self::average_shadow(luma, n),
        };

        let ev = if avg > 0.0 {
            (avg / 0.18).log2()
        } else {
            -10.0
        };
        let stops_from_correct = ev;
        let suggested_gain_db = -ev * 6.0;
        let total = n as f32;

        ExposureReadingEx {
            ev,
            stops_from_correct,
            suggested_gain_db,
            luminance_avg: avg,
            luminance_max: luma_max,
            luminance_min: luma_min,
            clipping_pct: clipping as f32 / total,
            crushed_pct: crushed as f32 / total,
        }
    }

    /// Returns the linear gain factor to apply for 18% grey correct exposure.
    ///
    /// A returned value of `1.0` means no adjustment needed.
    #[must_use]
    pub fn auto_exposure_gain(&self, luma: &[f32]) -> f32 {
        let reading = self.measure(luma);
        if reading.luminance_avg > 0.0 {
            0.18 / reading.luminance_avg
        } else {
            1.0
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn zero_reading() -> ExposureReadingEx {
        ExposureReadingEx {
            ev: -10.0,
            stops_from_correct: -10.0,
            suggested_gain_db: 60.0,
            luminance_avg: 0.0,
            luminance_max: 0.0,
            luminance_min: 0.0,
            clipping_pct: 0.0,
            crushed_pct: 0.0,
        }
    }

    /// Evaluative: 9×9 grid, perimeter zones weighted at 50%.
    #[allow(clippy::cast_precision_loss)]
    fn average_evaluative(&self, luma: &[f32], n: usize) -> f32 {
        const ZONES: u32 = 9;
        let zone_w = (self.width / ZONES).max(1);
        let zone_h = (self.height / ZONES).max(1);

        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;

        for idx in 0..n {
            let x = (idx as u32) % self.width;
            let y = (idx as u32) / self.width;
            let zx = (x / zone_w).min(ZONES - 1);
            let zy = (y / zone_h).min(ZONES - 1);
            // Perimeter zones get 50% weight
            let is_perimeter = zx == 0 || zx == ZONES - 1 || zy == 0 || zy == ZONES - 1;
            let w: f64 = if is_perimeter { 0.5 } else { 1.0 };
            weighted_sum += f64::from(luma[idx]) * w;
            weight_total += w;
        }

        if weight_total > 0.0 {
            (weighted_sum / weight_total) as f32
        } else {
            0.0
        }
    }

    /// Center-weighted: Gaussian with σ = 0.3 × min(w, h).
    #[allow(clippy::cast_precision_loss)]
    fn average_center_weighted(&self, luma: &[f32]) -> f32 {
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let sigma = 0.3 * (self.width.min(self.height)) as f64;
        let two_sigma2 = 2.0 * sigma * sigma;

        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;
        let w = self.width as usize;

        for (idx, &v) in luma.iter().enumerate() {
            let x = (idx % w) as f64;
            let y = (idx / w) as f64;
            let dx = x - cx;
            let dy = y - cy;
            let wt = (-(dx * dx + dy * dy) / two_sigma2).exp();
            weighted_sum += f64::from(v) * wt;
            weight_total += wt;
        }

        if weight_total > 0.0 {
            (weighted_sum / weight_total) as f32
        } else {
            0.0
        }
    }

    /// Spot: centre circle covering 5% of pixels by area.
    #[allow(clippy::cast_precision_loss)]
    fn average_spot(&self, luma: &[f32]) -> f32 {
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        // radius so that π·r² = 0.05 × w × h
        let area = (self.width as f64) * (self.height as f64) * 0.05;
        let radius = (area / std::f64::consts::PI).sqrt();
        let w = self.width as usize;

        let mut sum = 0.0_f64;
        let mut count = 0u64;
        for (idx, &v) in luma.iter().enumerate() {
            let x = (idx % w) as f64;
            let y = (idx / w) as f64;
            let dx = x - cx;
            let dy = y - cy;
            if (dx * dx + dy * dy).sqrt() <= radius {
                sum += f64::from(v);
                count += 1;
            }
        }

        if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.0
        }
    }

    /// Highlight bias: weight = luma².
    #[allow(clippy::cast_precision_loss)]
    fn average_highlight(luma: &[f32], n: usize) -> f32 {
        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;
        for &v in &luma[..n] {
            let w = f64::from(v) * f64::from(v);
            weighted_sum += f64::from(v) * w;
            weight_total += w;
        }
        if weight_total > 0.0 {
            (weighted_sum / weight_total) as f32
        } else {
            0.0
        }
    }

    /// Shadow bias: weight = (1 − luma)².
    #[allow(clippy::cast_precision_loss)]
    fn average_shadow(luma: &[f32], n: usize) -> f32 {
        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;
        for &v in &luma[..n] {
            let inv = 1.0_f64 - f64::from(v);
            let w = inv * inv;
            weighted_sum += f64::from(v) * w;
            weight_total += w;
        }
        if weight_total > 0.0 {
            (weighted_sum / weight_total) as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod exposure_meter_ex_tests {
    use super::*;

    fn flat_luma(width: u32, height: u32, value: f32) -> Vec<f32> {
        vec![value; (width * height) as usize]
    }

    fn gradient_luma(width: u32, height: u32) -> Vec<f32> {
        let n = (width * height) as usize;
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect()
    }

    // ── construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_meter_new() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 1920, 1080);
        assert_eq!(m.width, 1920);
        assert_eq!(m.height, 1080);
        assert_eq!(m.mode, MeteringModeEx::Evaluative);
    }

    // ── EV maths ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ev_18_gray() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 10, 10);
        let luma = flat_luma(10, 10, 0.18);
        let r = m.measure(&luma);
        assert!(
            r.ev.abs() < 0.05,
            "EV at 18% grey should be ≈0, got {}",
            r.ev
        );
    }

    #[test]
    fn test_ev_overexposed() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 10, 10);
        let luma = flat_luma(10, 10, 0.72); // 2 stops over
        let r = m.measure(&luma);
        assert!(
            r.ev > 1.5,
            "EV should be > 1.5 for bright frame, got {}",
            r.ev
        );
    }

    #[test]
    fn test_ev_underexposed() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 10, 10);
        let luma = flat_luma(10, 10, 0.045); // 2 stops under
        let r = m.measure(&luma);
        assert!(
            r.ev < -1.5,
            "EV should be < -1.5 for dark frame, got {}",
            r.ev
        );
    }

    // ── stops_from_correct & suggested_gain_db ────────────────────────────

    #[test]
    fn test_stops_from_correct_equals_ev() {
        let m = ExposureMeter::new(MeteringModeEx::Spot, 20, 20);
        let luma = flat_luma(20, 20, 0.36);
        let r = m.measure(&luma);
        assert!((r.stops_from_correct - r.ev).abs() < 1e-5);
    }

    #[test]
    fn test_suggested_gain_db() {
        let m = ExposureMeter::new(MeteringModeEx::Spot, 10, 10);
        let luma = flat_luma(10, 10, 0.18);
        let r = m.measure(&luma);
        // At 18% grey, EV≈0, so gain_db≈0
        assert!(r.suggested_gain_db.abs() < 0.5);
    }

    // ── clipping / crushed ────────────────────────────────────────────────

    #[test]
    fn test_clipping_pct_all_white() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 10, 10);
        let luma = flat_luma(10, 10, 1.0);
        let r = m.measure(&luma);
        assert!((r.clipping_pct - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_crushed_pct_all_black() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 10, 10);
        let luma = flat_luma(10, 10, 0.0);
        let r = m.measure(&luma);
        assert!((r.crushed_pct - 1.0).abs() < 1e-5);
    }

    // ── luminance min/max ────────────────────────────────────────────────

    #[test]
    fn test_luma_min_max() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 16, 16);
        let luma = gradient_luma(16, 16);
        let r = m.measure(&luma);
        assert!(r.luminance_max > 0.98);
        assert!(r.luminance_min < 0.02);
    }

    // ── mode-specific weighting ──────────────────────────────────────────

    #[test]
    fn test_highlight_mode_biases_bright() {
        let m_hl = ExposureMeter::new(MeteringModeEx::Highlight, 16, 16);
        let m_sh = ExposureMeter::new(MeteringModeEx::Shadow, 16, 16);
        let luma = gradient_luma(16, 16);
        let r_hl = m_hl.measure(&luma);
        let r_sh = m_sh.measure(&luma);
        assert!(
            r_hl.luminance_avg > r_sh.luminance_avg,
            "Highlight avg {} should exceed shadow avg {}",
            r_hl.luminance_avg,
            r_sh.luminance_avg
        );
    }

    #[test]
    fn test_center_weighted_uniform() {
        // Uniform luma → center-weighted avg should still equal that value
        let m = ExposureMeter::new(MeteringModeEx::CenterWeighted, 20, 20);
        let luma = flat_luma(20, 20, 0.5);
        let r = m.measure(&luma);
        assert!((r.luminance_avg - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_spot_subset() {
        // Uniform frame → spot average equals full average
        let m = ExposureMeter::new(MeteringModeEx::Spot, 100, 100);
        let luma = flat_luma(100, 100, 0.4);
        let r = m.measure(&luma);
        assert!((r.luminance_avg - 0.4).abs() < 0.01);
    }

    // ── auto_exposure_gain ───────────────────────────────────────────────

    #[test]
    fn test_auto_exposure_gain_grey() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 10, 10);
        let luma = flat_luma(10, 10, 0.18);
        let gain = m.auto_exposure_gain(&luma);
        assert!((gain - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_auto_exposure_gain_dark() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 10, 10);
        let luma = flat_luma(10, 10, 0.09); // 1 stop under
        let gain = m.auto_exposure_gain(&luma);
        assert!(gain > 1.8 && gain < 2.2, "Expected ~2× gain, got {gain}");
    }

    #[test]
    fn test_measure_empty_input() {
        let m = ExposureMeter::new(MeteringModeEx::Evaluative, 10, 10);
        let r = m.measure(&[]);
        assert_eq!(r.luminance_avg, 0.0);
        assert!(r.ev < -5.0);
    }
}
