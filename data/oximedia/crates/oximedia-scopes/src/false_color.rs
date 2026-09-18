//! False color exposure visualization for video analysis.
//!
//! False color overlays map pixel luminance values to distinct colors, making it
//! easy to identify exposure levels and ensure proper exposure across the image.
//! This is particularly useful for:
//! - Identifying overexposed (clipped) highlights
//! - Finding underexposed (crushed) shadows
//! - Ensuring skin tones are properly exposed
//! - Checking overall exposure distribution
//!
//! The false color scale typically uses IRE units or f-stops relative to middle gray.

use crate::render::{rgb_to_ycbcr, Canvas};
use crate::{ScopeData, ScopeType};
use oximedia_core::OxiResult;

/// False color display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalseColorMode {
    /// IRE-based (0-100 IRE).
    Ire,

    /// Stop-based (relative to middle gray at 18%).
    Stops,

    /// Exposure zones (under, good, over).
    Zones,

    /// Zebra pattern for highlights only.
    Zebra,
}

/// False color scale configuration.
#[derive(Debug, Clone)]
pub struct FalseColorScale {
    /// Exposure zones with colors.
    pub zones: Vec<(f32, f32, [u8; 4])>,

    /// Zebra stripe threshold (IRE).
    pub zebra_threshold: f32,

    /// Zebra stripe width in pixels.
    pub zebra_width: u32,
}

impl Default for FalseColorScale {
    fn default() -> Self {
        Self {
            zones: default_ire_zones(),
            zebra_threshold: 95.0,
            zebra_width: 4,
        }
    }
}

/// Default IRE-based false color zones.
///
/// These zones represent common exposure levels in video production.
#[must_use]
pub fn default_ire_zones() -> Vec<(f32, f32, [u8; 4])> {
    vec![
        // (min_ire, max_ire, color_rgba)
        (0.0, 5.0, [0, 0, 128, 255]),  // Very dark blue (crushed blacks)
        (5.0, 10.0, [0, 0, 255, 255]), // Blue (very underexposed)
        (10.0, 20.0, [0, 128, 255, 255]), // Cyan-blue (underexposed)
        (20.0, 35.0, [0, 255, 255, 255]), // Cyan (shadows)
        (35.0, 45.0, [0, 255, 0, 255]), // Green (good shadow detail)
        (45.0, 55.0, [128, 255, 0, 255]), // Yellow-green (proper exposure)
        (55.0, 65.0, [255, 255, 0, 255]), // Yellow (skin tones, good midtones)
        (65.0, 75.0, [255, 200, 0, 255]), // Orange (bright midtones)
        (75.0, 85.0, [255, 128, 0, 255]), // Orange-red (highlights)
        (85.0, 95.0, [255, 0, 0, 255]), // Red (near clipping)
        (95.0, 100.0, [255, 0, 128, 255]), // Magenta (clipping)
        (100.0, 110.0, [255, 0, 255, 255]), // Bright magenta (clipped)
    ]
}

/// Stop-based false color zones (relative to middle gray).
///
/// Middle gray is typically at 18% reflectance (about 42 IRE).
#[must_use]
pub fn stop_based_zones() -> Vec<(f32, f32, [u8; 4])> {
    vec![
        // (min_stops, max_stops, color_rgba)
        // Stops are relative to middle gray (0.0 = 18% gray)
        (-6.0, -5.0, [0, 0, 64, 255]),    // Very dark (6 stops under)
        (-5.0, -4.0, [0, 0, 128, 255]),   // Dark blue (5 stops under)
        (-4.0, -3.0, [0, 0, 255, 255]),   // Blue (4 stops under)
        (-3.0, -2.0, [0, 128, 255, 255]), // Cyan (3 stops under)
        (-2.0, -1.0, [0, 255, 255, 255]), // Bright cyan (2 stops under)
        (-1.0, 0.0, [0, 255, 0, 255]),    // Green (1 stop under)
        (0.0, 1.0, [255, 255, 0, 255]),   // Yellow (middle gray to 1 stop over)
        (1.0, 2.0, [255, 200, 0, 255]),   // Orange (1-2 stops over)
        (2.0, 3.0, [255, 128, 0, 255]),   // Orange-red (2-3 stops over)
        (3.0, 4.0, [255, 0, 0, 255]),     // Red (3-4 stops over)
        (4.0, 5.0, [255, 0, 128, 255]),   // Magenta (4-5 stops over)
        (5.0, 10.0, [255, 0, 255, 255]),  // Bright magenta (clipped)
    ]
}

/// Generates a false color overlay from RGB frame data.
///
/// The false color overlay maps luminance values to distinct colors
/// based on the selected mode and scale.
///
/// # Arguments
///
/// * `frame` - RGB24 frame data (width * height * 3 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `mode` - False color display mode
/// * `scale` - False color scale configuration
///
/// # Errors
///
/// Returns an error if frame data is invalid or insufficient.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub fn generate_false_color(
    frame: &[u8],
    width: u32,
    height: u32,
    mode: FalseColorMode,
    scale: &FalseColorScale,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let mut canvas = Canvas::new(width, height);

    match mode {
        FalseColorMode::Ire => {
            generate_ire_false_color(frame, width, height, &mut canvas, scale);
        }
        FalseColorMode::Stops => {
            generate_stop_false_color(frame, width, height, &mut canvas, scale);
        }
        FalseColorMode::Zones => {
            generate_zone_false_color(frame, width, height, &mut canvas);
        }
        FalseColorMode::Zebra => {
            generate_zebra_pattern(frame, width, height, &mut canvas, scale);
        }
    }

    Ok(ScopeData {
        width,
        height,
        data: canvas.data,
        scope_type: ScopeType::FalseColor,
    })
}

/// Generates IRE-based false color overlay.
fn generate_ire_false_color(
    frame: &[u8],
    width: u32,
    height: u32,
    canvas: &mut Canvas,
    scale: &FalseColorScale,
) {
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, _, _) = rgb_to_ycbcr(r, g, b);

            // Convert luma (0-255) to IRE (0-100)
            let ire = (f32::from(luma) / 255.0) * 100.0;

            // Find the appropriate zone color
            let color = find_zone_color(ire, &scale.zones);
            canvas.set_pixel(x, y, color);
        }
    }
}

/// Generates stop-based false color overlay.
fn generate_stop_false_color(
    frame: &[u8],
    width: u32,
    height: u32,
    canvas: &mut Canvas,
    scale: &FalseColorScale,
) {
    // Middle gray reference: 18% reflectance ≈ 42 IRE ≈ 108/255 in linear space
    const MIDDLE_GRAY: f32 = 0.18;

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, _, _) = rgb_to_ycbcr(r, g, b);

            // Convert to linear space (assuming sRGB)
            let linear_luma = srgb_to_linear(f32::from(luma) / 255.0);

            // Calculate stops relative to middle gray
            let stops = if linear_luma > 0.0 {
                (linear_luma / MIDDLE_GRAY).log2()
            } else {
                -10.0 // Very dark
            };

            // Find the appropriate zone color
            let color = find_zone_color(stops, &scale.zones);
            canvas.set_pixel(x, y, color);
        }
    }
}

/// Generates simple 3-zone false color (under/good/over).
fn generate_zone_false_color(frame: &[u8], width: u32, height: u32, canvas: &mut Canvas) {
    let zones = vec![
        (0.0, 20.0, [0, 0, 255, 255]),   // Underexposed (blue)
        (20.0, 80.0, [0, 255, 0, 255]),  // Good exposure (green)
        (80.0, 110.0, [255, 0, 0, 255]), // Overexposed (red)
    ];

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, _, _) = rgb_to_ycbcr(r, g, b);
            let ire = (f32::from(luma) / 255.0) * 100.0;

            let color = find_zone_color(ire, &zones);
            canvas.set_pixel(x, y, color);
        }
    }
}

/// Generates zebra stripe pattern for highlight detection.
///
/// Zebra stripes are diagonal lines overlaid on areas that exceed the threshold.
fn generate_zebra_pattern(
    frame: &[u8],
    width: u32,
    height: u32,
    canvas: &mut Canvas,
    scale: &FalseColorScale,
) {
    // Copy original frame to canvas first
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            canvas.set_pixel(x, y, [r, g, b, 255]);
        }
    }

    // Overlay zebra stripes on highlights
    let zebra_width = scale.zebra_width;
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, _, _) = rgb_to_ycbcr(r, g, b);
            let ire = (f32::from(luma) / 255.0) * 100.0;

            // If above threshold, draw zebra stripe
            if ire >= scale.zebra_threshold {
                // Diagonal stripe pattern
                #[allow(clippy::cast_possible_truncation)]
                let stripe_pos = ((x + y) / zebra_width) % 2;
                if stripe_pos == 0 {
                    // Draw red stripe
                    canvas.set_pixel(x, y, [255, 0, 0, 255]);
                }
            }
        }
    }
}

/// Finds the zone color for a given value.
fn find_zone_color(value: f32, zones: &[(f32, f32, [u8; 4])]) -> [u8; 4] {
    for (min, max, color) in zones {
        if value >= *min && value < *max {
            return *color;
        }
    }

    // Default to black if not in any zone
    [0, 0, 0, 255]
}

/// Converts sRGB to linear space.
#[must_use]
fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts linear to sRGB space.
#[must_use]
#[allow(dead_code)]
fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// False color statistics.
#[derive(Debug, Clone)]
pub struct FalseColorStats {
    /// Percentage of pixels in each zone.
    pub zone_distribution: Vec<(String, f32)>,

    /// Percentage of clipped highlights (> 95 IRE).
    pub highlight_clip_percent: f32,

    /// Percentage of crushed shadows (< 5 IRE).
    pub shadow_clip_percent: f32,

    /// Percentage of properly exposed pixels (20-80 IRE).
    pub good_exposure_percent: f32,
}

/// Computes false color statistics from frame data.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_false_color_stats(frame: &[u8], width: u32, height: u32) -> FalseColorStats {
    let pixel_count = width * height;
    let mut highlight_clip_count = 0u32;
    let mut shadow_clip_count = 0u32;
    let mut good_exposure_count = 0u32;

    // Zone counters
    let zone_names = [
        "Crushed Blacks (<5 IRE)",
        "Underexposed (5-20 IRE)",
        "Shadows (20-35 IRE)",
        "Good (35-65 IRE)",
        "Highlights (65-85 IRE)",
        "Near Clipping (85-95 IRE)",
        "Clipped (>95 IRE)",
    ];
    let mut zone_counts = vec![0u32; zone_names.len()];

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                break;
            }

            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, _, _) = rgb_to_ycbcr(r, g, b);
            let ire = (f32::from(luma) / 255.0) * 100.0;

            // Count clipping
            if ire < 5.0 {
                shadow_clip_count += 1;
            }
            if ire > 95.0 {
                highlight_clip_count += 1;
            }

            // Count good exposure
            if (20.0..=80.0).contains(&ire) {
                good_exposure_count += 1;
            }

            // Count zones
            match ire {
                x if x < 5.0 => zone_counts[0] += 1,
                x if x < 20.0 => zone_counts[1] += 1,
                x if x < 35.0 => zone_counts[2] += 1,
                x if x < 65.0 => zone_counts[3] += 1,
                x if x < 85.0 => zone_counts[4] += 1,
                x if x < 95.0 => zone_counts[5] += 1,
                _ => zone_counts[6] += 1,
            }
        }
    }

    let zone_distribution: Vec<(String, f32)> = zone_names
        .iter()
        .zip(zone_counts.iter())
        .map(|(name, &count)| {
            (
                (*name).to_string(),
                (count as f32 / pixel_count as f32) * 100.0,
            )
        })
        .collect();

    FalseColorStats {
        zone_distribution,
        highlight_clip_percent: (highlight_clip_count as f32 / pixel_count as f32) * 100.0,
        shadow_clip_percent: (shadow_clip_count as f32 / pixel_count as f32) * 100.0,
        good_exposure_percent: (good_exposure_count as f32 / pixel_count as f32) * 100.0,
    }
}

/// Generates a false color legend image.
///
/// The legend shows the color scale used for false color mapping.
#[must_use]
pub fn generate_false_color_legend(width: u32, height: u32, scale: &FalseColorScale) -> Vec<u8> {
    let mut canvas = Canvas::new(width, height);

    // Draw gradient from left (min) to right (max)
    for x in 0..width {
        let value = (x as f32 / width as f32) * 100.0; // 0-100 IRE

        let color = find_zone_color(value, &scale.zones);

        // Fill vertical stripe
        for y in 0..height {
            canvas.set_pixel(x, y, color);
        }
    }

    canvas.data
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create gradient from black to white
        for y in 0..height {
            let value = ((y * 255) / height) as u8;
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        frame
    }

    #[test]
    fn test_generate_false_color_ire() {
        let frame = create_test_frame(100, 100);
        let scale = FalseColorScale::default();

        let result = generate_false_color(&frame, 100, 100, FalseColorMode::Ire, &scale);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, 100);
        assert_eq!(scope.height, 100);
    }

    #[test]
    fn test_generate_false_color_stops() {
        let frame = create_test_frame(100, 100);
        let scale = FalseColorScale {
            zones: stop_based_zones(),
            ..Default::default()
        };

        let result = generate_false_color(&frame, 100, 100, FalseColorMode::Stops, &scale);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_false_color_zones() {
        let frame = create_test_frame(100, 100);
        let scale = FalseColorScale::default();

        let result = generate_false_color(&frame, 100, 100, FalseColorMode::Zones, &scale);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_zebra_pattern() {
        let mut frame = vec![0u8; 100 * 100 * 3];

        // Create overexposed area
        for i in (0..frame.len()).step_by(3) {
            frame[i] = 250;
            frame[i + 1] = 250;
            frame[i + 2] = 250;
        }

        let scale = FalseColorScale::default();
        let result = generate_false_color(&frame, 100, 100, FalseColorMode::Zebra, &scale);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_false_color_stats() {
        let frame = create_test_frame(100, 100);
        let stats = compute_false_color_stats(&frame, 100, 100);

        assert_eq!(stats.zone_distribution.len(), 7);
        assert!(stats.good_exposure_percent > 0.0);
        assert!(stats.good_exposure_percent <= 100.0);
    }

    #[test]
    fn test_srgb_linear_conversion() {
        let srgb = 0.5;
        let linear = srgb_to_linear(srgb);
        let srgb_back = linear_to_srgb(linear);

        assert!((srgb - srgb_back).abs() < 0.001);
    }

    #[test]
    fn test_find_zone_color() {
        let zones = default_ire_zones();

        // Test black (0-5 IRE)
        let color = find_zone_color(2.0, &zones);
        assert_eq!(color[2], 128); // Dark blue

        // Test white (95-100 IRE)
        let color = find_zone_color(97.0, &zones);
        assert_eq!(color[0], 255); // Magenta
    }

    #[test]
    fn test_generate_false_color_legend() {
        let scale = FalseColorScale::default();
        let legend = generate_false_color_legend(256, 20, &scale);

        assert_eq!(legend.len(), (256 * 20 * 4) as usize);
    }

    #[test]
    fn test_invalid_frame_size() {
        let frame = vec![0u8; 100]; // Too small
        let scale = FalseColorScale::default();

        let result = generate_false_color(&frame, 100, 100, FalseColorMode::Ire, &scale);
        assert!(result.is_err());
    }

    #[test]
    fn test_clipping_detection() {
        let mut frame = vec![0u8; 100 * 100 * 3];

        // Half black, half white
        for i in 0..(50 * 100 * 3) {
            frame[i] = 0; // Black
        }
        for i in (50 * 100 * 3)..frame.len() {
            frame[i] = 255; // White
        }

        let stats = compute_false_color_stats(&frame, 100, 100);

        assert!(stats.shadow_clip_percent > 40.0);
        assert!(stats.highlight_clip_percent > 40.0);
    }
}

// ============================================================================
// Enhanced False Color: FalseColorConfig / FalseColorZone / FalseColorProcessor
// ============================================================================

/// Configuration for the enhanced false color processor.
#[derive(Debug, Clone)]
pub struct FalseColorConfig {
    /// Luminance above this threshold is considered clipping (default 0.95).
    pub overexposed_threshold: f32,
    /// Luminance above this threshold is considered highlight zone (default 0.85).
    pub highlight_threshold: f32,
    /// Upper boundary of the midtone-high zone (default 0.55).
    pub midtone_high: f32,
    /// Lower boundary of the midtone-low zone (default 0.35).
    pub midtone_low: f32,
    /// Luminance below this threshold is considered shadow zone (default 0.15).
    pub shadow_threshold: f32,
    /// Luminance below this threshold is considered crushed blacks (default 0.02).
    pub underexposed_threshold: f32,
}

impl Default for FalseColorConfig {
    fn default() -> Self {
        Self {
            overexposed_threshold: 0.95,
            highlight_threshold: 0.85,
            midtone_high: 0.55,
            midtone_low: 0.35,
            shadow_threshold: 0.15,
            underexposed_threshold: 0.02,
        }
    }
}

impl FalseColorConfig {
    /// ARRI LogC-derived cinema standard thresholds.
    ///
    /// These are calibrated to the ARRI LogC3 encoding where 18% gray
    /// maps to approximately 0.391 scene-linear exposure.
    #[must_use]
    pub fn cinema_standard() -> Self {
        Self {
            overexposed_threshold: 0.97,
            highlight_threshold: 0.88,
            midtone_high: 0.60,
            midtone_low: 0.38,
            shadow_threshold: 0.12,
            underexposed_threshold: 0.01,
        }
    }

    /// Rec.709 broadcast standard thresholds.
    ///
    /// Thresholds aligned with legal broadcast range (16–235 on 8-bit),
    /// where 109/255 ≈ 0.427 maps to 18% gray.
    #[must_use]
    pub fn broadcast_standard() -> Self {
        Self {
            overexposed_threshold: 0.922, // 235/255
            highlight_threshold: 0.800,
            midtone_high: 0.540,
            midtone_low: 0.330,
            shadow_threshold: 0.125,       // 32/255
            underexposed_threshold: 0.063, // 16/255
        }
    }
}

/// Exposure zone classification for false color display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalseColorZone {
    /// Overexposed / clipped highlights — red.
    Clipping,
    /// Near-clipping highlights — yellow.
    Highlights,
    /// Upper midtones, good exposure — green.
    MidtonesHigh,
    /// Mid midtones, good exposure — grey.
    Midtones,
    /// Lower midtones, good exposure — pink.
    MidtonesLow,
    /// Shadow detail still present — blue.
    Shadows,
    /// Crushed blacks, no detail — purple.
    Crushed,
}

/// Processor that applies false color mapping to video frames.
pub struct FalseColorProcessor {
    config: FalseColorConfig,
}

impl FalseColorProcessor {
    /// Creates a new processor with the given configuration.
    #[must_use]
    pub fn new(config: FalseColorConfig) -> Self {
        Self { config }
    }

    /// Classifies a normalised luma value (0.0–1.0) into a [`FalseColorZone`].
    #[must_use]
    pub fn classify_luminance(&self, luma: f32) -> FalseColorZone {
        let c = &self.config;
        if luma >= c.overexposed_threshold {
            FalseColorZone::Clipping
        } else if luma >= c.highlight_threshold {
            FalseColorZone::Highlights
        } else if luma >= c.midtone_high {
            FalseColorZone::MidtonesHigh
        } else if luma >= c.midtone_low {
            FalseColorZone::Midtones
        } else if luma >= c.shadow_threshold {
            FalseColorZone::MidtonesLow
        } else if luma >= c.underexposed_threshold {
            FalseColorZone::Shadows
        } else {
            FalseColorZone::Crushed
        }
    }

    /// Returns the RGB display colour for a given zone.
    ///
    /// Colour conventions follow common broadcast false-colour standards:
    /// - Clipping  → red   (255,   0,   0)
    /// - Highlights→ yellow(255, 255,   0)
    /// - MidHigh   → green (  0, 200,   0)
    /// - Midtones  → grey  (128, 128, 128)
    /// - MidLow    → pink  (255, 105, 180)
    /// - Shadows   → blue  (  0,   0, 255)
    /// - Crushed   → purple(128,   0, 128)
    #[must_use]
    pub fn zone_color(zone: &FalseColorZone) -> (u8, u8, u8) {
        match zone {
            FalseColorZone::Clipping => (255, 0, 0),
            FalseColorZone::Highlights => (255, 255, 0),
            FalseColorZone::MidtonesHigh => (0, 200, 0),
            FalseColorZone::Midtones => (128, 128, 128),
            FalseColorZone::MidtonesLow => (255, 105, 180),
            FalseColorZone::Shadows => (0, 0, 255),
            FalseColorZone::Crushed => (128, 0, 128),
        }
    }

    /// Processes a YUV420p frame and returns a flat RGB false-color image.
    ///
    /// The input slice must be at least `width * height * 3 / 2` bytes for
    /// YUV420p.  Only the Y (luma) plane is used for classification.
    ///
    /// Returns an interleaved RGB byte vector of length `width * height * 3`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn process_frame(&self, yuv: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n_pixels = (width * height) as usize;
        // Accept both YUV420p (1.5× pixels) and pure-Y planes (1× pixels).
        if yuv.len() < n_pixels {
            return vec![0u8; n_pixels * 3];
        }

        let mut out = Vec::with_capacity(n_pixels * 3);
        for idx in 0..n_pixels {
            let y_raw = yuv[idx];
            let luma = y_raw as f32 / 255.0;
            let zone = self.classify_luminance(luma);
            let (r, g, b) = Self::zone_color(&zone);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }

    /// Computes the fraction of pixels in each [`FalseColorZone`].
    ///
    /// Returns a `Vec` of `(zone, fraction)` pairs ordered by zone enum
    /// discriminant (Clipping first, Crushed last).  Fractions sum to 1.0
    /// (or 0.0 on an empty/invalid frame).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn zone_coverage(&self, yuv: &[u8], width: u32, height: u32) -> Vec<(FalseColorZone, f32)> {
        let n_pixels = (width * height) as usize;
        if yuv.len() < n_pixels || n_pixels == 0 {
            return vec![
                (FalseColorZone::Clipping, 0.0),
                (FalseColorZone::Highlights, 0.0),
                (FalseColorZone::MidtonesHigh, 0.0),
                (FalseColorZone::Midtones, 0.0),
                (FalseColorZone::MidtonesLow, 0.0),
                (FalseColorZone::Shadows, 0.0),
                (FalseColorZone::Crushed, 0.0),
            ];
        }

        let mut counts = [0u64; 7];

        for idx in 0..n_pixels {
            let luma = yuv[idx] as f32 / 255.0;
            let zone = self.classify_luminance(luma);
            let bucket = match zone {
                FalseColorZone::Clipping => 0,
                FalseColorZone::Highlights => 1,
                FalseColorZone::MidtonesHigh => 2,
                FalseColorZone::Midtones => 3,
                FalseColorZone::MidtonesLow => 4,
                FalseColorZone::Shadows => 5,
                FalseColorZone::Crushed => 6,
            };
            counts[bucket] += 1;
        }

        let total = n_pixels as f32;
        vec![
            (FalseColorZone::Clipping, counts[0] as f32 / total),
            (FalseColorZone::Highlights, counts[1] as f32 / total),
            (FalseColorZone::MidtonesHigh, counts[2] as f32 / total),
            (FalseColorZone::Midtones, counts[3] as f32 / total),
            (FalseColorZone::MidtonesLow, counts[4] as f32 / total),
            (FalseColorZone::Shadows, counts[5] as f32 / total),
            (FalseColorZone::Crushed, counts[6] as f32 / total),
        ]
    }
}

#[cfg(test)]
mod false_color_processor_tests {
    use super::*;

    fn default_proc() -> FalseColorProcessor {
        FalseColorProcessor::new(FalseColorConfig::default())
    }

    // ── FalseColorConfig ────────────────────────────────────────────────────

    #[test]
    fn test_config_default_thresholds() {
        let cfg = FalseColorConfig::default();
        assert!((cfg.overexposed_threshold - 0.95).abs() < 1e-6);
        assert!((cfg.underexposed_threshold - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_config_cinema_standard() {
        let cfg = FalseColorConfig::cinema_standard();
        // Cinema standard should be slightly more permissive at the top
        assert!(cfg.overexposed_threshold > 0.95);
    }

    #[test]
    fn test_config_broadcast_standard() {
        let cfg = FalseColorConfig::broadcast_standard();
        // Broadcast clips at legal level 235/255 ≈ 0.922
        assert!((cfg.overexposed_threshold - (235.0_f32 / 255.0)).abs() < 0.01);
    }

    // ── classify_luminance ──────────────────────────────────────────────────

    #[test]
    fn test_classify_clipping() {
        let proc = default_proc();
        assert_eq!(proc.classify_luminance(1.0), FalseColorZone::Clipping);
        assert_eq!(proc.classify_luminance(0.96), FalseColorZone::Clipping);
    }

    #[test]
    fn test_classify_highlights() {
        let proc = default_proc();
        assert_eq!(proc.classify_luminance(0.90), FalseColorZone::Highlights);
    }

    #[test]
    fn test_classify_midtones_high() {
        let proc = default_proc();
        assert_eq!(proc.classify_luminance(0.60), FalseColorZone::MidtonesHigh);
    }

    #[test]
    fn test_classify_midtones() {
        let proc = default_proc();
        assert_eq!(proc.classify_luminance(0.45), FalseColorZone::Midtones);
    }

    #[test]
    fn test_classify_midtones_low() {
        let proc = default_proc();
        assert_eq!(proc.classify_luminance(0.20), FalseColorZone::MidtonesLow);
    }

    #[test]
    fn test_classify_shadows() {
        let proc = default_proc();
        assert_eq!(proc.classify_luminance(0.08), FalseColorZone::Shadows);
    }

    #[test]
    fn test_classify_crushed() {
        let proc = default_proc();
        assert_eq!(proc.classify_luminance(0.00), FalseColorZone::Crushed);
        assert_eq!(proc.classify_luminance(0.01), FalseColorZone::Crushed);
    }

    // ── zone_color ──────────────────────────────────────────────────────────

    #[test]
    fn test_zone_color_clipping_is_red() {
        let (r, g, b) = FalseColorProcessor::zone_color(&FalseColorZone::Clipping);
        assert_eq!((r, g, b), (255, 0, 0));
    }

    #[test]
    fn test_zone_color_crushed_is_purple() {
        let (r, g, b) = FalseColorProcessor::zone_color(&FalseColorZone::Crushed);
        // Purple: r and b non-zero, g=0
        assert!(r > 0);
        assert_eq!(g, 0);
        assert!(b > 0);
    }

    // ── process_frame ───────────────────────────────────────────────────────

    #[test]
    fn test_process_frame_output_size() {
        let proc = default_proc();
        // 4×4 pure luma plane (Y only)
        let yuv = vec![128u8; 16];
        let out = proc.process_frame(&yuv, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_process_frame_all_clipping() {
        let proc = default_proc();
        let yuv = vec![255u8; 100]; // 10×10 white
        let out = proc.process_frame(&yuv, 10, 10);
        // Every pixel should be the Clipping colour (red)
        let (cr, cg, cb) = FalseColorProcessor::zone_color(&FalseColorZone::Clipping);
        assert_eq!(out[0], cr);
        assert_eq!(out[1], cg);
        assert_eq!(out[2], cb);
    }

    #[test]
    fn test_process_frame_empty_input() {
        let proc = default_proc();
        let out = proc.process_frame(&[], 4, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
        // Should be all zero (fallback)
        assert!(out.iter().all(|&b| b == 0));
    }

    // ── zone_coverage ───────────────────────────────────────────────────────

    #[test]
    fn test_zone_coverage_sum_to_one() {
        let proc = default_proc();
        // Gradient from 0..=255
        let yuv: Vec<u8> = (0u8..=255).collect();
        let coverage = proc.zone_coverage(&yuv, 16, 16);
        let sum: f32 = coverage.iter().map(|(_, f)| f).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_zone_coverage_all_clipping() {
        let proc = default_proc();
        let yuv = vec![255u8; 256];
        let coverage = proc.zone_coverage(&yuv, 16, 16);
        let clipping = coverage
            .iter()
            .find(|(z, _)| *z == FalseColorZone::Clipping);
        let (_, frac) = clipping.expect("clipping zone present");
        assert!((frac - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_zone_coverage_empty() {
        let proc = default_proc();
        let coverage = proc.zone_coverage(&[], 4, 4);
        let sum: f32 = coverage.iter().map(|(_, f)| f).sum();
        assert!(sum.abs() < 1e-5);
    }
}
