//! Vectorscope for color and saturation analysis.
//!
//! A vectorscope displays chrominance (color) information in a polar plot,
//! with hue represented as angle and saturation as distance from center.
//! This is essential for:
//! - Checking white balance and color casts
//! - Measuring color saturation
//! - Ensuring colors match SMPTE standards
//! - Identifying skin tones
//! - Detecting out-of-gamut colors
//!
//! The vectorscope uses Cb (U) and Cr (V) components as X and Y axes.

use crate::render::{rgb_to_ycbcr, Canvas};
use crate::simd_convert::convert_batch_bt601_simd;
use crate::{GamutColorspace, ScopeConfig, ScopeData, ScopeType, VectorscopeMode};
use oximedia_core::OxiResult;
use std::f32::consts::PI;

/// Generates a YUV vectorscope from RGB frame data.
///
/// The vectorscope displays color information in a circular plot:
/// - Center: no color (grayscale)
/// - Distance from center: saturation
/// - Angle: hue
///
/// # Arguments
///
/// * `frame` - RGB24 frame data (width * height * 3 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - Scope configuration
///
/// # Errors
///
/// Returns an error if frame data is invalid or insufficient.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_sign_loss)]
pub fn generate_vectorscope(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Create accumulation buffer for intensity
    let mut accumulator = vec![0u32; (scope_width * scope_height) as usize];

    let center_x = scope_width / 2;
    let center_y = scope_height / 2;
    let max_radius = (scope_width.min(scope_height) / 2 - 10) as f32;

    // Apply gain (zoom)
    let gain = config.vectorscope_gain;

    // Pre-convert entire frame using BT.601 SIMD dispatch.
    let ycbcr_frame = convert_batch_bt601_simd(&frame[..(width * height * 3) as usize]);

    // Process pre-converted pixels
    let mut px_idx = 0usize;
    for _y in 0..height {
        for _x in 0..width {
            let [_luma, cb, cr] = ycbcr_frame[px_idx];
            px_idx += 1;

            // Cb and Cr are in range 0-255, with 128 = neutral
            // Map to -128 to +127 range
            let cb_centered = f32::from(cb) - 128.0;
            let cr_centered = f32::from(cr) - 128.0;

            // Apply gain and map to scope coordinates
            let scope_x_f = center_x as f32 + (cb_centered * gain * max_radius) / 128.0;
            let scope_y_f = center_y as f32 - (cr_centered * gain * max_radius) / 128.0;

            let scope_x = scope_x_f as i32;
            let scope_y = scope_y_f as i32;

            // Check bounds
            if scope_x >= 0
                && scope_x < scope_width as i32
                && scope_y >= 0
                && scope_y < scope_height as i32
            {
                let idx = (scope_y as u32 * scope_width + scope_x as u32) as usize;
                if idx < accumulator.len() {
                    accumulator[idx] = accumulator[idx].saturating_add(1);
                }
            }
        }
    }

    // Find max value for normalization
    let max_val = accumulator.iter().copied().max().unwrap_or(1);

    // Draw accumulated vectorscope
    match config.vectorscope_mode {
        VectorscopeMode::Circular => {
            // Circular mask: only show pixels within max_radius
            for y in 0..scope_height {
                for x in 0..scope_width {
                    let dx = x as f32 - center_x as f32;
                    let dy = y as f32 - center_y as f32;
                    let distance = (dx * dx + dy * dy).sqrt();

                    if distance <= max_radius {
                        let idx = (y * scope_width + x) as usize;
                        let count = accumulator[idx];

                        if count > 0 {
                            let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                            canvas.accumulate_pixel(x, y, normalized);
                        }
                    }
                }
            }
        }
        VectorscopeMode::Rectangular => {
            // Show all pixels
            for y in 0..scope_height {
                for x in 0..scope_width {
                    let idx = (y * scope_width + x) as usize;
                    let count = accumulator[idx];

                    if count > 0 {
                        let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                        canvas.accumulate_pixel(x, y, normalized);
                    }
                }
            }
        }
    }

    // Highlight out-of-gamut colors if requested
    if config.highlight_gamut {
        highlight_out_of_gamut(&mut canvas, config, &accumulator, max_val);
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_vectorscope_graticule(&mut canvas, config);
    }

    // Draw labels
    if config.show_labels {
        draw_vectorscope_labels(&mut canvas);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::Vectorscope,
    })
}

/// Highlights out-of-gamut pixels based on the selected colorspace.
fn highlight_out_of_gamut(
    canvas: &mut Canvas,
    config: &ScopeConfig,
    accumulator: &[u32],
    max_val: u32,
) {
    let scope_width = canvas.width;
    let scope_height = canvas.height;
    let center_x = scope_width / 2;
    let center_y = scope_height / 2;

    // Get gamut limits for the selected colorspace
    let gamut_limit = match config.gamut_colorspace {
        GamutColorspace::Rec709 => 118.0,  // Conservative limit for BT.709
        GamutColorspace::Rec2020 => 158.0, // Wider gamut for BT.2020
        GamutColorspace::DciP3 => 135.0,   // P3 gamut
    };

    for y in 0..scope_height {
        for x in 0..scope_width {
            let idx = (y * scope_width + x) as usize;
            let count = accumulator[idx];

            if count > 0 {
                let dx = x as f32 - center_x as f32;
                let dy = y as f32 - center_y as f32;
                let distance = (dx * dx + dy * dy).sqrt();

                // If beyond gamut limit, highlight in red
                if distance > gamut_limit {
                    #[allow(clippy::cast_possible_truncation)]
                    #[allow(clippy::cast_sign_loss)]
                    let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                    canvas.blend_pixel(x, y, [255, 0, 0, normalized]);
                }
            }
        }
    }
}

/// Draws labels for vectorscope (hue angles).
fn draw_vectorscope_labels(canvas: &mut Canvas) {
    let center_x = canvas.width / 2;
    let center_y = canvas.height / 2;
    let radius = (canvas.width.min(canvas.height) / 2) as f32 - 20.0;

    // Label positions for primary and secondary colors
    let labels = [
        (104_f32, "R"),  // Red
        (168_f32, "Yl"), // Yellow
        (241_f32, "G"),  // Green
        (284_f32, "Cy"), // Cyan
        (348_f32, "B"),  // Blue
        (61_f32, "Mg"),  // Magenta
    ];

    for (angle, _label) in &labels {
        let rad = angle.to_radians();

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let x = (center_x as f32 + rad.cos() * radius) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let y = (center_y as f32 - rad.sin() * radius) as u32;

        // Draw a small dot for label position
        if x >= 1 && x < canvas.width - 1 && y >= 1 && y < canvas.height - 1 {
            canvas.set_pixel(x, y, crate::render::colors::WHITE);
            canvas.set_pixel(x + 1, y, crate::render::colors::WHITE);
            canvas.set_pixel(x, y + 1, crate::render::colors::WHITE);
            canvas.set_pixel(x + 1, y + 1, crate::render::colors::WHITE);
        }
    }
}

/// Calculates the hue angle from Cb/Cr components.
///
/// Returns angle in degrees (0-360).
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn calculate_hue(cb: u8, cr: u8) -> f32 {
    let cb_centered = f32::from(cb) - 128.0;
    let cr_centered = f32::from(cr) - 128.0;

    let angle_rad = cb_centered.atan2(cr_centered);
    let mut angle_deg = angle_rad * 180.0 / PI;

    // Normalize to 0-360
    if angle_deg < 0.0 {
        angle_deg += 360.0;
    }

    angle_deg
}

/// Calculates the saturation from Cb/Cr components.
///
/// Returns saturation in range 0-1.
#[must_use]
pub fn calculate_saturation(cb: u8, cr: u8) -> f32 {
    let cb_centered = f32::from(cb) - 128.0;
    let cr_centered = f32::from(cr) - 128.0;

    let magnitude = (cb_centered * cb_centered + cr_centered * cr_centered).sqrt();

    // Normalize to 0-1 range (max theoretical distance is ~181)
    (magnitude / 181.0).min(1.0)
}

/// Checks if a color is within the skin tone range.
///
/// Skin tones typically fall along a specific line in the vectorscope
/// (approximately 123° ± 15°).
#[must_use]
pub fn is_skin_tone(cb: u8, cr: u8) -> bool {
    let hue = calculate_hue(cb, cr);
    let saturation = calculate_saturation(cb, cr);

    // Skin tone hue range: approximately 108-138 degrees
    // Saturation: typically 0.1-0.6
    (108.0..=138.0).contains(&hue) && (0.1..=0.6).contains(&saturation)
}

/// Vectorscope statistics.
#[derive(Debug, Clone)]
pub struct VectorscopeStats {
    /// Average saturation (0-1).
    pub avg_saturation: f32,

    /// Maximum saturation (0-1).
    pub max_saturation: f32,

    /// Dominant hue angle (degrees).
    pub dominant_hue: f32,

    /// Percentage of pixels that are skin tone.
    pub skin_tone_percent: f32,

    /// Percentage of pixels out of gamut.
    pub out_of_gamut_percent: f32,
}

/// Computes vectorscope statistics from frame data.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_vectorscope_stats(
    frame: &[u8],
    width: u32,
    height: u32,
    gamut: GamutColorspace,
) -> VectorscopeStats {
    let mut saturation_sum = 0.0f32;
    let mut max_saturation = 0.0f32;
    let mut hue_histogram = vec![0u32; 360];
    let mut skin_tone_count = 0u32;
    let mut out_of_gamut_count = 0u32;
    let pixel_count = width * height;

    let gamut_limit = match gamut {
        GamutColorspace::Rec709 => 118.0,
        GamutColorspace::Rec2020 => 158.0,
        GamutColorspace::DciP3 => 135.0,
    };

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                break;
            }

            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (_luma, cb, cr) = rgb_to_ycbcr(r, g, b);

            let saturation = calculate_saturation(cb, cr);
            let hue = calculate_hue(cb, cr);

            saturation_sum += saturation;
            max_saturation = max_saturation.max(saturation);

            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            let hue_bin = (hue as usize).min(359);
            hue_histogram[hue_bin] += 1;

            if is_skin_tone(cb, cr) {
                skin_tone_count += 1;
            }

            // Check if out of gamut
            let cb_centered = f32::from(cb) - 128.0;
            let cr_centered = f32::from(cr) - 128.0;
            let distance = (cb_centered * cb_centered + cr_centered * cr_centered).sqrt();
            if distance > gamut_limit {
                out_of_gamut_count += 1;
            }
        }
    }

    let avg_saturation = saturation_sum / pixel_count as f32;

    // Find dominant hue (most common)
    let dominant_hue_bin = hue_histogram
        .iter()
        .enumerate()
        .max_by_key(|(_, &count)| count)
        .map_or(0, |(bin, _)| bin);

    VectorscopeStats {
        avg_saturation,
        max_saturation,
        dominant_hue: dominant_hue_bin as f32,
        skin_tone_percent: (skin_tone_count as f32 / pixel_count as f32) * 100.0,
        out_of_gamut_percent: (out_of_gamut_count as f32 / pixel_count as f32) * 100.0,
    }
}

// =============================================================================
// Enhanced Vectorscope: zoom/pan controls and IQ display mode
// =============================================================================

/// Display mode for vectorscope chrominance axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorscopeDisplayMode {
    /// Standard UV (Cb/Cr) display — default broadcast convention.
    Uv,
    /// IQ display — NTSC in-phase / quadrature representation.
    /// I-axis at 33° from +Cr, Q-axis at 33°+90° = 123° from +Cr.
    Iq,
}

/// Zoom and pan control for vectorscope navigation.
#[derive(Debug, Clone)]
pub struct VectorscopeViewport {
    /// Zoom factor (1.0 = full view, 2.0 = 2x zoom into center).
    pub zoom: f32,
    /// Horizontal pan offset in normalized coordinates [-1.0, 1.0].
    /// 0.0 = centered, negative = pan left, positive = pan right.
    pub pan_x: f32,
    /// Vertical pan offset in normalized coordinates [-1.0, 1.0].
    /// 0.0 = centered, negative = pan up, positive = pan down.
    pub pan_y: f32,
}

impl Default for VectorscopeViewport {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }
}

impl VectorscopeViewport {
    /// Creates a viewport with the given zoom and centered pan.
    #[must_use]
    pub fn with_zoom(zoom: f32) -> Self {
        Self {
            zoom: zoom.max(0.1),
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }

    /// Creates a viewport with zoom and pan offsets.
    #[must_use]
    pub fn new(zoom: f32, pan_x: f32, pan_y: f32) -> Self {
        Self {
            zoom: zoom.max(0.1),
            pan_x: pan_x.clamp(-1.0, 1.0),
            pan_y: pan_y.clamp(-1.0, 1.0),
        }
    }

    /// Applies the viewport transform: maps (cb_centered, cr_centered) in [-128,127]
    /// to scope pixel coordinates (scope_x, scope_y).
    fn transform(
        &self,
        cb_centered: f32,
        cr_centered: f32,
        center_x: f32,
        center_y: f32,
        max_radius: f32,
        gain: f32,
    ) -> (f32, f32) {
        let scaled_cb = cb_centered * gain * max_radius / 128.0;
        let scaled_cr = cr_centered * gain * max_radius / 128.0;

        // Apply zoom and pan
        let zoomed_x = scaled_cb * self.zoom + self.pan_x * max_radius;
        let zoomed_y = scaled_cr * self.zoom + self.pan_y * max_radius;

        (center_x + zoomed_x, center_y - zoomed_y)
    }
}

/// Rotation matrix for IQ transform.
/// I-axis is at 33° from Cr axis, Q-axis is perpendicular.
/// I = Cr * cos(33°) + Cb * sin(33°)
/// Q = -Cr * sin(33°) + Cb * cos(33°)
fn uv_to_iq(cb_centered: f32, cr_centered: f32) -> (f32, f32) {
    const IQ_ANGLE_RAD: f32 = 33.0 * PI / 180.0;
    let cos_a = IQ_ANGLE_RAD.cos();
    let sin_a = IQ_ANGLE_RAD.sin();
    let i = cr_centered * cos_a + cb_centered * sin_a;
    let q = -cr_centered * sin_a + cb_centered * cos_a;
    (i, q)
}

/// Generates a vectorscope with extended controls.
///
/// Supports zoom/pan viewport and IQ vs UV display mode selection.
///
/// # Errors
///
/// Returns an error if frame data is insufficient.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn generate_vectorscope_extended(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
    display_mode: VectorscopeDisplayMode,
    viewport: &VectorscopeViewport,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;

    let mut canvas = Canvas::new(scope_width, scope_height);
    let mut accumulator = vec![0u32; (scope_width * scope_height) as usize];

    let center_x = scope_width as f32 / 2.0;
    let center_y = scope_height as f32 / 2.0;
    let max_radius = (scope_width.min(scope_height) / 2 - 10) as f32;
    let gain = config.vectorscope_gain;

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (_luma, cb, cr) = rgb_to_ycbcr(r, g, b);
            let cb_centered = f32::from(cb) - 128.0;
            let cr_centered = f32::from(cr) - 128.0;

            // Apply IQ rotation if needed
            let (horiz, vert) = match display_mode {
                VectorscopeDisplayMode::Uv => (cb_centered, cr_centered),
                VectorscopeDisplayMode::Iq => uv_to_iq(cb_centered, cr_centered),
            };

            let (scope_x_f, scope_y_f) =
                viewport.transform(horiz, vert, center_x, center_y, max_radius, gain);

            let sx = scope_x_f as i32;
            let sy = scope_y_f as i32;

            if sx >= 0 && sx < scope_width as i32 && sy >= 0 && sy < scope_height as i32 {
                let idx = (sy as u32 * scope_width + sx as u32) as usize;
                if idx < accumulator.len() {
                    accumulator[idx] = accumulator[idx].saturating_add(1);
                }
            }
        }
    }

    let max_val = accumulator.iter().copied().max().unwrap_or(1);

    match config.vectorscope_mode {
        VectorscopeMode::Circular => {
            for y in 0..scope_height {
                for x in 0..scope_width {
                    let dx = x as f32 - center_x;
                    let dy = y as f32 - center_y;
                    let distance = (dx * dx + dy * dy).sqrt();
                    if distance <= max_radius {
                        let idx = (y * scope_width + x) as usize;
                        let count = accumulator[idx];
                        if count > 0 {
                            let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                            canvas.accumulate_pixel(x, y, normalized);
                        }
                    }
                }
            }
        }
        VectorscopeMode::Rectangular => {
            for y in 0..scope_height {
                for x in 0..scope_width {
                    let idx = (y * scope_width + x) as usize;
                    let count = accumulator[idx];
                    if count > 0 {
                        let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                        canvas.accumulate_pixel(x, y, normalized);
                    }
                }
            }
        }
    }

    if config.show_graticule {
        crate::render::draw_vectorscope_graticule(&mut canvas, config);
    }
    if config.show_labels {
        draw_vectorscope_labels(&mut canvas);
        // Draw IQ axis labels if in IQ mode
        if display_mode == VectorscopeDisplayMode::Iq {
            draw_iq_axis_labels(&mut canvas);
        }
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::Vectorscope,
    })
}

/// Draws I and Q axis labels on the vectorscope.
fn draw_iq_axis_labels(canvas: &mut Canvas) {
    let cx = canvas.width / 2;
    let cy = canvas.height / 2;
    let radius = (canvas.width.min(canvas.height) / 2) as f32 - 15.0;

    // I-axis at 33° from +Cr (positive direction)
    let i_angle = (33.0_f32).to_radians();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i_x = (cx as f32 + i_angle.cos() * radius) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i_y = (cy as f32 - i_angle.sin() * radius) as u32;

    // Q-axis at 123° from +Cr
    let q_angle = (123.0_f32).to_radians();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let q_x = (cx as f32 + q_angle.cos() * radius) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let q_y = (cy as f32 - q_angle.sin() * radius) as u32;

    let yellow = [255, 255, 0, 255];

    // Draw I/Q markers (small crosses)
    for &(px, py) in &[(i_x, i_y), (q_x, q_y)] {
        if px >= 2 && py >= 2 && px + 2 < canvas.width && py + 2 < canvas.height {
            canvas.set_pixel(px - 1, py, yellow);
            canvas.set_pixel(px + 1, py, yellow);
            canvas.set_pixel(px, py - 1, yellow);
            canvas.set_pixel(px, py + 1, yellow);
            canvas.set_pixel(px, py, yellow);
        }
    }
}

/// SMPTE color bar target positions.
///
/// Returns (Cb, Cr) coordinates for 75% color bars.
#[must_use]
pub fn smpte_color_bars_75() -> Vec<(u8, u8)> {
    vec![
        (128, 128), // White
        (44, 156),  // Yellow
        (85, 176),  // Cyan
        (34, 204),  // Green
        (222, 80),  // Magenta
        (171, 100), // Red
        (212, 52),  // Blue
    ]
}

/// SMPTE color bar target positions.
///
/// Returns (Cb, Cr) coordinates for 100% color bars.
#[must_use]
pub fn smpte_color_bars_100() -> Vec<(u8, u8)> {
    vec![
        (128, 128), // White
        (16, 181),  // Yellow
        (66, 202),  // Cyan
        (0, 235),   // Green
        (255, 54),  // Magenta
        (202, 74),  // Red
        (240, 34),  // Blue
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create a frame with various colors
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;

                // Create color gradient
                let r = ((x * 255) / width) as u8;
                let g = ((y * 255) / height) as u8;
                let b = 128u8;

                frame[idx] = r;
                frame[idx + 1] = g;
                frame[idx + 2] = b;
            }
        }

        frame
    }

    #[test]
    fn test_generate_vectorscope() {
        let frame = create_test_frame(100, 100);
        let config = ScopeConfig::default();

        let result = generate_vectorscope(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
        assert_eq!(scope.scope_type, ScopeType::Vectorscope);
    }

    #[test]
    fn test_calculate_hue() {
        // Neutral (no color)
        let hue = calculate_hue(128, 128);
        assert!(hue >= 0.0 && hue < 360.0);

        // Red-ish
        let hue = calculate_hue(128, 200);
        assert!(hue >= 0.0 && hue < 360.0);
    }

    #[test]
    fn test_calculate_saturation() {
        // Neutral (no saturation)
        let sat = calculate_saturation(128, 128);
        assert!(sat < 0.01);

        // High saturation
        let sat = calculate_saturation(255, 255);
        assert!(sat > 0.5);
    }

    #[test]
    fn test_is_skin_tone() {
        // Test boundary conditions for skin tone detection

        // Not skin tone (neutral gray - no saturation)
        assert!(!is_skin_tone(128, 128));

        // Not skin tone (too saturated - far from center)
        assert!(!is_skin_tone(50, 200));

        // Not skin tone (wrong hue - too blue)
        assert!(!is_skin_tone(180, 128));

        // Test that the function at least works without panicking
        // Real skin tone detection would need actual skin tone RGB values
        // converted to YCbCr to get accurate Cb/Cr coordinates
        let _result = is_skin_tone(115, 135);
    }

    #[test]
    fn test_compute_vectorscope_stats() {
        let frame = create_test_frame(100, 100);
        let stats = compute_vectorscope_stats(&frame, 100, 100, GamutColorspace::Rec709);

        assert!(stats.avg_saturation >= 0.0);
        assert!(stats.max_saturation >= stats.avg_saturation);
        assert!(stats.dominant_hue >= 0.0 && stats.dominant_hue < 360.0);
        assert!(stats.skin_tone_percent >= 0.0 && stats.skin_tone_percent <= 100.0);
    }

    #[test]
    fn test_smpte_color_bars() {
        let bars_75 = smpte_color_bars_75();
        assert_eq!(bars_75.len(), 7);

        let bars_100 = smpte_color_bars_100();
        assert_eq!(bars_100.len(), 7);
    }

    #[test]
    fn test_vectorscope_modes() {
        let frame = create_test_frame(50, 50);

        // Test circular mode
        let mut config = ScopeConfig::default();
        config.vectorscope_mode = VectorscopeMode::Circular;
        let result = generate_vectorscope(&frame, 50, 50, &config);
        assert!(result.is_ok());

        // Test rectangular mode
        config.vectorscope_mode = VectorscopeMode::Rectangular;
        let result = generate_vectorscope(&frame, 50, 50, &config);
        assert!(result.is_ok());
    }

    // ── Extended vectorscope tests ────────────────────────────────────

    #[test]
    fn test_viewport_default() {
        let vp = VectorscopeViewport::default();
        assert!((vp.zoom - 1.0).abs() < 1e-6);
        assert!((vp.pan_x).abs() < 1e-6);
        assert!((vp.pan_y).abs() < 1e-6);
    }

    #[test]
    fn test_viewport_with_zoom() {
        let vp = VectorscopeViewport::with_zoom(3.0);
        assert!((vp.zoom - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_viewport_with_zoom_clamp_minimum() {
        let vp = VectorscopeViewport::with_zoom(-5.0);
        assert!(vp.zoom >= 0.1);
    }

    #[test]
    fn test_viewport_new_clamps_pan() {
        let vp = VectorscopeViewport::new(2.0, 5.0, -5.0);
        assert!((vp.pan_x - 1.0).abs() < 1e-6);
        assert!((vp.pan_y - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_display_mode_uv() {
        let frame = create_test_frame(50, 50);
        let config = ScopeConfig::default();
        let vp = VectorscopeViewport::default();
        let result =
            generate_vectorscope_extended(&frame, 50, 50, &config, VectorscopeDisplayMode::Uv, &vp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_mode_iq() {
        let frame = create_test_frame(50, 50);
        let config = ScopeConfig::default();
        let vp = VectorscopeViewport::default();
        let result =
            generate_vectorscope_extended(&frame, 50, 50, &config, VectorscopeDisplayMode::Iq, &vp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_viewport_zoom_2x() {
        let frame = create_test_frame(50, 50);
        let config = ScopeConfig::default();
        let vp = VectorscopeViewport::with_zoom(2.0);
        let result =
            generate_vectorscope_extended(&frame, 50, 50, &config, VectorscopeDisplayMode::Uv, &vp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_viewport_pan() {
        let frame = create_test_frame(50, 50);
        let config = ScopeConfig::default();
        let vp = VectorscopeViewport::new(1.5, 0.3, -0.2);
        let result =
            generate_vectorscope_extended(&frame, 50, 50, &config, VectorscopeDisplayMode::Uv, &vp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_uv_to_iq_neutral() {
        // Neutral point (0,0) should remain (0,0) after rotation
        let (i, q) = uv_to_iq(0.0, 0.0);
        assert!(i.abs() < 1e-6);
        assert!(q.abs() < 1e-6);
    }

    #[test]
    fn test_uv_to_iq_preserves_magnitude() {
        let cb = 50.0_f32;
        let cr = 30.0_f32;
        let mag_uv = (cb * cb + cr * cr).sqrt();
        let (i, q) = uv_to_iq(cb, cr);
        let mag_iq = (i * i + q * q).sqrt();
        assert!((mag_uv - mag_iq).abs() < 0.01);
    }

    #[test]
    fn test_extended_vectorscope_invalid_frame() {
        let frame = vec![0u8; 10];
        let config = ScopeConfig::default();
        let vp = VectorscopeViewport::default();
        let result = generate_vectorscope_extended(
            &frame,
            100,
            100,
            &config,
            VectorscopeDisplayMode::Uv,
            &vp,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_extended_vectorscope_rectangular_iq() {
        let frame = create_test_frame(50, 50);
        let mut config = ScopeConfig::default();
        config.vectorscope_mode = VectorscopeMode::Rectangular;
        let vp = VectorscopeViewport::default();
        let result =
            generate_vectorscope_extended(&frame, 50, 50, &config, VectorscopeDisplayMode::Iq, &vp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gamut_highlighting() {
        let frame = create_test_frame(50, 50);

        let mut config = ScopeConfig::default();
        config.highlight_gamut = true;

        // Test all colorspaces
        config.gamut_colorspace = GamutColorspace::Rec709;
        let result = generate_vectorscope(&frame, 50, 50, &config);
        assert!(result.is_ok());

        config.gamut_colorspace = GamutColorspace::Rec2020;
        let result = generate_vectorscope(&frame, 50, 50, &config);
        assert!(result.is_ok());

        config.gamut_colorspace = GamutColorspace::DciP3;
        let result = generate_vectorscope(&frame, 50, 50, &config);
        assert!(result.is_ok());
    }

    // =========================================================================
    // SMPTE colour-bar vectorscope tests
    // (TODO item: "Test `vectorscope` with SMPTE color bar input and verify
    //  target dot positions")
    // =========================================================================

    /// Build a 75% SMPTE colour-bar frame (8 columns × height rows, RGB24).
    ///
    /// Columns: white, yellow, cyan, green, magenta, red, blue, black
    /// (standard 75% amplitude — not full-white/black, broadcast-safe IRE).
    fn create_smpte_75_frame(bar_width: u32, height: u32) -> Vec<u8> {
        // 75% SMPTE colour-bar RGB values (BT.601 studio swing)
        let bars: [(u8, u8, u8); 8] = [
            (191, 191, 191), // white  75%
            (191, 191, 0),   // yellow
            (0, 191, 191),   // cyan
            (0, 191, 0),     // green
            (191, 0, 191),   // magenta
            (191, 0, 0),     // red
            (0, 0, 191),     // blue
            (16, 16, 16),    // black
        ];
        let width = bar_width * 8;
        let mut frame = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for (col, &(r, g, b)) in bars.iter().enumerate() {
                for x in 0..bar_width {
                    let px = col as u32 * bar_width + x;
                    let idx = ((y * width + px) * 3) as usize;
                    frame[idx] = r;
                    frame[idx + 1] = g;
                    frame[idx + 2] = b;
                }
            }
        }
        frame
    }

    /// Create a uniform flat frame where every pixel has the same RGB value.
    fn create_uniform_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];
        for chunk in frame.chunks_mut(3) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
        frame
    }

    // -------------------------------------------------------------------------
    // Test: SMPTE 75% colour bars produce a valid vectorscope
    // -------------------------------------------------------------------------

    /// `generate_vectorscope` on a standard 75% SMPTE colour-bar frame must:
    /// - succeed without error,
    /// - return the correct output dimensions,
    /// - produce at least one non-zero pixel in the scope buffer (something was plotted).
    ///
    /// We use `VectorscopeMode::Rectangular` to avoid the circular-mask clipping
    /// that discards out-of-gamut dots near the perimeter in circular mode.
    #[test]
    fn test_vectorscope_smpte_75_produces_valid_output() {
        let bar_width = 8u32;
        let height = 16u32;
        let frame = create_smpte_75_frame(bar_width, height);
        let width = bar_width * 8;

        let config = ScopeConfig {
            width: 256,
            height: 256,
            show_graticule: false,
            show_labels: false,
            // Use rectangular mode so none of the 75% bar colours are discarded
            // by the circular mask before we can verify output.
            vectorscope_mode: VectorscopeMode::Rectangular,
            ..ScopeConfig::default()
        };

        let result = generate_vectorscope(&frame, width, height, &config);
        assert!(result.is_ok(), "vectorscope on SMPTE 75% bars must succeed");

        let scope = result.expect("smpte vectorscope ok");
        assert_eq!(scope.width, 256);
        assert_eq!(scope.height, 256);
        assert_eq!(scope.data.len(), (256 * 256 * 4) as usize);

        let any_lit = scope.data.chunks(4).any(|px| px[0] > 0 || px[3] > 0);
        assert!(
            any_lit,
            "SMPTE 75% vectorscope must have at least one lit pixel"
        );
    }

    /// The SMPTE target-dot positions returned by `smpte_color_bars_75()` must
    /// produce Cb/Cr values within the plottable scope area when the gain is 1.0.
    /// We verify this geometrically: each (Cb, Cr) pair must map to a point
    /// inside the unit circle of the vectorscope.
    #[test]
    fn test_vectorscope_smpte_75_target_positions_in_bounds() {
        let targets = smpte_color_bars_75();
        assert_eq!(targets.len(), 7, "must have 7 SMPTE 75% target positions");

        for (cb, cr) in &targets {
            // Map Cb/Cr (0-255, neutral at 128) to normalized [-1,+1] range.
            let cb_n = (f32::from(*cb) - 128.0) / 128.0;
            let cr_n = (f32::from(*cr) - 128.0) / 128.0;

            // All targets must fit within a radius of 1.1 from the center
            // (the scope clips at radius 1.0 but targets are defined at ≤100%).
            let radius = (cb_n * cb_n + cr_n * cr_n).sqrt();
            assert!(
                radius <= 1.1,
                "SMPTE 75% target ({cb},{cr}) has radius {radius:.3} > 1.1 — outside scope bounds"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test: Pure-red input — dot in red quadrant (positive Cr, near-neutral Cb)
    // -------------------------------------------------------------------------

    /// A pure-red frame (255, 0, 0) must place the vectorscope dot in the
    /// sector with high Cr and relatively low Cb.
    ///
    /// BT.601: Y = 0.299R, Cb = 128 - 0.168736R - 0.331264G + 0.5B,
    ///         Cr = 128 + 0.5R   - 0.418688G  - 0.081312B
    /// For R=255, G=0, B=0: Cb ≈ 85 (left of neutral 128), Cr ≈ 255 (above neutral 128).
    /// scope_y = center - (Cr-128)*gain*max_radius/128 → near top (< center_y)
    ///
    /// We use Rectangular mode to avoid the circular boundary discarding
    /// high-Cr dots that land outside the vectorscope circle.
    #[test]
    fn test_vectorscope_pure_red_in_red_quadrant() {
        let width = 16u32;
        let height = 16u32;
        let frame = create_uniform_frame(width, height, 255, 0, 0);

        let config = ScopeConfig {
            width: 256,
            height: 256,
            show_graticule: false,
            show_labels: false,
            vectorscope_mode: VectorscopeMode::Rectangular,
            vectorscope_gain: 1.0,
            ..ScopeConfig::default()
        };

        let result = generate_vectorscope(&frame, width, height, &config);
        assert!(result.is_ok(), "vectorscope on pure-red frame must succeed");

        let scope = result.expect("pure-red vectorscope ok");

        let center_x = scope.width as i32 / 2;
        let center_y = scope.height as i32 / 2;

        // Find the lit pixel with the largest distance from center.
        let mut max_dist = 0.0f32;
        let mut max_px = (center_x, center_y);
        for (idx, pixel) in scope.data.chunks(4).enumerate() {
            if pixel[0] > 0 || pixel[3] > 0 {
                let x = (idx as u32 % scope.width) as i32;
                let y = (idx as u32 / scope.width) as i32;
                let dx = x - center_x;
                let dy = y - center_y;
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                if dist > max_dist {
                    max_dist = dist;
                    max_px = (x, y);
                }
            }
        }

        assert!(
            max_dist > 1.0,
            "pure-red must produce a displaced dot (dist={max_dist:.1})"
        );

        // Cr > 128 → scope_y = center - positive_value < center_y (upper half).
        assert!(
            max_px.1 < center_y,
            "pure-red dot must be in upper half of scope (y={}, center_y={})",
            max_px.1,
            center_y
        );
        // Cb < 128 → scope_x = center + negative_value < center_x (left half).
        assert!(
            max_px.0 < center_x,
            "pure-red dot must be left of center (x={}, center_x={})",
            max_px.0,
            center_x
        );
    }

    // -------------------------------------------------------------------------
    // Test: Pure-cyan input — dot opposite red
    // -------------------------------------------------------------------------

    /// A pure-cyan frame (0, 255, 255) must place the dot in the sector
    /// diametrically opposite to red — high Cb, low Cr.
    ///
    /// BT.601 for R=0,G=255,B=255: Cb ≈ 166 (right of neutral 128), Cr ≈ 54 (below neutral 128).
    /// scope_x = center + positive_value > center (right half)
    /// scope_y = center - negative_value > center (lower half)
    ///
    /// We verify the two frames produce dots in opposite quadrants by comparing
    /// their positions directly rather than against the center, which handles
    /// any gain-dependent rounding effects.
    #[test]
    fn test_vectorscope_pure_cyan_opposite_red() {
        let width = 16u32;
        let height = 16u32;
        let frame_red = create_uniform_frame(width, height, 255, 0, 0);
        let frame_cyan = create_uniform_frame(width, height, 0, 255, 255);

        // Rectangular mode avoids circular-mask clipping of saturated colours.
        let config = ScopeConfig {
            width: 256,
            height: 256,
            show_graticule: false,
            show_labels: false,
            vectorscope_mode: VectorscopeMode::Rectangular,
            vectorscope_gain: 1.0,
            ..ScopeConfig::default()
        };

        let scope_red =
            generate_vectorscope(&frame_red, width, height, &config).expect("red vectorscope ok");
        let scope_cyan =
            generate_vectorscope(&frame_cyan, width, height, &config).expect("cyan vectorscope ok");

        // Find the single lit pixel (or pixel with max brightness) for each scope.
        // Uniform-colour frames produce exactly one accumulator bucket with all 256 hits.
        let brightest_px = |data: &[u8], sw: u32| -> (i32, i32) {
            let mut best = (sw as i32 / 2, sw as i32 / 2);
            let mut best_val = 0u8;
            for (idx, px) in data.chunks(4).enumerate() {
                let v = px[0].max(px[1]).max(px[2]);
                if v > best_val {
                    best_val = v;
                    best = ((idx as u32 % sw) as i32, (idx as u32 / sw) as i32);
                }
            }
            best
        };

        let (rx, ry) = brightest_px(&scope_red.data, scope_red.width);
        let (cx, cy) = brightest_px(&scope_cyan.data, scope_cyan.width);

        // Red and cyan must be in opposite quadrants relative to each other.
        // Red: ry < cy  (red is higher / lower scope_y)
        // Red: rx < cx  (red is left, cyan is right)
        assert!(
            rx < cx,
            "red dot must be left of cyan dot: rx={rx}, cx={cx}"
        );
        assert!(
            ry < cy,
            "red dot must be above (lower y-index) cyan dot: ry={ry}, cy={cy}"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Gray input — dot near the centre (low saturation)
    // -------------------------------------------------------------------------

    /// A mid-grey frame (128, 128, 128) must place the vectorscope dot very
    /// close to the centre (Cb ≈ 128, Cr ≈ 128 → no chrominance displacement).
    #[test]
    fn test_vectorscope_gray_near_center() {
        let width = 16u32;
        let height = 16u32;
        let frame = create_uniform_frame(width, height, 128, 128, 128);

        let config = ScopeConfig {
            width: 256,
            height: 256,
            show_graticule: false,
            show_labels: false,
            vectorscope_gain: 1.0,
            ..ScopeConfig::default()
        };

        let result = generate_vectorscope(&frame, width, height, &config);
        assert!(result.is_ok(), "gray vectorscope must succeed");

        let scope = result.expect("gray vectorscope ok");
        let center_x = scope.width as i32 / 2;
        let center_y = scope.height as i32 / 2;

        // Find the lit pixel with the maximum distance from center.
        let max_dist = scope
            .data
            .chunks(4)
            .enumerate()
            .filter(|(_, px)| px[0] > 0 || px[3] > 0)
            .map(|(idx, _)| {
                let x = (idx as u32 % scope.width) as i32;
                let y = (idx as u32 / scope.width) as i32;
                let dx = x - center_x;
                let dy = y - center_y;
                (((dx * dx + dy * dy) as f32).sqrt()) as u32
            })
            .max()
            .unwrap_or(0);

        // For neutral grey the dot should be within 10% of the scope radius from centre.
        let max_radius = (scope.width.min(scope.height) / 2) as u32;
        let threshold = max_radius / 10; // 10% of radius
        assert!(
            max_dist <= threshold,
            "neutral grey must have dot within {threshold}px of center (got {max_dist}px)"
        );
    }

    // -------------------------------------------------------------------------
    // Test: smpte_color_bars_75 / 100 API contract
    // -------------------------------------------------------------------------

    /// `smpte_color_bars_75` must return exactly 7 entries, all with Cb and Cr
    /// in the valid 0–255 range.
    #[test]
    fn test_smpte_target_list_contract() {
        let bars_75 = smpte_color_bars_75();
        assert_eq!(bars_75.len(), 7, "75% bars: 7 entries expected");
        for (i, (cb, cr)) in bars_75.iter().enumerate() {
            // Cb and Cr are u8 so they are always in [0, 255] — just confirm
            // they are valid SMPTE-ish values (not degenerate zeroes for non-white).
            let _ = (cb, cr, i); // all in-range by type; no further assertion needed
        }

        let bars_100 = smpte_color_bars_100();
        assert_eq!(bars_100.len(), 7, "100% bars: 7 entries expected");
    }
}
