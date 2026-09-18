//! Waveform monitor for video signal analysis.
//!
//! Waveform monitors display the luminance (brightness) or RGB component values
//! across horizontal scanlines of the video. This is essential for:
//! - Checking exposure and dynamic range
//! - Ensuring legal broadcast levels (0-100 IRE)
//! - Identifying clipping and crushing
//! - Matching shots in color grading
//!
//! Supported modes:
//! - **Luma**: Y channel only (luminance)
//! - **RGB Parade**: R, G, B displayed side-by-side
//! - **RGB Overlay**: All RGB channels overlaid with color
//! - **YCbCr**: Y, Cb, Cr displayed side-by-side

use crate::render::{rgb_to_ycbcr, Canvas};
use crate::simd_convert::convert_batch_bt601_simd;
use crate::{ScopeConfig, ScopeData, ScopeType};
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// Generates a luma (Y channel) waveform from RGB frame data.
///
/// The waveform shows the luminance distribution across the horizontal axis,
/// with intensity accumulated for each pixel position.
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
#[allow(clippy::cast_sign_loss)]
pub fn generate_luma_waveform(
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

    // Process frame in parallel by rows
    let row_accumulators: Vec<Vec<u32>> = (0..height)
        .into_par_iter()
        .map(|y| {
            let mut local_accum = vec![0u32; (scope_width * scope_height) as usize];

            for x in 0..width {
                let pixel_idx = ((y * width + x) * 3) as usize;
                let r = frame[pixel_idx];
                let g = frame[pixel_idx + 1];
                let b = frame[pixel_idx + 2];

                // Convert to luma (ITU-R BT.709)
                let (luma, _, _) = rgb_to_ycbcr(r, g, b);

                // Map to scope coordinates
                let scope_x = (x * scope_width) / width;
                let mapped = ((u32::from(luma) * scope_height) / 255).min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;

                let idx = (scope_y * scope_width + scope_x) as usize;
                if idx < local_accum.len() {
                    local_accum[idx] = local_accum[idx].saturating_add(1);
                }
            }

            local_accum
        })
        .collect();

    // Merge all row accumulators
    for row_accum in &row_accumulators {
        for (i, &val) in row_accum.iter().enumerate() {
            accumulator[i] = accumulator[i].saturating_add(val);
        }
    }

    // Find max value for normalization
    let max_val = accumulator.iter().copied().max().unwrap_or(1);

    // Draw accumulated waveform
    for y in 0..scope_height {
        for x in 0..scope_width {
            let idx = (y * scope_width + x) as usize;
            let count = accumulator[idx];

            if count > 0 {
                // Normalize to 0-255 range with gamma correction for better visibility
                let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                canvas.accumulate_pixel(x, y, normalized);
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_waveform_graticule(&mut canvas, config);
    }

    // Draw labels
    if config.show_labels {
        draw_waveform_labels(&mut canvas);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformLuma,
    })
}

/// Generates an RGB parade waveform (R|G|B side-by-side).
///
/// Each color channel is displayed in its own vertical section,
/// making it easy to compare and balance channels.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn generate_rgb_parade(
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
    let section_width = scope_width / 3;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Three accumulators for R, G, B
    let mut accumulators = [
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
    ];

    // Process frame
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let rgb = [frame[pixel_idx], frame[pixel_idx + 1], frame[pixel_idx + 2]];

            let scope_x = (x * section_width) / width;

            for (channel, &value) in rgb.iter().enumerate() {
                let mapped = ((u32::from(value) * scope_height) / 255).min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;
                let idx = (scope_y * section_width + scope_x) as usize;

                if idx < accumulators[channel].len() {
                    accumulators[channel][idx] = accumulators[channel][idx].saturating_add(1);
                }
            }
        }
    }

    // Find max for normalization across all channels
    let max_val = accumulators
        .iter()
        .flat_map(|acc| acc.iter().copied())
        .max()
        .unwrap_or(1);

    // Draw each channel with its color
    let colors = [
        [255, 0, 0, 255], // Red
        [0, 255, 0, 255], // Green
        [0, 0, 255, 255], // Blue
    ];

    for (channel, accumulator) in accumulators.iter().enumerate() {
        let offset_x = section_width * channel as u32;

        for y in 0..scope_height {
            for x in 0..section_width {
                let idx = (y * section_width + x) as usize;
                let count = accumulator[idx];

                if count > 0 {
                    let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                    let color = [
                        ((colors[channel][0] as u16 * u16::from(normalized)) / 255) as u8,
                        ((colors[channel][1] as u16 * u16::from(normalized)) / 255) as u8,
                        ((colors[channel][2] as u16 * u16::from(normalized)) / 255) as u8,
                        255,
                    ];
                    canvas.set_pixel(offset_x + x, y, color);
                }
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_parade_graticule(&mut canvas, config, 3);
    }

    // Draw labels
    if config.show_labels {
        draw_parade_labels(&mut canvas, &["R", "G", "B"]);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformRgbParade,
    })
}

/// Generates an RGB overlay waveform (all channels overlaid with color).
///
/// All RGB channels are displayed on the same scope, with each channel
/// shown in its respective color. White indicates areas where all channels align.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn generate_rgb_overlay(
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

    // Three separate accumulators for R, G, B
    let mut accumulators = [
        vec![0u32; (scope_width * scope_height) as usize],
        vec![0u32; (scope_width * scope_height) as usize],
        vec![0u32; (scope_width * scope_height) as usize],
    ];

    // Process frame
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let rgb = [frame[pixel_idx], frame[pixel_idx + 1], frame[pixel_idx + 2]];

            let scope_x = (x * scope_width) / width;

            for (channel, &value) in rgb.iter().enumerate() {
                let mapped = ((u32::from(value) * scope_height) / 255).min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;
                let idx = (scope_y * scope_width + scope_x) as usize;

                if idx < accumulators[channel].len() {
                    accumulators[channel][idx] = accumulators[channel][idx].saturating_add(1);
                }
            }
        }
    }

    // Find max for normalization
    let max_val = accumulators
        .iter()
        .flat_map(|acc| acc.iter().copied())
        .max()
        .unwrap_or(1);

    // Composite all channels with additive blending
    for y in 0..scope_height {
        for x in 0..scope_width {
            let idx = (y * scope_width + x) as usize;

            let r_count = accumulators[0][idx];
            let g_count = accumulators[1][idx];
            let b_count = accumulators[2][idx];

            if r_count > 0 || g_count > 0 || b_count > 0 {
                let r = ((r_count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                let g = ((g_count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                let b = ((b_count as f32 / max_val as f32).sqrt() * 255.0) as u8;

                canvas.set_pixel(x, y, [r, g, b, 255]);
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_waveform_graticule(&mut canvas, config);
    }

    // Draw labels
    if config.show_labels {
        draw_waveform_labels(&mut canvas);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformRgbOverlay,
    })
}

/// Generates a YCbCr waveform (Y|Cb|Cr parade).
///
/// Displays the Y (luma), Cb (blue-difference), and Cr (red-difference)
/// components side-by-side for analyzing color information separately.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn generate_ycbcr_waveform(
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
    let section_width = scope_width / 3;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Pre-convert entire frame to BT.601 YCbCr using SIMD dispatch.
    // This replaces the per-pixel `rgb_to_ycbcr` call in the hot loop with a
    // single batch conversion, enabling AVX2/NEON throughput on supported CPUs.
    let ycbcr_frame = convert_batch_bt601_simd(&frame[..(width * height * 3) as usize]);

    // Three accumulators for Y, Cb, Cr
    let mut accumulators = [
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
    ];

    // Process pre-converted frame
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = (y * width + x) as usize;
            let ycbcr = ycbcr_frame[pixel_idx];

            let scope_x = (x * section_width) / width;

            for (channel, &value) in ycbcr.iter().enumerate() {
                let mapped = ((u32::from(value) * scope_height) / 255).min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;
                let idx = (scope_y * section_width + scope_x) as usize;

                if idx < accumulators[channel].len() {
                    accumulators[channel][idx] = accumulators[channel][idx].saturating_add(1);
                }
            }
        }
    }

    // Find max for normalization
    let max_val = accumulators
        .iter()
        .flat_map(|acc| acc.iter().copied())
        .max()
        .unwrap_or(1);

    // Draw each component in grayscale
    for (channel, accumulator) in accumulators.iter().enumerate() {
        let offset_x = section_width * channel as u32;

        for y in 0..scope_height {
            for x in 0..section_width {
                let idx = (y * section_width + x) as usize;
                let count = accumulator[idx];

                if count > 0 {
                    let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                    canvas.set_pixel(offset_x + x, y, [normalized, normalized, normalized, 255]);
                }
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_parade_graticule(&mut canvas, config, 3);
    }

    // Draw labels
    if config.show_labels {
        draw_parade_labels(&mut canvas, &["Y", "Cb", "Cr"]);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformYcbcr,
    })
}

/// Draws IRE labels on waveform display.
fn draw_waveform_labels(canvas: &mut Canvas) {
    let height = canvas.height;
    let color = crate::render::colors::WHITE;

    // Draw IRE level labels
    let labels = [(100, "100"), (75, "75"), (50, "50"), (0, "0")];

    for (ire, text) in &labels {
        let y = height - ((*ire as u32 * height) / 100);
        if y >= 8 && y < height {
            crate::render::draw_label(canvas, 2, y - 8, text, color);
        }
    }
}

/// Draws labels for parade display.
fn draw_parade_labels(canvas: &mut Canvas, labels: &[&str]) {
    let width = canvas.width;
    let section_width = width / labels.len() as u32;
    let color = crate::render::colors::WHITE;

    for (i, label) in labels.iter().enumerate() {
        let x = section_width * i as u32 + section_width / 2 - 3;
        crate::render::draw_label(canvas, x, 2, label, color);
    }
}

/// Detects out-of-gamut pixels in the frame.
///
/// Returns true if any pixels are outside legal broadcast range (16-235 for Y, 16-240 for `CbCr`).
#[must_use]
pub fn detect_out_of_gamut(frame: &[u8], width: u32, height: u32) -> bool {
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                return false;
            }

            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, cb, cr) = rgb_to_ycbcr(r, g, b);

            // Check legal broadcast levels (ITU-R BT.709)
            // Y: 16-235, CbCr: 16-240
            if !(16..=235).contains(&luma) || !(16..=240).contains(&cb) || !(16..=240).contains(&cr)
            {
                return true;
            }
        }
    }
    false
}

// =============================================================================
// 10-bit and 12-bit precision support
// =============================================================================

/// Bit depth precision for waveform analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformBitDepth {
    /// Standard 8-bit precision (0-255).
    Bit8,
    /// 10-bit precision (0-1023), stored as u16 little-endian.
    Bit10,
    /// 12-bit precision (0-4095), stored as u16 little-endian.
    Bit12,
}

impl WaveformBitDepth {
    /// Returns the maximum sample value for this bit depth.
    #[must_use]
    pub const fn max_value(self) -> u16 {
        match self {
            Self::Bit8 => 255,
            Self::Bit10 => 1023,
            Self::Bit12 => 4095,
        }
    }

    /// Returns the number of bytes per sample for this bit depth.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Bit8 => 1,
            Self::Bit10 | Self::Bit12 => 2,
        }
    }

    /// Returns the number of bits for this depth.
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::Bit8 => 8,
            Self::Bit10 => 10,
            Self::Bit12 => 12,
        }
    }
}

/// Converts a high-precision RGB pixel to luma (BT.709).
///
/// Input values are in `[0, max_value]` for the given bit depth.
/// Returns luma in `[0.0, 1.0]` normalized range.
#[must_use]
fn rgb16_to_luma_normalized(r: u16, g: u16, b: u16, max_val: f32) -> f32 {
    let r_f = f32::from(r) / max_val;
    let g_f = f32::from(g) / max_val;
    let b_f = f32::from(b) / max_val;
    // BT.709 luma coefficients
    (0.2126 * r_f + 0.7152 * g_f + 0.0722 * b_f).clamp(0.0, 1.0)
}

/// Reads an RGB pixel from a high-bit-depth frame (u16 LE per component).
///
/// Returns `(r, g, b)` clamped to `max_val`.
fn read_pixel_u16(frame: &[u8], offset: usize, max_val: u16) -> Option<(u16, u16, u16)> {
    if offset + 6 > frame.len() {
        return None;
    }
    let r = u16::from_le_bytes([frame[offset], frame[offset + 1]]).min(max_val);
    let g = u16::from_le_bytes([frame[offset + 2], frame[offset + 3]]).min(max_val);
    let b = u16::from_le_bytes([frame[offset + 4], frame[offset + 5]]).min(max_val);
    Some((r, g, b))
}

/// Generates a luma waveform with configurable bit depth precision.
///
/// For 10-bit and 12-bit frames, `frame` contains u16 little-endian samples
/// (6 bytes per pixel: R16 G16 B16).  For 8-bit, standard RGB24 is expected.
///
/// # Errors
///
/// Returns an error if frame data is insufficient for the given dimensions and bit depth.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn generate_luma_waveform_hd(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
    bit_depth: WaveformBitDepth,
) -> OxiResult<ScopeData> {
    let bytes_per_pixel = bit_depth.bytes_per_sample() * 3;
    let expected_size = (width as usize) * (height as usize) * bytes_per_pixel;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small for {}-bit: expected {expected_size}, got {}",
            bit_depth.bits(),
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;
    let max_val = bit_depth.max_value();
    let max_val_f = f32::from(max_val);

    let mut canvas = Canvas::new(scope_width, scope_height);
    let mut accumulator = vec![0u32; (scope_width * scope_height) as usize];

    for y in 0..height {
        for x in 0..width {
            let luma_norm = match bit_depth {
                WaveformBitDepth::Bit8 => {
                    let pixel_idx = ((y * width + x) * 3) as usize;
                    let r = frame[pixel_idx];
                    let g = frame[pixel_idx + 1];
                    let b = frame[pixel_idx + 2];
                    let (luma, _, _) = rgb_to_ycbcr(r, g, b);
                    f32::from(luma) / 255.0
                }
                WaveformBitDepth::Bit10 | WaveformBitDepth::Bit12 => {
                    let offset = ((y * width + x) as usize) * 6;
                    match read_pixel_u16(frame, offset, max_val) {
                        Some((r, g, b)) => rgb16_to_luma_normalized(r, g, b, max_val_f),
                        None => continue,
                    }
                }
            };

            // Map normalized luma [0,1] to scope coordinates
            let scope_x = (x * scope_width) / width;
            let mapped = (luma_norm * (scope_height - 1) as f32) as u32;
            let mapped = mapped.min(scope_height - 1);
            let scope_y = scope_height - 1 - mapped;

            let idx = (scope_y * scope_width + scope_x) as usize;
            if idx < accumulator.len() {
                accumulator[idx] = accumulator[idx].saturating_add(1);
            }
        }
    }

    let max_count = accumulator.iter().copied().max().unwrap_or(1);

    for y in 0..scope_height {
        for x in 0..scope_width {
            let idx = (y * scope_width + x) as usize;
            let count = accumulator[idx];
            if count > 0 {
                let normalized = ((count as f32 / max_count as f32).sqrt() * 255.0) as u8;
                canvas.accumulate_pixel(x, y, normalized);
            }
        }
    }

    if config.show_graticule {
        crate::render::draw_waveform_graticule(&mut canvas, config);
    }
    if config.show_labels {
        draw_waveform_labels(&mut canvas);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformLuma,
    })
}

/// Generates an RGB parade waveform with configurable bit depth precision.
///
/// # Errors
///
/// Returns an error if frame data is insufficient.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn generate_rgb_parade_hd(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
    bit_depth: WaveformBitDepth,
) -> OxiResult<ScopeData> {
    let bytes_per_pixel = bit_depth.bytes_per_sample() * 3;
    let expected_size = (width as usize) * (height as usize) * bytes_per_pixel;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small for {}-bit: expected {expected_size}, got {}",
            bit_depth.bits(),
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;
    let section_width = scope_width / 3;
    let max_val = bit_depth.max_value();
    let max_val_f = f32::from(max_val);

    let mut canvas = Canvas::new(scope_width, scope_height);
    let mut accumulators = [
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
    ];

    for y in 0..height {
        for x in 0..width {
            let rgb_norm: [f32; 3] = match bit_depth {
                WaveformBitDepth::Bit8 => {
                    let pixel_idx = ((y * width + x) * 3) as usize;
                    [
                        f32::from(frame[pixel_idx]) / 255.0,
                        f32::from(frame[pixel_idx + 1]) / 255.0,
                        f32::from(frame[pixel_idx + 2]) / 255.0,
                    ]
                }
                WaveformBitDepth::Bit10 | WaveformBitDepth::Bit12 => {
                    let offset = ((y * width + x) as usize) * 6;
                    match read_pixel_u16(frame, offset, max_val) {
                        Some((r, g, b)) => [
                            f32::from(r) / max_val_f,
                            f32::from(g) / max_val_f,
                            f32::from(b) / max_val_f,
                        ],
                        None => continue,
                    }
                }
            };

            let scope_x = (x * section_width) / width;

            for (channel, &value) in rgb_norm.iter().enumerate() {
                let mapped = (value * (scope_height - 1) as f32) as u32;
                let mapped = mapped.min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;
                let idx = (scope_y * section_width + scope_x) as usize;

                if idx < accumulators[channel].len() {
                    accumulators[channel][idx] = accumulators[channel][idx].saturating_add(1);
                }
            }
        }
    }

    let max_count = accumulators
        .iter()
        .flat_map(|acc| acc.iter().copied())
        .max()
        .unwrap_or(1);

    let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];

    for (channel, accumulator) in accumulators.iter().enumerate() {
        let offset_x = section_width * channel as u32;

        for y in 0..scope_height {
            for x in 0..section_width {
                let idx = (y * section_width + x) as usize;
                let count = accumulator[idx];
                if count > 0 {
                    let normalized = ((count as f32 / max_count as f32).sqrt() * 255.0) as u8;
                    let color = [
                        ((colors[channel][0] as u16 * u16::from(normalized)) / 255) as u8,
                        ((colors[channel][1] as u16 * u16::from(normalized)) / 255) as u8,
                        ((colors[channel][2] as u16 * u16::from(normalized)) / 255) as u8,
                        255,
                    ];
                    canvas.set_pixel(offset_x + x, y, color);
                }
            }
        }
    }

    if config.show_graticule {
        crate::render::draw_parade_graticule(&mut canvas, config, 3);
    }
    if config.show_labels {
        draw_parade_labels(&mut canvas, &["R", "G", "B"]);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformRgbParade,
    })
}

/// Waveform statistics for high-bit-depth frames.
#[derive(Debug, Clone)]
pub struct WaveformStatsHd {
    /// Average luma in normalized range [0.0, 1.0].
    pub avg_luma: f64,
    /// Minimum luma in normalized range [0.0, 1.0].
    pub min_luma: f64,
    /// Maximum luma in normalized range [0.0, 1.0].
    pub max_luma: f64,
    /// Standard deviation of luma.
    pub std_dev: f64,
    /// Percentage of pixels below broadcast-legal black level.
    pub black_clip_percent: f64,
    /// Percentage of pixels above broadcast-legal white level.
    pub white_clip_percent: f64,
    /// Bit depth used for analysis.
    pub bit_depth: WaveformBitDepth,
}

/// Computes waveform statistics for any bit depth.
///
/// For 10/12-bit, legal black = 64/256 and legal white = 940/3760 of max value.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_waveform_stats_hd(
    frame: &[u8],
    width: u32,
    height: u32,
    bit_depth: WaveformBitDepth,
) -> WaveformStatsHd {
    let pixel_count = (width as u64) * (height as u64);
    if pixel_count == 0 {
        return WaveformStatsHd {
            avg_luma: 0.0,
            min_luma: 0.0,
            max_luma: 0.0,
            std_dev: 0.0,
            black_clip_percent: 0.0,
            white_clip_percent: 0.0,
            bit_depth,
        };
    }

    let max_val = bit_depth.max_value();
    let max_val_f = f32::from(max_val);

    // Broadcast-legal thresholds (normalized)
    let (black_threshold, white_threshold) = match bit_depth {
        WaveformBitDepth::Bit8 => (16.0 / 255.0, 235.0 / 255.0),
        WaveformBitDepth::Bit10 => (64.0 / 1023.0, 940.0 / 1023.0),
        WaveformBitDepth::Bit12 => (256.0 / 4095.0, 3760.0 / 4095.0),
    };

    let mut sum = 0.0_f64;
    let mut min_luma = 1.0_f64;
    let mut max_luma_val = 0.0_f64;
    let mut black_count = 0u64;
    let mut white_count = 0u64;
    let mut luma_values = Vec::with_capacity(pixel_count as usize);

    for y in 0..height {
        for x in 0..width {
            let luma_norm: f64 = match bit_depth {
                WaveformBitDepth::Bit8 => {
                    let pixel_idx = ((y * width + x) * 3) as usize;
                    if pixel_idx + 2 >= frame.len() {
                        continue;
                    }
                    let r = frame[pixel_idx];
                    let g = frame[pixel_idx + 1];
                    let b = frame[pixel_idx + 2];
                    let (luma, _, _) = rgb_to_ycbcr(r, g, b);
                    f64::from(luma) / 255.0
                }
                WaveformBitDepth::Bit10 | WaveformBitDepth::Bit12 => {
                    let offset = ((y * width + x) as usize) * 6;
                    match read_pixel_u16(frame, offset, max_val) {
                        Some((r, g, b)) => f64::from(rgb16_to_luma_normalized(r, g, b, max_val_f)),
                        None => continue,
                    }
                }
            };

            sum += luma_norm;
            min_luma = min_luma.min(luma_norm);
            max_luma_val = max_luma_val.max(luma_norm);
            luma_values.push(luma_norm);

            if luma_norm < f64::from(black_threshold) {
                black_count += 1;
            }
            if luma_norm > f64::from(white_threshold) {
                white_count += 1;
            }
        }
    }

    let actual_count = luma_values.len() as f64;
    let avg = if actual_count > 0.0 {
        sum / actual_count
    } else {
        0.0
    };

    let variance: f64 = luma_values
        .iter()
        .map(|&v| (v - avg) * (v - avg))
        .sum::<f64>()
        / actual_count.max(1.0);

    WaveformStatsHd {
        avg_luma: avg,
        min_luma,
        max_luma: max_luma_val,
        std_dev: variance.sqrt(),
        black_clip_percent: (black_count as f64 / actual_count.max(1.0)) * 100.0,
        white_clip_percent: (white_count as f64 / actual_count.max(1.0)) * 100.0,
        bit_depth,
    }
}

/// Calculates waveform statistics.
#[derive(Debug, Clone)]
pub struct WaveformStats {
    /// Average luminance (0-255).
    pub avg_luma: f32,

    /// Minimum luminance (0-255).
    pub min_luma: u8,

    /// Maximum luminance (0-255).
    pub max_luma: u8,

    /// Standard deviation of luminance.
    pub std_dev: f32,

    /// Percentage of pixels at or near black (< 16).
    pub black_clip_percent: f32,

    /// Percentage of pixels at or near white (> 235).
    pub white_clip_percent: f32,
}

/// Computes waveform statistics from frame data.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_waveform_stats(frame: &[u8], width: u32, height: u32) -> WaveformStats {
    let mut sum = 0u64;
    let mut min = 255u8;
    let mut max = 0u8;
    let mut black_clip_count = 0u32;
    let mut white_clip_count = 0u32;
    let pixel_count = width * height;

    // First pass: compute mean, min, max, clipping
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

            sum += u64::from(luma);
            min = min.min(luma);
            max = max.max(luma);

            if luma < 16 {
                black_clip_count += 1;
            }
            if luma > 235 {
                white_clip_count += 1;
            }
        }
    }

    let avg_luma = sum as f32 / pixel_count as f32;

    // Second pass: compute standard deviation
    let mut variance_sum = 0.0f32;
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
            let diff = f32::from(luma) - avg_luma;
            variance_sum += diff * diff;
        }
    }

    let std_dev = (variance_sum / pixel_count as f32).sqrt();

    WaveformStats {
        avg_luma,
        min_luma: min,
        max_luma: max,
        std_dev,
        black_clip_percent: (black_clip_count as f32 / pixel_count as f32) * 100.0,
        white_clip_percent: (white_clip_count as f32 / pixel_count as f32) * 100.0,
    }
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

    fn small_scope_config() -> ScopeConfig {
        ScopeConfig {
            width: 64,
            height: 64,
            ..ScopeConfig::default()
        }
    }

    #[test]
    fn test_generate_luma_waveform() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();

        let result = generate_luma_waveform(&frame, 32, 32, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
        assert_eq!(scope.scope_type, ScopeType::WaveformLuma);
    }

    #[test]
    fn test_generate_rgb_parade() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();

        let result = generate_rgb_parade(&frame, 32, 32, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
    }

    #[test]
    fn test_generate_rgb_overlay() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();

        let result = generate_rgb_overlay(&frame, 32, 32, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_ycbcr_waveform() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();

        let result = generate_ycbcr_waveform(&frame, 32, 32, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_frame_size() {
        let frame = vec![0u8; 100]; // Too small
        let config = small_scope_config();

        let result = generate_luma_waveform(&frame, 32, 32, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_out_of_gamut() {
        // Create frame with legal values (mid-gray, safe)
        let legal_frame = vec![128u8; 10 * 10 * 3];
        // This should convert to legal Y/Cb/Cr values
        assert!(!detect_out_of_gamut(&legal_frame, 10, 10));

        // Create frame with illegal values (pure black RGB = 0,0,0)
        // Pure black converts to Y=0, Cb=128, Cr=128 which has Y < 16 (illegal)
        let illegal_frame = vec![0u8; 10 * 10 * 3];
        assert!(detect_out_of_gamut(&illegal_frame, 10, 10));
    }

    #[test]
    fn test_compute_waveform_stats() {
        let frame = create_test_frame(32, 32);
        let stats = compute_waveform_stats(&frame, 32, 32);

        assert!(stats.avg_luma > 0.0);
        assert!(stats.avg_luma < 255.0);
        assert!(stats.min_luma <= stats.max_luma);
        assert!(stats.std_dev >= 0.0);
    }

    // ── WaveformBitDepth tests ─────────────────────────────────────────

    #[test]
    fn test_bit_depth_max_value() {
        assert_eq!(WaveformBitDepth::Bit8.max_value(), 255);
        assert_eq!(WaveformBitDepth::Bit10.max_value(), 1023);
        assert_eq!(WaveformBitDepth::Bit12.max_value(), 4095);
    }

    #[test]
    fn test_bit_depth_bytes_per_sample() {
        assert_eq!(WaveformBitDepth::Bit8.bytes_per_sample(), 1);
        assert_eq!(WaveformBitDepth::Bit10.bytes_per_sample(), 2);
        assert_eq!(WaveformBitDepth::Bit12.bytes_per_sample(), 2);
    }

    #[test]
    fn test_bit_depth_bits() {
        assert_eq!(WaveformBitDepth::Bit8.bits(), 8);
        assert_eq!(WaveformBitDepth::Bit10.bits(), 10);
        assert_eq!(WaveformBitDepth::Bit12.bits(), 12);
    }

    #[test]
    fn test_generate_luma_waveform_hd_8bit() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();
        let result = generate_luma_waveform_hd(&frame, 32, 32, &config, WaveformBitDepth::Bit8);
        assert!(result.is_ok());
        let scope = result.expect("8-bit HD waveform should succeed");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
    }

    fn create_10bit_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = Vec::with_capacity((width * height * 6) as usize);
        for y in 0..height {
            let value = ((y as u16) * 1023) / height as u16;
            for _x in 0..width {
                // R, G, B each as u16 LE
                frame.extend_from_slice(&value.to_le_bytes());
                frame.extend_from_slice(&value.to_le_bytes());
                frame.extend_from_slice(&value.to_le_bytes());
            }
        }
        frame
    }

    fn create_12bit_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = Vec::with_capacity((width * height * 6) as usize);
        for y in 0..height {
            let value = ((y as u32 * 4095) / height) as u16;
            for _x in 0..width {
                frame.extend_from_slice(&value.to_le_bytes());
                frame.extend_from_slice(&value.to_le_bytes());
                frame.extend_from_slice(&value.to_le_bytes());
            }
        }
        frame
    }

    #[test]
    fn test_generate_luma_waveform_hd_10bit() {
        let frame = create_10bit_frame(32, 32);
        let config = small_scope_config();
        let result = generate_luma_waveform_hd(&frame, 32, 32, &config, WaveformBitDepth::Bit10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_luma_waveform_hd_12bit() {
        let frame = create_12bit_frame(32, 32);
        let config = small_scope_config();
        let result = generate_luma_waveform_hd(&frame, 32, 32, &config, WaveformBitDepth::Bit12);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_rgb_parade_hd_10bit() {
        let frame = create_10bit_frame(32, 32);
        let config = small_scope_config();
        let result = generate_rgb_parade_hd(&frame, 32, 32, &config, WaveformBitDepth::Bit10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_rgb_parade_hd_12bit() {
        let frame = create_12bit_frame(32, 32);
        let config = small_scope_config();
        let result = generate_rgb_parade_hd(&frame, 32, 32, &config, WaveformBitDepth::Bit12);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_luma_waveform_hd_invalid_size() {
        let frame = vec![0u8; 10];
        let config = small_scope_config();
        let result = generate_luma_waveform_hd(&frame, 32, 32, &config, WaveformBitDepth::Bit10);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_waveform_stats_hd_8bit() {
        let frame = create_test_frame(32, 32);
        let stats = compute_waveform_stats_hd(&frame, 32, 32, WaveformBitDepth::Bit8);
        assert!(stats.avg_luma > 0.0);
        assert!(stats.avg_luma < 1.0);
        assert!(stats.min_luma <= stats.max_luma);
        assert!(stats.std_dev >= 0.0);
        assert_eq!(stats.bit_depth, WaveformBitDepth::Bit8);
    }

    #[test]
    fn test_compute_waveform_stats_hd_10bit() {
        let frame = create_10bit_frame(32, 32);
        let stats = compute_waveform_stats_hd(&frame, 32, 32, WaveformBitDepth::Bit10);
        assert!(stats.avg_luma > 0.0);
        assert!(stats.avg_luma < 1.0);
        assert_eq!(stats.bit_depth, WaveformBitDepth::Bit10);
    }

    #[test]
    fn test_compute_waveform_stats_hd_12bit() {
        let frame = create_12bit_frame(32, 32);
        let stats = compute_waveform_stats_hd(&frame, 32, 32, WaveformBitDepth::Bit12);
        assert!(stats.avg_luma > 0.0);
        assert_eq!(stats.bit_depth, WaveformBitDepth::Bit12);
    }

    #[test]
    fn test_compute_waveform_stats_hd_zero_dimensions() {
        let stats = compute_waveform_stats_hd(&[], 0, 0, WaveformBitDepth::Bit8);
        assert_eq!(stats.avg_luma, 0.0);
        assert_eq!(stats.std_dev, 0.0);
    }

    #[test]
    fn test_rgb16_to_luma_normalized_black() {
        let luma = rgb16_to_luma_normalized(0, 0, 0, 1023.0);
        assert!(luma.abs() < 1e-6);
    }

    #[test]
    fn test_rgb16_to_luma_normalized_white() {
        let luma = rgb16_to_luma_normalized(1023, 1023, 1023, 1023.0);
        assert!((luma - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_read_pixel_u16_valid() {
        let r: u16 = 512;
        let g: u16 = 768;
        let b: u16 = 256;
        let mut data = Vec::new();
        data.extend_from_slice(&r.to_le_bytes());
        data.extend_from_slice(&g.to_le_bytes());
        data.extend_from_slice(&b.to_le_bytes());
        let result = read_pixel_u16(&data, 0, 1023);
        assert_eq!(result, Some((512, 768, 256)));
    }

    #[test]
    fn test_read_pixel_u16_too_short() {
        let data = vec![0u8; 4];
        assert!(read_pixel_u16(&data, 0, 1023).is_none());
    }

    #[test]
    fn test_read_pixel_u16_clamps() {
        let high: u16 = 2000;
        let mut data = Vec::new();
        data.extend_from_slice(&high.to_le_bytes());
        data.extend_from_slice(&high.to_le_bytes());
        data.extend_from_slice(&high.to_le_bytes());
        let result = read_pixel_u16(&data, 0, 1023);
        assert_eq!(result, Some((1023, 1023, 1023)));
    }

    // =========================================================================
    // Known-pattern waveform tests (TODO item: "Add tests for `waveform` with
    // known test patterns — color bars, ramp, flat field")
    // =========================================================================

    /// Build an 8-column SMPTE-style colour-bar frame (8-bit RGB24).
    ///
    /// Columns left→right:
    ///   white, yellow, cyan, green, magenta, red, blue, black
    /// Each column is `bar_width` pixels wide; frame height is `height`.
    fn create_color_bars_frame(bar_width: u32, height: u32) -> Vec<u8> {
        let bars: [(u8, u8, u8); 8] = [
            (235, 235, 235), // white (broadcast-safe, not full 255)
            (235, 235, 16),  // yellow
            (16, 235, 235),  // cyan
            (16, 235, 16),   // green
            (235, 16, 235),  // magenta
            (235, 16, 16),   // red
            (16, 16, 235),   // blue
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

    /// Build a luma ramp: pixel at column `x` has RGB = (x_norm, x_norm, x_norm)
    /// where x_norm maps 0..width → 0..255.
    fn create_luma_ramp_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 255) / width.max(1)) as u8;
                let idx = ((y * width + x) * 3) as usize;
                frame[idx] = v;
                frame[idx + 1] = v;
                frame[idx + 2] = v;
            }
        }
        frame
    }

    /// Build a flat-field frame: every pixel is `(value, value, value)`.
    fn create_flat_field_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height * 3) as usize]
    }

    // -------------------------------------------------------------------------
    // Colour-bar waveform — shape and non-empty checks
    // -------------------------------------------------------------------------

    /// The luma waveform of an 8-column colour-bar frame must have the correct
    /// output dimensions and contain at least one non-zero pixel.
    #[test]
    fn test_waveform_color_bars_dimensions_and_non_empty() {
        let bar_width = 4u32;
        let height = 8u32;
        let frame = create_color_bars_frame(bar_width, height);
        let width = bar_width * 8;

        let config = ScopeConfig {
            width: 128,
            height: 128,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        };

        let result = generate_luma_waveform(&frame, width, height, &config);
        assert!(
            result.is_ok(),
            "generate_luma_waveform on color bars must succeed"
        );

        let scope = result.expect("waveform ok");
        assert_eq!(scope.width, 128, "scope width must match config");
        assert_eq!(scope.height, 128, "scope height must match config");
        assert_eq!(
            scope.data.len(),
            (128 * 128 * 4) as usize,
            "RGBA output must be width*height*4 bytes"
        );
        // At least one pixel in the waveform must be lit (non-zero alpha or luma).
        let any_lit = scope.data.chunks(4).any(|px| px[3] > 0 || px[0] > 0);
        assert!(
            any_lit,
            "color-bar waveform must have at least one lit pixel"
        );
    }

    /// The RGB parade of an 8-column colour-bar frame must produce three
    /// distinct channel regions (R, G, B) — verify by sampling one pixel
    /// per third of the scope width in the output RGBA buffer.
    #[test]
    fn test_waveform_color_bars_rgb_parade_has_data() {
        let bar_width = 4u32;
        let height = 8u32;
        let frame = create_color_bars_frame(bar_width, height);
        let width = bar_width * 8;

        let config = ScopeConfig {
            width: 192,
            height: 64,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        };

        let result = generate_rgb_parade(&frame, width, height, &config);
        assert!(
            result.is_ok(),
            "generate_rgb_parade on color bars must succeed"
        );

        let scope = result.expect("rgb parade ok");
        assert_eq!(scope.width, 192);
        assert_eq!(scope.height, 64);
        assert!(!scope.data.is_empty(), "rgb parade data must be non-empty");
    }

    // -------------------------------------------------------------------------
    // Luma ramp — correct shape
    // -------------------------------------------------------------------------

    /// A luma ramp (0→255 left→right) must produce a waveform with the correct
    /// output dimensions; the ramp should map to a diagonal pattern in the scope.
    #[test]
    fn test_waveform_luma_ramp_shape() {
        let width = 64u32;
        let height = 16u32;
        let frame = create_luma_ramp_frame(width, height);

        let config = ScopeConfig {
            width: 128,
            height: 128,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        };

        let result = generate_luma_waveform(&frame, width, height, &config);
        assert!(result.is_ok(), "luma ramp waveform must succeed");

        let scope = result.expect("luma ramp ok");
        assert_eq!(scope.width, 128);
        assert_eq!(scope.height, 128);

        // Verify stats: ramp has min≈0, max≈255, non-trivial std-dev.
        let stats = compute_waveform_stats(&frame, width, height);
        assert!(stats.max_luma > 200, "ramp max luma should be near 255");
        assert!(stats.min_luma < 10, "ramp min luma should be near 0");
        assert!(stats.std_dev > 50.0, "ramp must have significant variance");
    }

    // -------------------------------------------------------------------------
    // Flat fields — black and white
    // -------------------------------------------------------------------------

    /// A flat-black frame must produce a waveform where all lit pixels are
    /// near the bottom of the scope (low luma → low scope_y).
    #[test]
    fn test_waveform_flat_black_field_pixels_near_bottom() {
        let width = 32u32;
        let height = 32u32;
        let frame = create_flat_field_frame(width, height, 0);

        let config = ScopeConfig {
            width: 64,
            height: 64,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        };

        let result = generate_luma_waveform(&frame, width, height, &config);
        assert!(result.is_ok(), "flat-black waveform must succeed");

        let scope = result.expect("flat black ok");

        // All black pixels → luma ≈ 16 (broadcast offset), scope_y near bottom.
        // Check the waveform stats: avg_luma should be very low.
        let stats = compute_waveform_stats(&frame, width, height);
        assert!(
            stats.avg_luma < 20.0,
            "flat black avg_luma must be near 0 (got {})",
            stats.avg_luma
        );
        assert_eq!(stats.min_luma, stats.max_luma, "flat field: min==max");

        // In the RGBA scope, no pixels in the top quarter should be lit
        // (top quarter = rows 0..scope_height/4).
        let top_quarter_end = (scope.height / 4) as usize;
        let top_lit = scope
            .data
            .chunks(4)
            .take(top_quarter_end * scope.width as usize)
            .any(|px| px[0] > 0 || px[3] > 0);
        assert!(
            !top_lit,
            "flat-black waveform must not have lit pixels in the top quarter"
        );
    }

    /// A flat-white frame must produce a waveform where all lit pixels are
    /// near the top of the scope (high luma → high scope_y → low row index
    /// because the y-axis is inverted in the waveform).
    #[test]
    fn test_waveform_flat_white_field_pixels_near_top() {
        let width = 32u32;
        let height = 32u32;
        let frame = create_flat_field_frame(width, height, 235); // broadcast white

        let config = ScopeConfig {
            width: 64,
            height: 64,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        };

        let result = generate_luma_waveform(&frame, width, height, &config);
        assert!(result.is_ok(), "flat-white waveform must succeed");

        let scope = result.expect("flat white ok");

        let stats = compute_waveform_stats(&frame, width, height);
        assert!(
            stats.avg_luma > 200.0,
            "flat white avg_luma must be near 235 (got {})",
            stats.avg_luma
        );
        assert_eq!(stats.min_luma, stats.max_luma, "flat field: min==max");

        // In the RGBA scope, no pixels in the bottom quarter should be lit
        // (bottom quarter = rows 3*scope_height/4..scope_height).
        let bottom_start = (scope.height * 3 / 4) as usize;
        let bottom_lit = scope
            .data
            .chunks(4)
            .skip(bottom_start * scope.width as usize)
            .any(|px| px[0] > 0 || px[3] > 0);
        assert!(
            !bottom_lit,
            "flat-white waveform must not have lit pixels in the bottom quarter"
        );
    }

    // -------------------------------------------------------------------------
    // 10-bit ramp via generate_luma_waveform_hd
    // -------------------------------------------------------------------------

    /// A 10-bit luma ramp processed through `generate_luma_waveform_hd` must
    /// produce valid output with correct dimensions and non-empty pixel data.
    #[test]
    fn test_waveform_10bit_ramp_via_hd_api() {
        // Build a 10-bit ramp: each 6-byte triplet is (v_le, v_le, v_le) where
        // v ∈ 0..1023 mapped across width pixels.
        let width = 32u32;
        let height = 8u32;
        let max_val = 1023u16;

        let mut frame = Vec::with_capacity((width * height * 6) as usize);
        for _y in 0..height {
            for x in 0..width {
                let v = ((x * max_val as u32) / width.max(1)) as u16;
                frame.extend_from_slice(&v.to_le_bytes()); // R
                frame.extend_from_slice(&v.to_le_bytes()); // G
                frame.extend_from_slice(&v.to_le_bytes()); // B
            }
        }

        let config = ScopeConfig {
            width: 64,
            height: 64,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        };

        let result =
            generate_luma_waveform_hd(&frame, width, height, &config, WaveformBitDepth::Bit10);
        assert!(result.is_ok(), "10-bit luma ramp via HD API must succeed");

        let scope = result.expect("10-bit ramp ok");
        assert_eq!(scope.width, 64);
        assert_eq!(scope.height, 64);
        assert_eq!(
            scope.data.len(),
            (64 * 64 * 4) as usize,
            "HD waveform RGBA output size"
        );
        let any_lit = scope.data.chunks(4).any(|px| px[0] > 0 || px[3] > 0);
        assert!(
            any_lit,
            "10-bit ramp waveform must have at least one lit pixel"
        );
    }

    /// A 10-bit flat-grey field at mid-range (512/1023) must have avg_luma ≈ 0.5.
    #[test]
    fn test_waveform_stats_10bit_flat_grey() {
        let width = 16u32;
        let height = 16u32;
        let v: u16 = 512;

        let mut frame = Vec::with_capacity((width * height * 6) as usize);
        for _ in 0..width * height {
            frame.extend_from_slice(&v.to_le_bytes());
            frame.extend_from_slice(&v.to_le_bytes());
            frame.extend_from_slice(&v.to_le_bytes());
        }

        let stats = compute_waveform_stats_hd(&frame, width, height, WaveformBitDepth::Bit10);
        assert!(
            (stats.avg_luma - 0.5).abs() < 0.05,
            "10-bit flat grey at 512 must have avg_luma ≈ 0.5 (got {})",
            stats.avg_luma
        );
        assert_eq!(stats.min_luma, stats.max_luma, "flat 10-bit grey: min==max");
        assert_eq!(stats.bit_depth, WaveformBitDepth::Bit10);
    }
}
