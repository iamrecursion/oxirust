//! HDR (High Dynamic Range) support for video scopes.
//!
//! This module provides HDR-specific features for analyzing and monitoring
//! high dynamic range video content:
//! - **PQ (Perceptual Quantizer)**: SMPTE ST 2084 transfer function
//! - **HLG (Hybrid Log-Gamma)**: ITU-R BT.2100 HLG transfer function
//! - **Nits scale**: Luminance measurement in candelas per square meter
//! - **HDR waveform**: Waveform with HDR-specific scales and overlays
//! - **MaxCLL/MaxFALL**: Maximum content light level and frame average light level
//!
//! HDR monitoring is essential for ensuring proper exposure and avoiding clipping
//! in HDR production workflows.

use crate::render::{rgb_to_ycbcr, Canvas};
use crate::{ScopeData, ScopeType};
use oximedia_core::OxiResult;

/// HDR transfer function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrTransferFunction {
    /// PQ (Perceptual Quantizer) - SMPTE ST 2084.
    Pq,

    /// HLG (Hybrid Log-Gamma) - ITU-R BT.2100.
    Hlg,

    /// Linear (no transfer function).
    Linear,
}

/// HDR measurement scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrScale {
    /// Nits (candelas per square meter).
    Nits,

    /// Stops (relative to middle gray).
    Stops,

    /// Percentage (0-100%).
    Percent,
}

/// HDR waveform configuration.
#[derive(Debug, Clone)]
pub struct HdrWaveformConfig {
    /// Transfer function.
    pub transfer_function: HdrTransferFunction,

    /// Measurement scale.
    pub scale: HdrScale,

    /// Maximum nits for PQ (typically 10,000).
    pub max_nits: f32,

    /// Reference white in nits (typically 100-203).
    pub reference_white: f32,

    /// Show clipping warnings.
    pub show_clipping: bool,
}

impl Default for HdrWaveformConfig {
    fn default() -> Self {
        Self {
            transfer_function: HdrTransferFunction::Pq,
            scale: HdrScale::Nits,
            max_nits: 10000.0,
            reference_white: 100.0,
            show_clipping: true,
        }
    }
}

/// Converts linear light (0-1) to PQ (Perceptual Quantizer) code value.
///
/// PQ is defined in SMPTE ST 2084 and maps 0-10,000 nits to 0-1 range.
#[must_use]
pub fn linear_to_pq(linear: f32) -> f32 {
    // PQ constants (SMPTE ST 2084)
    const M1: f32 = 2610.0 / 16384.0; // 0.1593017578125
    const M2: f32 = 2523.0 / 4096.0 * 128.0; // 78.84375
    const C1: f32 = 3424.0 / 4096.0; // 0.8359375
    const C2: f32 = 2413.0 / 4096.0 * 32.0; // 18.8515625
    const C3: f32 = 2392.0 / 4096.0 * 32.0; // 18.6875

    let y = linear.max(0.0);
    let y_m1 = y.powf(M1);
    let numerator = C1 + C2 * y_m1;
    let denominator = 1.0 + C3 * y_m1;

    (numerator / denominator).powf(M2)
}

/// Converts PQ code value to linear light (0-1).
#[must_use]
pub fn pq_to_linear(pq: f32) -> f32 {
    // PQ constants (SMPTE ST 2084)
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;

    let pq = pq.clamp(0.0, 1.0);
    let pq_m2 = pq.powf(1.0 / M2);
    let numerator = (pq_m2 - C1).max(0.0);
    let denominator = C2 - C3 * pq_m2;

    if denominator == 0.0 {
        return 0.0;
    }

    (numerator / denominator).powf(1.0 / M1)
}

/// Converts linear light (0-1) to HLG (Hybrid Log-Gamma) signal.
///
/// HLG is defined in ITU-R BT.2100 and uses a hybrid gamma/logarithmic curve.
#[must_use]
pub fn linear_to_hlg(linear: f32) -> f32 {
    const A: f32 = 0.17883277;
    const B: f32 = 0.28466892;
    const C: f32 = 0.559_910_7;

    let linear = linear.max(0.0);

    if linear <= 1.0 / 12.0 {
        (3.0 * linear).sqrt()
    } else {
        A * (12.0 * linear - B).ln() + C
    }
}

/// Converts HLG signal to linear light (0-1).
#[must_use]
pub fn hlg_to_linear(hlg: f32) -> f32 {
    const A: f32 = 0.17883277;
    const B: f32 = 0.28466892;
    const C: f32 = 0.559_910_7;

    let hlg = hlg.clamp(0.0, 1.0);

    if hlg <= 0.5 {
        (hlg * hlg) / 3.0
    } else {
        ((hlg - C) / A).exp() / 12.0 + B / 12.0
    }
}

/// Converts linear light value to nits for PQ.
#[must_use]
pub fn linear_to_nits_pq(linear: f32, max_nits: f32) -> f32 {
    linear * max_nits
}

/// Converts nits to linear light value for PQ.
#[must_use]
pub fn nits_to_linear_pq(nits: f32, max_nits: f32) -> f32 {
    nits / max_nits
}

/// Converts linear light value to nits for HLG (scene-referred).
#[must_use]
pub fn linear_to_nits_hlg(linear: f32, reference_white: f32) -> f32 {
    // HLG is scene-referred, so we need to apply system gamma
    const SYSTEM_GAMMA: f32 = 1.2;
    linear.powf(1.0 / SYSTEM_GAMMA) * reference_white
}

/// Generates an HDR waveform with proper scale and transfer function.
///
/// # Arguments
///
/// * `frame` - RGB24 frame data (width * height * 3 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - HDR waveform configuration
///
/// # Errors
///
/// Returns an error if frame data is invalid or insufficient.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::too_many_lines)]
pub fn generate_hdr_waveform(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &HdrWaveformConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = 512u32;
    let scope_height = 512u32;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Create accumulation buffer
    let mut accumulator = vec![0u32; (scope_width * scope_height) as usize];

    // Process frame
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            // Convert to linear light (assuming sRGB input)
            let (luma, _, _) = rgb_to_ycbcr(r, g, b);
            let linear_luma = srgb_to_linear(f32::from(luma) / 255.0);

            // Apply HDR transfer function
            let hdr_value = match config.transfer_function {
                HdrTransferFunction::Pq => {
                    let nits = linear_to_nits_pq(linear_luma, config.max_nits);
                    let linear_normalized = nits_to_linear_pq(nits, config.max_nits);
                    linear_to_pq(linear_normalized)
                }
                HdrTransferFunction::Hlg => linear_to_hlg(linear_luma),
                HdrTransferFunction::Linear => linear_luma,
            };

            // Map to scope coordinates based on scale
            let scope_value = match config.scale {
                HdrScale::Nits => {
                    let nits = match config.transfer_function {
                        HdrTransferFunction::Pq => {
                            linear_to_nits_pq(pq_to_linear(hdr_value), config.max_nits)
                        }
                        HdrTransferFunction::Hlg => {
                            linear_to_nits_hlg(hlg_to_linear(hdr_value), config.reference_white)
                        }
                        HdrTransferFunction::Linear => linear_luma * config.reference_white,
                    };
                    nits / config.max_nits // Normalize to 0-1
                }
                HdrScale::Stops => {
                    // Calculate stops relative to reference white
                    let nits = match config.transfer_function {
                        HdrTransferFunction::Pq => {
                            linear_to_nits_pq(pq_to_linear(hdr_value), config.max_nits)
                        }
                        HdrTransferFunction::Hlg => {
                            linear_to_nits_hlg(hlg_to_linear(hdr_value), config.reference_white)
                        }
                        HdrTransferFunction::Linear => linear_luma * config.reference_white,
                    };
                    let stops = (nits / config.reference_white).log2();
                    (stops + 6.0) / 12.0 // Map -6 to +6 stops to 0-1
                }
                HdrScale::Percent => hdr_value, // Already 0-1
            };

            let scope_x = (x * scope_width) / width;
            let scope_y = scope_height
                - 1
                - ((scope_value.clamp(0.0, 1.0) * scope_height as f32) as u32)
                    .min(scope_height - 1);

            let idx = (scope_y * scope_width + scope_x) as usize;
            if idx < accumulator.len() {
                accumulator[idx] = accumulator[idx].saturating_add(1);
            }
        }
    }

    // Find max value for normalization
    let max_val = accumulator.iter().copied().max().unwrap_or(1);

    // Draw waveform
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

    // Draw HDR-specific graticule
    draw_hdr_graticule(&mut canvas, config);

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::HdrWaveform,
    })
}

/// Draws HDR-specific graticule with nits/stops scale.
fn draw_hdr_graticule(canvas: &mut Canvas, config: &HdrWaveformConfig) {
    let width = canvas.width;
    let height = canvas.height;
    let color = [255, 255, 255, 128];

    // Draw horizontal lines based on scale
    match config.scale {
        HdrScale::Nits => {
            // Mark key nit levels
            let nit_levels = if config.max_nits >= 10000.0 {
                vec![100.0, 500.0, 1000.0, 4000.0, 10000.0]
            } else if config.max_nits >= 1000.0 {
                vec![100.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
            } else {
                vec![50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
            };

            for nits in &nit_levels {
                if *nits <= config.max_nits {
                    let y = height - ((nits / config.max_nits * height as f32) as u32);
                    if y < height {
                        canvas.draw_hline(0, width - 1, y, color);
                    }
                }
            }
        }
        HdrScale::Stops => {
            // Mark stop levels relative to reference white
            for stops in -6..=6 {
                let normalized = (stops as f32 + 6.0) / 12.0;
                let y = height - ((normalized * height as f32) as u32);
                if y < height {
                    canvas.draw_hline(0, width - 1, y, color);
                }
            }
        }
        HdrScale::Percent => {
            // Mark percentage levels
            for percent in &[0, 25, 50, 75, 100] {
                let y = height - ((*percent as f32 / 100.0 * height as f32) as u32);
                if y < height {
                    canvas.draw_hline(0, width - 1, y, color);
                }
            }
        }
    }

    // Draw vertical lines
    for i in 1..4 {
        let x = (width * i) / 4;
        canvas.draw_vline(x, 0, height - 1, color);
    }
}

/// Converts sRGB to linear light.
#[must_use]
fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// HDR metadata for content light levels.
#[derive(Debug, Clone)]
pub struct HdrMetadata {
    /// Maximum Content Light Level (nits).
    pub max_cll: f32,

    /// Maximum Frame-Average Light Level (nits).
    pub max_fall: f32,

    /// Average light level across all frames (nits).
    pub avg_light_level: f32,

    /// Percentage of pixels exceeding reference white.
    pub over_reference_percent: f32,

    /// Percentage of pixels at or near peak (> 90% of max).
    pub peak_clip_percent: f32,
}

/// Computes HDR metadata from frame data.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_hdr_metadata(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &HdrWaveformConfig,
) -> HdrMetadata {
    let pixel_count = width * height;
    let mut max_nits = 0.0f32;
    let mut total_nits = 0.0f32;
    let mut over_reference_count = 0u32;
    let mut peak_count = 0u32;

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
            let linear_luma = srgb_to_linear(f32::from(luma) / 255.0);

            let nits = match config.transfer_function {
                HdrTransferFunction::Pq => {
                    let linear_normalized = nits_to_linear_pq(linear_luma, 1.0);
                    let pq_value = linear_to_pq(linear_normalized);
                    linear_to_nits_pq(pq_to_linear(pq_value), config.max_nits)
                }
                HdrTransferFunction::Hlg => linear_to_nits_hlg(linear_luma, config.reference_white),
                HdrTransferFunction::Linear => linear_luma * config.reference_white,
            };

            max_nits = max_nits.max(nits);
            total_nits += nits;

            if nits > config.reference_white {
                over_reference_count += 1;
            }

            if nits > config.max_nits * 0.9 {
                peak_count += 1;
            }
        }
    }

    let avg_light_level = total_nits / pixel_count as f32;
    let over_reference_percent = (over_reference_count as f32 / pixel_count as f32) * 100.0;
    let peak_clip_percent = (peak_count as f32 / pixel_count as f32) * 100.0;

    HdrMetadata {
        max_cll: max_nits,
        max_fall: avg_light_level,
        avg_light_level,
        over_reference_percent,
        peak_clip_percent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_pq_roundtrip() {
        let linear = 0.5;
        let pq = linear_to_pq(linear);
        let linear_back = pq_to_linear(pq);

        assert!((linear - linear_back).abs() < 0.001);
    }

    #[test]
    fn test_linear_to_hlg_roundtrip() {
        let linear = 0.5;
        let hlg = linear_to_hlg(linear);
        let linear_back = hlg_to_linear(hlg);

        assert!((linear - linear_back).abs() < 0.01);
    }

    #[test]
    fn test_pq_zero_and_one() {
        let pq_zero = linear_to_pq(0.0);
        assert!(pq_zero.abs() < 1e-6);

        let pq_one = linear_to_pq(1.0);
        assert!(pq_one > 0.9 && pq_one <= 1.0);
    }

    #[test]
    fn test_hlg_zero_and_one() {
        let hlg_zero = linear_to_hlg(0.0);
        assert_eq!(hlg_zero, 0.0);

        let hlg_one = linear_to_hlg(1.0);
        assert!(hlg_one > 0.9 && hlg_one <= 1.0);
    }

    #[test]
    fn test_nits_conversion_pq() {
        let linear = 0.5;
        let nits = linear_to_nits_pq(linear, 10000.0);
        assert_eq!(nits, 5000.0);

        let linear_back = nits_to_linear_pq(nits, 10000.0);
        assert_eq!(linear_back, 0.5);
    }

    #[test]
    fn test_generate_hdr_waveform() {
        let mut frame = vec![0u8; 100 * 100 * 3];

        // Create gradient
        for y in 0..100 {
            let value = ((y * 255) / 100) as u8;
            for x in 0..100 {
                let idx = ((y * 100 + x) * 3) as usize;
                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        let config = HdrWaveformConfig::default();
        let result = generate_hdr_waveform(&frame, 100, 100, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hdr_transfer_functions() {
        let frame = vec![128u8; 50 * 50 * 3];

        // Test PQ
        let config_pq = HdrWaveformConfig {
            transfer_function: HdrTransferFunction::Pq,
            ..Default::default()
        };
        let result = generate_hdr_waveform(&frame, 50, 50, &config_pq);
        assert!(result.is_ok());

        // Test HLG
        let config_hlg = HdrWaveformConfig {
            transfer_function: HdrTransferFunction::Hlg,
            ..Default::default()
        };
        let result = generate_hdr_waveform(&frame, 50, 50, &config_hlg);
        assert!(result.is_ok());

        // Test Linear
        let config_linear = HdrWaveformConfig {
            transfer_function: HdrTransferFunction::Linear,
            ..Default::default()
        };
        let result = generate_hdr_waveform(&frame, 50, 50, &config_linear);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hdr_scales() {
        let frame = vec![128u8; 50 * 50 * 3];

        // Test Nits scale
        let config_nits = HdrWaveformConfig {
            scale: HdrScale::Nits,
            ..Default::default()
        };
        let result = generate_hdr_waveform(&frame, 50, 50, &config_nits);
        assert!(result.is_ok());

        // Test Stops scale
        let config_stops = HdrWaveformConfig {
            scale: HdrScale::Stops,
            ..Default::default()
        };
        let result = generate_hdr_waveform(&frame, 50, 50, &config_stops);
        assert!(result.is_ok());

        // Test Percent scale
        let config_percent = HdrWaveformConfig {
            scale: HdrScale::Percent,
            ..Default::default()
        };
        let result = generate_hdr_waveform(&frame, 50, 50, &config_percent);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_hdr_metadata() {
        let mut frame = vec![0u8; 100 * 100 * 3];

        // Create gradient
        for y in 0..100 {
            let value = ((y * 255) / 100) as u8;
            for x in 0..100 {
                let idx = ((y * 100 + x) * 3) as usize;
                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        let config = HdrWaveformConfig::default();
        let metadata = compute_hdr_metadata(&frame, 100, 100, &config);

        assert!(metadata.max_cll > 0.0);
        assert!(metadata.max_fall > 0.0);
        assert!(metadata.avg_light_level > 0.0);
        assert!(metadata.over_reference_percent >= 0.0);
        assert!(metadata.peak_clip_percent >= 0.0);
    }

    #[test]
    fn test_srgb_to_linear() {
        let linear = srgb_to_linear(0.5);
        assert!(linear > 0.0 && linear < 1.0);

        let linear_zero = srgb_to_linear(0.0);
        assert_eq!(linear_zero, 0.0);

        let linear_one = srgb_to_linear(1.0);
        assert_eq!(linear_one, 1.0);
    }

    #[test]
    fn test_invalid_frame_size() {
        let frame = vec![0u8; 100]; // Too small
        let config = HdrWaveformConfig::default();

        let result = generate_hdr_waveform(&frame, 100, 100, &config);
        assert!(result.is_err());
    }
}
