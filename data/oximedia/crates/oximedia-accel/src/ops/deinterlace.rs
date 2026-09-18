//! Deinterlacing operations for interlaced video content.
//!
//! Provides three deinterlacing methods:
//!
//! - **Bob**: Interpolates missing lines from the selected field by averaging
//!   adjacent field lines. Doubles frame rate, no temporal artifacts.
//!
//! - **Weave**: Combines top and bottom fields directly. Preserves full
//!   vertical resolution but produces combing artifacts on motion.
//!
//! - **Motion-adaptive**: Blends bob and weave based on per-pixel motion
//!   detection. Uses weave where the image is static for sharpness, and
//!   bob where motion is detected to reduce combing.

use crate::error::{AccelError, AccelResult};
use rayon::prelude::*;

/// Deinterlacing method to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMethod {
    /// Bob deinterlacing: doubles the selected field vertically.
    /// Fast and free of combing, but loses half the vertical resolution.
    Bob,
    /// Weave deinterlacing: interleaves top and bottom fields.
    /// Full resolution but combing artifacts on motion.
    Weave,
    /// Motion-adaptive deinterlacing: blends bob and weave per-pixel
    /// based on inter-field motion detection.
    MotionAdaptive,
}

/// Which field to treat as the "current" field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// Top field first (even rows = top field).
    TopFieldFirst,
    /// Bottom field first (odd rows = top field).
    BottomFieldFirst,
}

/// Configuration for the deinterlace operation.
#[derive(Debug, Clone)]
pub struct DeinterlaceConfig {
    /// Deinterlacing method.
    pub method: DeinterlaceMethod,
    /// Field order of the input.
    pub field_order: FieldOrder,
    /// Motion threshold for motion-adaptive mode (0..255).
    /// Pixels with inter-field difference above this threshold are treated
    /// as moving and use bob; below use weave.
    pub motion_threshold: u8,
}

impl Default for DeinterlaceConfig {
    fn default() -> Self {
        Self {
            method: DeinterlaceMethod::MotionAdaptive,
            field_order: FieldOrder::TopFieldFirst,
            motion_threshold: 20,
        }
    }
}

/// Performs deinterlacing on a single grayscale or per-channel frame.
///
/// `input` must contain `width * height * channels` bytes in row-major order.
///
/// # Errors
///
/// Returns an error if the input buffer size does not match
/// `width * height * channels`.
pub fn deinterlace(
    input: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    config: &DeinterlaceConfig,
) -> AccelResult<Vec<u8>> {
    let expected = (width * height * channels) as usize;
    if input.len() != expected {
        return Err(AccelError::BufferSizeMismatch {
            expected,
            actual: input.len(),
        });
    }
    if height < 2 {
        return Ok(input.to_vec());
    }

    match config.method {
        DeinterlaceMethod::Bob => deinterlace_bob(input, width, height, channels, config),
        DeinterlaceMethod::Weave => Ok(input.to_vec()), // weave is identity for single-frame
        DeinterlaceMethod::MotionAdaptive => {
            deinterlace_motion_adaptive(input, width, height, channels, config)
        }
    }
}

/// Performs deinterlacing on interlaced content given two fields.
///
/// `top_field` contains even rows; `bottom_field` contains odd rows.
/// Each field has `width * (height/2) * channels` bytes.
///
/// # Errors
///
/// Returns an error if field sizes do not match expectations.
pub fn deinterlace_fields(
    top_field: &[u8],
    bottom_field: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    config: &DeinterlaceConfig,
) -> AccelResult<Vec<u8>> {
    let field_height = height / 2;
    let field_expected = (width * field_height * channels) as usize;

    if top_field.len() != field_expected {
        return Err(AccelError::BufferSizeMismatch {
            expected: field_expected,
            actual: top_field.len(),
        });
    }
    if bottom_field.len() != field_expected {
        return Err(AccelError::BufferSizeMismatch {
            expected: field_expected,
            actual: bottom_field.len(),
        });
    }

    let stride = (width * channels) as usize;
    let mut frame = vec![0u8; (width * height * channels) as usize];

    // Interleave fields into a full frame
    for y in 0..field_height as usize {
        let top_row_start = y * stride;
        let bottom_row_start = y * stride;
        let even_row_start = (y * 2) * stride;
        let odd_row_start = (y * 2 + 1) * stride;

        frame[even_row_start..even_row_start + stride]
            .copy_from_slice(&top_field[top_row_start..top_row_start + stride]);
        frame[odd_row_start..odd_row_start + stride]
            .copy_from_slice(&bottom_field[bottom_row_start..bottom_row_start + stride]);
    }

    match config.method {
        DeinterlaceMethod::Weave => Ok(frame),
        DeinterlaceMethod::Bob => deinterlace_bob(&frame, width, height, channels, config),
        DeinterlaceMethod::MotionAdaptive => {
            deinterlace_motion_adaptive(&frame, width, height, channels, config)
        }
    }
}

/// Bob deinterlacing: interpolates missing lines from neighboring field lines.
fn deinterlace_bob(
    input: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    config: &DeinterlaceConfig,
) -> AccelResult<Vec<u8>> {
    let stride = (width * channels) as usize;
    let mut output = input.to_vec();

    // Determine which lines belong to the "other" field (to be interpolated)
    let interpolate_even = matches!(config.field_order, FieldOrder::BottomFieldFirst);

    output
        .par_chunks_exact_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            let is_even = y % 2 == 0;
            let should_interpolate =
                (is_even && interpolate_even) || (!is_even && !interpolate_even);

            if should_interpolate {
                // Interpolate this row from its neighbors
                let above = if y > 0 { y - 1 } else { y + 1 };
                let below = if y + 1 < height as usize {
                    y + 1
                } else if y > 0 {
                    y - 1
                } else {
                    y
                };

                for x in 0..stride {
                    let a = f32::from(input[above * stride + x]);
                    let b = f32::from(input[below * stride + x]);
                    row[x] = ((a + b) * 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        });

    Ok(output)
}

/// Motion-adaptive deinterlacing: blends bob and weave per-pixel.
///
/// For each pixel in the interpolated field lines, computes the absolute
/// difference between the lines above and below (inter-field motion).
/// If motion exceeds the threshold, uses the bob (interpolated) value;
/// otherwise keeps the weave (original) value for maximum sharpness.
fn deinterlace_motion_adaptive(
    input: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    config: &DeinterlaceConfig,
) -> AccelResult<Vec<u8>> {
    let stride = (width * channels) as usize;
    let bob = deinterlace_bob(input, width, height, channels, config)?;
    let mut output = input.to_vec();
    let threshold = f32::from(config.motion_threshold);

    let interpolate_even = matches!(config.field_order, FieldOrder::BottomFieldFirst);

    output
        .par_chunks_exact_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            let is_even = y % 2 == 0;
            let should_interpolate =
                (is_even && interpolate_even) || (!is_even && !interpolate_even);

            if should_interpolate {
                let above = if y > 0 { y - 1 } else { y + 1 };
                let below = if y + 1 < height as usize {
                    y + 1
                } else if y > 0 {
                    y - 1
                } else {
                    y
                };

                for x in 0..stride {
                    let a = f32::from(input[above * stride + x]);
                    let b = f32::from(input[below * stride + x]);
                    let motion = (a - b).abs();

                    if motion > threshold {
                        // High motion: use bob (interpolated) value
                        row[x] = bob[y * stride + x];
                    }
                    // else: keep weave (original) value for sharpness
                }
            }
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_interlaced_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height) as usize];
        for y in 0..height {
            let val = if y % 2 == 0 { 200u8 } else { 50u8 };
            for x in 0..width {
                frame[(y * width + x) as usize] = val;
            }
        }
        frame
    }

    #[test]
    fn test_bob_deinterlace_basic() {
        let width = 8u32;
        let height = 4u32;
        let frame = make_interlaced_frame(width, height);
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::Bob,
            field_order: FieldOrder::TopFieldFirst,
            motion_threshold: 20,
        };
        let result =
            deinterlace(&frame, width, height, 1, &config).expect("deinterlace should succeed");
        assert_eq!(result.len(), (width * height) as usize);
        // Even rows (top field) should be unchanged
        assert_eq!(result[0], 200);
        assert_eq!(result[(2 * width) as usize], 200);
        // Odd rows (interpolated) should be average of neighbors
        // Row 1: avg(200, 200) = 200
        assert_eq!(result[width as usize], 200);
    }

    #[test]
    fn test_weave_deinterlace_identity() {
        let width = 4u32;
        let height = 4u32;
        let frame = make_interlaced_frame(width, height);
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::Weave,
            ..Default::default()
        };
        let result =
            deinterlace(&frame, width, height, 1, &config).expect("deinterlace should succeed");
        assert_eq!(result, frame);
    }

    #[test]
    fn test_motion_adaptive_deinterlace() {
        let width = 8u32;
        let height = 4u32;
        let frame = make_interlaced_frame(width, height);
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::MotionAdaptive,
            field_order: FieldOrder::TopFieldFirst,
            motion_threshold: 10,
        };
        let result =
            deinterlace(&frame, width, height, 1, &config).expect("deinterlace should succeed");
        assert_eq!(result.len(), (width * height) as usize);
        // Even rows should remain unchanged
        assert_eq!(result[0], 200);
    }

    #[test]
    fn test_deinterlace_buffer_mismatch() {
        let config = DeinterlaceConfig::default();
        let result = deinterlace(&[0u8; 10], 4, 4, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_deinterlace_tiny_height() {
        let width = 4u32;
        let height = 1u32;
        let frame = vec![128u8; (width * height) as usize];
        let config = DeinterlaceConfig::default();
        let result =
            deinterlace(&frame, width, height, 1, &config).expect("deinterlace should succeed");
        assert_eq!(result, frame);
    }

    #[test]
    fn test_bob_multichannel() {
        let width = 4u32;
        let height = 4u32;
        let channels = 3u32;
        let mut frame = vec![0u8; (width * height * channels) as usize];
        for y in 0..height {
            let val = if y % 2 == 0 { 240u8 } else { 16u8 };
            for x in 0..width {
                for c in 0..channels {
                    frame[((y * width + x) * channels + c) as usize] = val;
                }
            }
        }
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::Bob,
            field_order: FieldOrder::TopFieldFirst,
            motion_threshold: 20,
        };
        let result = deinterlace(&frame, width, height, channels, &config)
            .expect("deinterlace should succeed");
        assert_eq!(result.len(), (width * height * channels) as usize);
        // Even row pixels unchanged
        assert_eq!(result[0], 240);
        assert_eq!(result[1], 240);
        assert_eq!(result[2], 240);
    }

    #[test]
    fn test_bottom_field_first_bob() {
        let width = 4u32;
        let height = 4u32;
        let frame = make_interlaced_frame(width, height);
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::Bob,
            field_order: FieldOrder::BottomFieldFirst,
            motion_threshold: 20,
        };
        let result =
            deinterlace(&frame, width, height, 1, &config).expect("deinterlace should succeed");
        // In BFF mode, even rows are interpolated, odd rows kept
        assert_eq!(result[width as usize], 50); // odd row unchanged
    }

    #[test]
    fn test_deinterlace_fields_weave() {
        let width = 4u32;
        let height = 4u32;
        let field_height = height / 2;
        let top = vec![200u8; (width * field_height) as usize];
        let bottom = vec![50u8; (width * field_height) as usize];
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::Weave,
            ..Default::default()
        };
        let result = deinterlace_fields(&top, &bottom, width, height, 1, &config)
            .expect("deinterlace_fields should succeed");
        assert_eq!(result.len(), (width * height) as usize);
        // Even rows from top field
        assert_eq!(result[0], 200);
        // Odd rows from bottom field
        assert_eq!(result[width as usize], 50);
    }

    #[test]
    fn test_deinterlace_fields_bob() {
        let width = 4u32;
        let height = 4u32;
        let field_height = height / 2;
        let top = vec![200u8; (width * field_height) as usize];
        let bottom = vec![50u8; (width * field_height) as usize];
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::Bob,
            field_order: FieldOrder::TopFieldFirst,
            motion_threshold: 20,
        };
        let result = deinterlace_fields(&top, &bottom, width, height, 1, &config)
            .expect("deinterlace_fields should succeed");
        assert_eq!(result.len(), (width * height) as usize);
    }

    #[test]
    fn test_deinterlace_fields_size_mismatch() {
        let config = DeinterlaceConfig::default();
        let result = deinterlace_fields(&[0u8; 10], &[0u8; 8], 4, 4, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_motion_adaptive_static_content() {
        // Uniform frame: no motion detected, should use weave (identity)
        let width = 8u32;
        let height = 4u32;
        let frame = vec![128u8; (width * height) as usize];
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::MotionAdaptive,
            field_order: FieldOrder::TopFieldFirst,
            motion_threshold: 10,
        };
        let result =
            deinterlace(&frame, width, height, 1, &config).expect("deinterlace should succeed");
        // Static content: all values should remain 128
        assert!(result.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_deinterlace_config_default() {
        let config = DeinterlaceConfig::default();
        assert_eq!(config.method, DeinterlaceMethod::MotionAdaptive);
        assert_eq!(config.field_order, FieldOrder::TopFieldFirst);
        assert_eq!(config.motion_threshold, 20);
    }

    #[test]
    fn test_bob_preserves_field_lines() {
        let width = 4u32;
        let height = 6u32;
        let mut frame = vec![0u8; (width * height) as usize];
        // Set each row to a distinct value
        for y in 0..height {
            for x in 0..width {
                frame[(y * width + x) as usize] = (y * 40) as u8;
            }
        }
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::Bob,
            field_order: FieldOrder::TopFieldFirst,
            motion_threshold: 20,
        };
        let result =
            deinterlace(&frame, width, height, 1, &config).expect("deinterlace should succeed");
        // Even rows (field lines) should be preserved
        assert_eq!(result[0], 0);
        assert_eq!(result[(2 * width) as usize], 80);
        assert_eq!(result[(4 * width) as usize], 160);
    }

    #[test]
    fn test_large_frame_deinterlace() {
        let width = 1920u32;
        let height = 1080u32;
        let frame = vec![100u8; (width * height) as usize];
        let config = DeinterlaceConfig {
            method: DeinterlaceMethod::Bob,
            field_order: FieldOrder::TopFieldFirst,
            motion_threshold: 20,
        };
        let result =
            deinterlace(&frame, width, height, 1, &config).expect("deinterlace should succeed");
        assert_eq!(result.len(), (width * height) as usize);
    }
}
