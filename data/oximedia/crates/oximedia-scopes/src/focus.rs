//! Focus assist tools for critical focus monitoring.
//!
//! Focus assist provides visual aids for checking and achieving critical focus:
//! - **Edge detection**: Sobel, Canny, Laplacian operators
//! - **Peaking**: Colored highlighting of in-focus edges
//! - **Monochrome mode**: Grayscale image with peaking for critical focus
//! - **Focus scoring**: Quantitative sharpness metrics
//!
//! Focus assist is essential for manual focus pulling in cinematography and
//! for checking focus in critical applications.

use crate::render::{rgb_to_ycbcr, Canvas};
use crate::{ScopeData, ScopeType};
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// Edge detection algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDetector {
    /// Sobel operator (3x3).
    Sobel,

    /// Laplacian operator (3x3).
    Laplacian,

    /// Scharr operator (3x3, more accurate than Sobel).
    Scharr,

    /// Simple gradient (faster, less accurate).
    Gradient,
}

/// Peaking color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeakingColor {
    /// Red peaking.
    Red,

    /// Green peaking.
    Green,

    /// Blue peaking.
    Blue,

    /// Yellow peaking.
    Yellow,

    /// Cyan peaking.
    Cyan,

    /// Magenta peaking.
    Magenta,

    /// White peaking.
    White,
}

impl PeakingColor {
    /// Returns the RGBA color for peaking.
    #[must_use]
    pub const fn to_rgba(self) -> [u8; 4] {
        match self {
            Self::Red => [255, 0, 0, 255],
            Self::Green => [0, 255, 0, 255],
            Self::Blue => [0, 0, 255, 255],
            Self::Yellow => [255, 255, 0, 255],
            Self::Cyan => [0, 255, 255, 255],
            Self::Magenta => [255, 0, 255, 255],
            Self::White => [255, 255, 255, 255],
        }
    }
}

/// Focus assist configuration.
#[derive(Debug, Clone)]
pub struct FocusAssistConfig {
    /// Edge detection algorithm.
    pub detector: EdgeDetector,

    /// Peaking color.
    pub peaking_color: PeakingColor,

    /// Peaking threshold (0-255).
    pub threshold: u8,

    /// Peaking intensity (0.0-1.0).
    pub intensity: f32,

    /// Whether to show monochrome background.
    pub monochrome: bool,
}

impl Default for FocusAssistConfig {
    fn default() -> Self {
        Self {
            detector: EdgeDetector::Sobel,
            peaking_color: PeakingColor::Red,
            threshold: 50,
            intensity: 0.8,
            monochrome: false,
        }
    }
}

/// Generates focus assist overlay with edge peaking.
///
/// # Arguments
///
/// * `frame` - RGB24 frame data (width * height * 3 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - Focus assist configuration
///
/// # Errors
///
/// Returns an error if frame data is invalid or insufficient.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::too_many_lines)]
pub fn generate_focus_assist(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &FocusAssistConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let mut canvas = Canvas::new(width, height);

    // Convert frame to luminance for edge detection
    let luma: Vec<u8> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            (0..width)
                .map(|x| {
                    let pixel_idx = ((y * width + x) * 3) as usize;
                    let r = frame[pixel_idx];
                    let g = frame[pixel_idx + 1];
                    let b = frame[pixel_idx + 2];
                    let (luma, _, _) = rgb_to_ycbcr(r, g, b);
                    luma
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Detect edges
    let edges = match config.detector {
        EdgeDetector::Sobel => detect_edges_sobel(&luma, width, height),
        EdgeDetector::Laplacian => detect_edges_laplacian(&luma, width, height),
        EdgeDetector::Scharr => detect_edges_scharr(&luma, width, height),
        EdgeDetector::Gradient => detect_edges_gradient(&luma, width, height),
    };

    // Draw base image (monochrome or color)
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let color = if config.monochrome {
                let (luma, _, _) = rgb_to_ycbcr(r, g, b);
                [luma, luma, luma, 255]
            } else {
                [r, g, b, 255]
            };

            canvas.set_pixel(x, y, color);
        }
    }

    // Overlay peaking on detected edges
    let peaking_rgba = config.peaking_color.to_rgba();
    let threshold = config.threshold;
    let intensity = config.intensity;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let edge_strength = edges[idx];

            if edge_strength > threshold {
                // Calculate peaking alpha based on edge strength and intensity
                let alpha = ((f32::from(edge_strength) / 255.0) * intensity * 255.0) as u8;

                let peaking_color = [peaking_rgba[0], peaking_rgba[1], peaking_rgba[2], alpha];

                canvas.blend_pixel(x, y, peaking_color);
            }
        }
    }

    Ok(ScopeData {
        width,
        height,
        data: canvas.data,
        scope_type: ScopeType::FocusAssist,
    })
}

/// Detects edges using Sobel operator.
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
fn detect_edges_sobel(luma: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut edges = vec![0u8; (width * height) as usize];

    // Sobel kernels
    let sobel_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    let sobel_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut gx = 0i32;
            let mut gy = 0i32;

            // Apply 3x3 kernel
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel_y = (y as i32 + ky - 1) as u32;
                    let pixel_x = (x as i32 + kx - 1) as u32;
                    let pixel_idx = (pixel_y * width + pixel_x) as usize;
                    let kernel_idx = (ky * 3 + kx) as usize;

                    let pixel_value = i32::from(luma[pixel_idx]);
                    gx += pixel_value * sobel_x[kernel_idx];
                    gy += pixel_value * sobel_y[kernel_idx];
                }
            }

            // Calculate gradient magnitude
            let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
            let edge_value = magnitude.min(255.0) as u8;

            let idx = (y * width + x) as usize;
            edges[idx] = edge_value;
        }
    }

    edges
}

/// Detects edges using Laplacian operator.
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn detect_edges_laplacian(luma: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut edges = vec![0u8; (width * height) as usize];

    // Laplacian kernel
    let laplacian = [0, 1, 0, 1, -4, 1, 0, 1, 0];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut sum = 0i32;

            // Apply 3x3 kernel
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel_y = (y as i32 + ky - 1) as u32;
                    let pixel_x = (x as i32 + kx - 1) as u32;
                    let pixel_idx = (pixel_y * width + pixel_x) as usize;
                    let kernel_idx = (ky * 3 + kx) as usize;

                    let pixel_value = i32::from(luma[pixel_idx]);
                    sum += pixel_value * laplacian[kernel_idx];
                }
            }

            let edge_value = sum.abs().min(255) as u8;

            let idx = (y * width + x) as usize;
            edges[idx] = edge_value;
        }
    }

    edges
}

/// Detects edges using Scharr operator (more accurate than Sobel).
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
fn detect_edges_scharr(luma: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut edges = vec![0u8; (width * height) as usize];

    // Scharr kernels
    let scharr_x = [-3, 0, 3, -10, 0, 10, -3, 0, 3];
    let scharr_y = [-3, -10, -3, 0, 0, 0, 3, 10, 3];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut gx = 0i32;
            let mut gy = 0i32;

            // Apply 3x3 kernel
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel_y = (y as i32 + ky - 1) as u32;
                    let pixel_x = (x as i32 + kx - 1) as u32;
                    let pixel_idx = (pixel_y * width + pixel_x) as usize;
                    let kernel_idx = (ky * 3 + kx) as usize;

                    let pixel_value = i32::from(luma[pixel_idx]);
                    gx += pixel_value * scharr_x[kernel_idx];
                    gy += pixel_value * scharr_y[kernel_idx];
                }
            }

            // Calculate gradient magnitude
            let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
            let edge_value = (magnitude / 4.0).min(255.0) as u8; // Scale down

            let idx = (y * width + x) as usize;
            edges[idx] = edge_value;
        }
    }

    edges
}

/// Detects edges using simple gradient (fast but less accurate).
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn detect_edges_gradient(luma: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut edges = vec![0u8; (width * height) as usize];

    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            let idx = (y * width + x) as usize;
            let idx_right = (y * width + x + 1) as usize;
            let idx_down = ((y + 1) * width + x) as usize;

            let current = i32::from(luma[idx]);
            let right = i32::from(luma[idx_right]);
            let down = i32::from(luma[idx_down]);

            let gx = (right - current).abs();
            let gy = (down - current).abs();

            let magnitude = gx + gy; // Manhattan distance (faster)
            edges[idx] = magnitude.min(255) as u8;
        }
    }

    edges
}

/// Computes focus score (sharpness metric) for the frame.
///
/// Higher values indicate sharper/better focused images.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_focus_score(frame: &[u8], width: u32, height: u32) -> f32 {
    // Convert to luminance
    let luma: Vec<u8> = (0..(width * height))
        .map(|i| {
            let pixel_idx = (i * 3) as usize;
            if pixel_idx + 2 < frame.len() {
                let r = frame[pixel_idx];
                let g = frame[pixel_idx + 1];
                let b = frame[pixel_idx + 2];
                let (luma, _, _) = rgb_to_ycbcr(r, g, b);
                luma
            } else {
                0
            }
        })
        .collect();

    // Use Laplacian variance as focus metric
    let edges = detect_edges_laplacian(&luma, width, height);

    // Calculate variance of edge map
    let mean = edges.iter().map(|&x| f32::from(x)).sum::<f32>() / edges.len() as f32;
    let variance = edges
        .iter()
        .map(|&x| {
            let diff = f32::from(x) - mean;
            diff * diff
        })
        .sum::<f32>()
        / edges.len() as f32;

    variance
}

/// Focus statistics.
#[derive(Debug, Clone)]
pub struct FocusStats {
    /// Overall focus score (sharpness metric).
    pub focus_score: f32,

    /// Average edge strength (0-255).
    pub avg_edge_strength: f32,

    /// Maximum edge strength (0-255).
    pub max_edge_strength: u8,

    /// Percentage of pixels with strong edges.
    pub edge_pixel_percent: f32,
}

/// Computes focus statistics from frame data.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_focus_stats(frame: &[u8], width: u32, height: u32) -> FocusStats {
    let focus_score = compute_focus_score(frame, width, height);

    // Convert to luminance
    let luma: Vec<u8> = (0..(width * height))
        .map(|i| {
            let pixel_idx = (i * 3) as usize;
            if pixel_idx + 2 < frame.len() {
                let r = frame[pixel_idx];
                let g = frame[pixel_idx + 1];
                let b = frame[pixel_idx + 2];
                let (luma, _, _) = rgb_to_ycbcr(r, g, b);
                luma
            } else {
                0
            }
        })
        .collect();

    let edges = detect_edges_sobel(&luma, width, height);

    let mut edge_sum = 0u64;
    let mut max_edge = 0u8;
    let mut strong_edge_count = 0u32;
    let threshold = 50u8;

    for &edge_value in &edges {
        edge_sum += u64::from(edge_value);
        max_edge = max_edge.max(edge_value);

        if edge_value > threshold {
            strong_edge_count += 1;
        }
    }

    let avg_edge_strength = edge_sum as f32 / edges.len() as f32;
    let edge_pixel_percent = (strong_edge_count as f32 / edges.len() as f32) * 100.0;

    FocusStats {
        focus_score,
        avg_edge_strength,
        max_edge_strength: max_edge,
        edge_pixel_percent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, blur: bool) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create checkerboard pattern (sharp) or solid (blurry)
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;

                let value = if blur {
                    128 // Uniform gray (blurry)
                } else if (x / 10 + y / 10) % 2 == 0 {
                    255 // Checkerboard (sharp)
                } else {
                    0
                };

                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        frame
    }

    #[test]
    fn test_generate_focus_assist() {
        let frame = create_test_frame(100, 100, false);
        let config = FocusAssistConfig::default();

        let result = generate_focus_assist(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, 100);
        assert_eq!(scope.height, 100);
    }

    #[test]
    fn test_edge_detectors() {
        let frame = create_test_frame(50, 50, false);
        let luma: Vec<u8> = (0..2500)
            .map(|i| {
                let pixel_idx = i * 3;
                let r = frame[pixel_idx];
                let g = frame[pixel_idx + 1];
                let b = frame[pixel_idx + 2];
                let (luma, _, _) = rgb_to_ycbcr(r, g, b);
                luma
            })
            .collect();

        let edges_sobel = detect_edges_sobel(&luma, 50, 50);
        let edges_laplacian = detect_edges_laplacian(&luma, 50, 50);
        let edges_scharr = detect_edges_scharr(&luma, 50, 50);
        let edges_gradient = detect_edges_gradient(&luma, 50, 50);

        assert_eq!(edges_sobel.len(), 2500);
        assert_eq!(edges_laplacian.len(), 2500);
        assert_eq!(edges_scharr.len(), 2500);
        assert_eq!(edges_gradient.len(), 2500);

        // Sharp image should have higher edge values
        assert!(edges_sobel.iter().any(|&x| x > 50));
    }

    #[test]
    fn test_compute_focus_score() {
        let sharp_frame = create_test_frame(100, 100, false);
        let blur_frame = create_test_frame(100, 100, true);

        let sharp_score = compute_focus_score(&sharp_frame, 100, 100);
        let blur_score = compute_focus_score(&blur_frame, 100, 100);

        // Sharp image should have higher focus score
        assert!(sharp_score > blur_score);
    }

    #[test]
    fn test_compute_focus_stats() {
        let frame = create_test_frame(100, 100, false);
        let stats = compute_focus_stats(&frame, 100, 100);

        assert!(stats.focus_score > 0.0);
        assert!(stats.avg_edge_strength >= 0.0);
        assert!(stats.max_edge_strength > 0);
        assert!(stats.edge_pixel_percent >= 0.0 && stats.edge_pixel_percent <= 100.0);
    }

    #[test]
    fn test_peaking_colors() {
        assert_eq!(PeakingColor::Red.to_rgba(), [255, 0, 0, 255]);
        assert_eq!(PeakingColor::Green.to_rgba(), [0, 255, 0, 255]);
        assert_eq!(PeakingColor::Blue.to_rgba(), [0, 0, 255, 255]);
        assert_eq!(PeakingColor::Yellow.to_rgba(), [255, 255, 0, 255]);
    }

    #[test]
    fn test_focus_assist_modes() {
        let frame = create_test_frame(50, 50, false);

        // Test different detectors
        for detector in &[
            EdgeDetector::Sobel,
            EdgeDetector::Laplacian,
            EdgeDetector::Scharr,
            EdgeDetector::Gradient,
        ] {
            let config = FocusAssistConfig {
                detector: *detector,
                ..Default::default()
            };

            let result = generate_focus_assist(&frame, 50, 50, &config);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_focus_assist_monochrome() {
        let frame = create_test_frame(50, 50, false);

        let config = FocusAssistConfig {
            monochrome: true,
            ..Default::default()
        };

        let result = generate_focus_assist(&frame, 50, 50, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_frame_size() {
        let frame = vec![0u8; 100]; // Too small
        let config = FocusAssistConfig::default();

        let result = generate_focus_assist(&frame, 100, 100, &config);
        assert!(result.is_err());
    }
}
