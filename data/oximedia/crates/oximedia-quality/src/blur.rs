//! Blur detection for video quality assessment.
//!
//! Detects blur using multiple methods:
//! - Laplacian variance (measures focus/sharpness)
//! - Tenengrad variance (gradient magnitude variance)
//! - Edge width analysis
//!
//! Higher scores indicate sharper images (better quality).

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;

/// Blur detection method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlurMethod {
    /// Laplacian variance method
    Laplacian,
    /// Tenengrad variance method
    Tenengrad,
    /// Combined method (average of all)
    Combined,
}

/// Blur detector for video quality assessment.
pub struct BlurDetector {
    /// Detection method to use
    method: BlurMethod,
    /// Threshold for edge detection
    edge_threshold: f64,
}

impl BlurDetector {
    /// Creates a new blur detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            method: BlurMethod::Combined,
            edge_threshold: 10.0,
        }
    }

    /// Creates a blur detector with specific method.
    #[must_use]
    pub fn with_method(method: BlurMethod) -> Self {
        Self {
            method,
            edge_threshold: 10.0,
        }
    }

    /// Detects blur in a frame.
    ///
    /// Returns a sharpness score where higher values indicate less blur.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is too small.
    pub fn detect(&self, frame: &Frame) -> OxiResult<QualityScore> {
        let mut score = QualityScore::new(MetricType::Blur, 0.0);

        // Calculate blur for Y plane
        let y_sharpness = self.detect_plane(&frame.planes[0], frame.width, frame.height)?;

        score.add_component("Y", y_sharpness);
        score.score = y_sharpness;

        Ok(score)
    }

    /// Detects blur in a single plane.
    fn detect_plane(&self, plane: &[u8], width: usize, height: usize) -> OxiResult<f64> {
        if width < 8 || height < 8 {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame too small for blur detection".to_string(),
            ));
        }

        let sharpness = match self.method {
            BlurMethod::Laplacian => self.laplacian_variance(plane, width, height),
            BlurMethod::Tenengrad => self.tenengrad_variance(plane, width, height),
            BlurMethod::Combined => {
                let lap = self.laplacian_variance(plane, width, height);
                let ten = self.tenengrad_variance(plane, width, height);
                (lap + ten) / 2.0
            }
        };

        Ok(sharpness)
    }

    /// Computes Laplacian variance (measures focus).
    fn laplacian_variance(&self, plane: &[u8], width: usize, height: usize) -> f64 {
        // Laplacian kernel
        let kernel = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

        let mut laplacian_values = Vec::new();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0.0;

                for dy in 0..3 {
                    for dx in 0..3 {
                        let idx = (y + dy - 1) * width + (x + dx - 1);
                        sum += f64::from(plane[idx]) * kernel[dy * 3 + dx];
                    }
                }

                laplacian_values.push(sum);
            }
        }

        if laplacian_values.is_empty() {
            return 0.0;
        }

        // Variance of Laplacian
        let mean = laplacian_values.iter().sum::<f64>() / laplacian_values.len() as f64;
        let variance = laplacian_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / laplacian_values.len() as f64;

        variance
    }

    /// Computes Tenengrad variance (gradient magnitude).
    fn tenengrad_variance(&self, plane: &[u8], width: usize, height: usize) -> f64 {
        // Sobel kernels
        let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let mut gradient_magnitudes = Vec::new();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for dy in 0..3 {
                    for dx in 0..3 {
                        let idx = (y + dy - 1) * width + (x + dx - 1);
                        let val = f64::from(plane[idx]);
                        let kernel_idx = dy * 3 + dx;
                        gx += val * sobel_x[kernel_idx];
                        gy += val * sobel_y[kernel_idx];
                    }
                }

                let magnitude = (gx * gx + gy * gy).sqrt();

                // Only consider pixels above threshold (edges)
                if magnitude > self.edge_threshold {
                    gradient_magnitudes.push(magnitude);
                }
            }
        }

        if gradient_magnitudes.is_empty() {
            return 0.0;
        }

        // Variance of gradient magnitudes
        let mean = gradient_magnitudes.iter().sum::<f64>() / gradient_magnitudes.len() as f64;
        let variance = gradient_magnitudes
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / gradient_magnitudes.len() as f64;

        variance
    }
}

impl Default for BlurDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        frame.planes[0].fill(value);
        frame
    }

    fn create_sharp_frame(width: usize, height: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");

        // Create sharp edges (checkerboard pattern)
        for y in 0..height {
            for x in 0..width {
                let value = if (x / 8 + y / 8) % 2 == 0 { 0 } else { 255 };
                frame.planes[0][y * width + x] = value;
            }
        }

        frame
    }

    fn create_gradient_frame(width: usize, height: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");

        // Create smooth gradient (blurrier)
        for y in 0..height {
            for x in 0..width {
                frame.planes[0][y * width + x] = (x * 255 / width) as u8;
            }
        }

        frame
    }

    #[test]
    fn test_blur_detection() {
        let detector = BlurDetector::new();
        let frame = create_test_frame(64, 64, 128);

        let result = detector.detect(&frame).expect("should succeed in test");
        assert!(result.score >= 0.0);
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_sharp_vs_blurry() {
        let detector = BlurDetector::new();

        // Sharp frame should have higher sharpness score
        let sharp = create_sharp_frame(64, 64);
        let sharp_score = detector.detect(&sharp).expect("should succeed in test");

        // Smooth gradient (blurrier) should have lower score
        let blurry = create_gradient_frame(64, 64);
        let blurry_score = detector.detect(&blurry).expect("should succeed in test");

        // Uniform frame (no edges) should have very low score
        let uniform = create_test_frame(64, 64, 128);
        let uniform_score = detector.detect(&uniform).expect("should succeed in test");

        // Sharp should have highest score
        assert!(sharp_score.score > blurry_score.score);
        assert!(sharp_score.score > uniform_score.score);
    }

    #[test]
    fn test_laplacian_method() {
        let detector = BlurDetector::with_method(BlurMethod::Laplacian);
        let frame = create_sharp_frame(64, 64);

        let result = detector.detect(&frame).expect("should succeed in test");
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_tenengrad_method() {
        let detector = BlurDetector::with_method(BlurMethod::Tenengrad);
        let frame = create_sharp_frame(64, 64);

        let result = detector.detect(&frame).expect("should succeed in test");
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_laplacian_variance() {
        let detector = BlurDetector::new();
        let plane = vec![
            0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0,
        ];

        let variance = detector.laplacian_variance(&plane, 5, 4);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_tenengrad_variance() {
        let detector = BlurDetector::new();
        let plane = vec![
            0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0,
        ];

        let variance = detector.tenengrad_variance(&plane, 5, 4);
        assert!(variance >= 0.0);
    }
}
