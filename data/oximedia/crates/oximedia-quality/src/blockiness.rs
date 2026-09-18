//! Blockiness detection for compression artifacts.
//!
//! Detects blocking artifacts caused by block-based compression (DCT, etc.).
//! Uses edge detection at block boundaries and compares to edge detection
//! within blocks.

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;

/// Blockiness detector for compression artifact assessment.
pub struct BlockinessDetector {
    /// Block size to check (typically 8 for DCT)
    block_size: usize,
}

impl BlockinessDetector {
    /// Creates a new blockiness detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self { block_size: 8 }
    }

    /// Creates a blockiness detector with custom block size.
    #[must_use]
    pub fn with_block_size(block_size: usize) -> Self {
        Self { block_size }
    }

    /// Detects blockiness in a frame.
    ///
    /// Returns a score where higher values indicate more blockiness (worse quality).
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is too small.
    pub fn detect(&self, frame: &Frame) -> OxiResult<QualityScore> {
        let mut score = QualityScore::new(MetricType::Blockiness, 0.0);

        // Calculate blockiness for Y plane
        let y_blockiness = self.detect_plane(&frame.planes[0], frame.width, frame.height)?;

        score.add_component("Y", y_blockiness);
        score.score = y_blockiness;

        Ok(score)
    }

    /// Detects blockiness in a single plane.
    fn detect_plane(&self, plane: &[u8], width: usize, height: usize) -> OxiResult<f64> {
        if width < self.block_size * 2 || height < self.block_size * 2 {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame too small for blockiness detection".to_string(),
            ));
        }

        // Compute horizontal and vertical blockiness
        let h_blockiness = self.compute_horizontal_blockiness(plane, width, height);
        let v_blockiness = self.compute_vertical_blockiness(plane, width, height);

        // Average blockiness
        let blockiness = (h_blockiness + v_blockiness) / 2.0;

        Ok(blockiness)
    }

    /// Computes horizontal blockiness (vertical edges).
    fn compute_horizontal_blockiness(&self, plane: &[u8], width: usize, height: usize) -> f64 {
        let mut block_edge_sum = 0.0;
        let mut non_block_edge_sum = 0.0;
        let mut block_edge_count = 0.0;
        let mut non_block_edge_count = 0.0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let diff = f64::from(
                    (i32::from(plane[y * width + x + 1]) - i32::from(plane[y * width + x - 1]))
                        .abs(),
                );

                if x % self.block_size == 0 {
                    // Block boundary
                    block_edge_sum += diff;
                    block_edge_count += 1.0;
                } else {
                    // Non-block boundary
                    non_block_edge_sum += diff;
                    non_block_edge_count += 1.0;
                }
            }
        }

        let avg_block_edge = if block_edge_count > 0.0 {
            block_edge_sum / block_edge_count
        } else {
            0.0
        };

        let avg_non_block_edge = if non_block_edge_count > 0.0 {
            non_block_edge_sum / non_block_edge_count
        } else {
            1.0
        };

        // Blockiness is the ratio of block edge strength to non-block edge strength
        if avg_non_block_edge > 1e-10 {
            (avg_block_edge / avg_non_block_edge - 1.0).max(0.0) * 100.0
        } else {
            0.0
        }
    }

    /// Computes vertical blockiness (horizontal edges).
    fn compute_vertical_blockiness(&self, plane: &[u8], width: usize, height: usize) -> f64 {
        let mut block_edge_sum = 0.0;
        let mut non_block_edge_sum = 0.0;
        let mut block_edge_count = 0.0;
        let mut non_block_edge_count = 0.0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let diff = f64::from(
                    (i32::from(plane[(y + 1) * width + x]) - i32::from(plane[(y - 1) * width + x]))
                        .abs(),
                );

                if y % self.block_size == 0 {
                    // Block boundary
                    block_edge_sum += diff;
                    block_edge_count += 1.0;
                } else {
                    // Non-block boundary
                    non_block_edge_sum += diff;
                    non_block_edge_count += 1.0;
                }
            }
        }

        let avg_block_edge = if block_edge_count > 0.0 {
            block_edge_sum / block_edge_count
        } else {
            0.0
        };

        let avg_non_block_edge = if non_block_edge_count > 0.0 {
            non_block_edge_sum / non_block_edge_count
        } else {
            1.0
        };

        if avg_non_block_edge > 1e-10 {
            (avg_block_edge / avg_non_block_edge - 1.0).max(0.0) * 100.0
        } else {
            0.0
        }
    }
}

impl Default for BlockinessDetector {
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

    fn create_blocky_frame(width: usize, height: usize, block_size: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");

        for y in 0..height {
            for x in 0..width {
                let block_x = x / block_size;
                let block_y = y / block_size;
                let value = ((block_x + block_y) % 2) * 128 + 64;
                frame.planes[0][y * width + x] = value as u8;
            }
        }

        frame
    }

    #[test]
    fn test_blockiness_detection() {
        let detector = BlockinessDetector::new();
        let frame = create_test_frame(128, 128, 128);

        let result = detector.detect(&frame).expect("should succeed in test");
        assert!(result.score >= 0.0);
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_blocky_vs_smooth() {
        let detector = BlockinessDetector::new();

        // Smooth frame should have low blockiness
        let smooth = create_test_frame(128, 128, 128);
        let smooth_score = detector.detect(&smooth).expect("should succeed in test");

        // Blocky frame should have higher blockiness
        let blocky = create_blocky_frame(128, 128, 8);
        let blocky_score = detector.detect(&blocky).expect("should succeed in test");

        // Blocky frame should have higher score (more blockiness)
        assert!(blocky_score.score >= smooth_score.score);
    }

    #[test]
    fn test_custom_block_size() {
        let detector = BlockinessDetector::with_block_size(16);
        assert_eq!(detector.block_size, 16);

        let frame = create_blocky_frame(128, 128, 16);
        let result = detector.detect(&frame).expect("should succeed in test");
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_horizontal_blockiness() {
        let detector = BlockinessDetector::new();
        let frame = create_blocky_frame(128, 128, 8);

        let h_block = detector.compute_horizontal_blockiness(&frame.planes[0], 128, 128);
        assert!(h_block >= 0.0);
    }

    #[test]
    fn test_vertical_blockiness() {
        let detector = BlockinessDetector::new();
        let frame = create_blocky_frame(128, 128, 8);

        let v_block = detector.compute_vertical_blockiness(&frame.planes[0], 128, 128);
        assert!(v_block >= 0.0);
    }
}
