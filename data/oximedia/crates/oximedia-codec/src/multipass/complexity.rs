//! Frame complexity analysis for multi-pass encoding.
//!
//! This module provides advanced frame complexity metrics used in multi-pass
//! encoding to make better bitrate allocation decisions.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use crate::frame::{FrameType, VideoFrame};

/// Frame complexity metrics for bitrate allocation.
#[derive(Clone, Debug)]
pub struct FrameComplexity {
    /// Frame index in the stream.
    pub frame_index: u64,
    /// Frame type (Key, Inter, BiDir).
    pub frame_type: FrameType,
    /// Spatial complexity (0.0-1.0, based on variance).
    pub spatial_complexity: f64,
    /// Temporal complexity (0.0-1.0, based on motion).
    pub temporal_complexity: f64,
    /// Combined complexity metric.
    pub combined_complexity: f64,
    /// Sum of Absolute Differences with previous frame.
    pub sad: u64,
    /// Average luma variance across blocks.
    pub variance: f64,
    /// Estimated encoding difficulty (1.0 = average).
    pub encoding_difficulty: f64,
    /// Is this frame a scene change.
    pub is_scene_change: bool,
}

impl FrameComplexity {
    /// Create a new frame complexity with default values.
    #[must_use]
    pub fn new(frame_index: u64, frame_type: FrameType) -> Self {
        Self {
            frame_index,
            frame_type,
            spatial_complexity: 0.5,
            temporal_complexity: 0.5,
            combined_complexity: 0.5,
            sad: 0,
            variance: 0.0,
            encoding_difficulty: 1.0,
            is_scene_change: false,
        }
    }

    /// Calculate relative difficulty compared to average frame.
    #[must_use]
    pub fn relative_difficulty(&self) -> f64 {
        self.encoding_difficulty
    }
}

/// Complexity analyzer for video frames.
pub struct ComplexityAnalyzer {
    width: u32,
    height: u32,
    block_size: usize,
    prev_frame: Option<Vec<u8>>,
    spatial_history: Vec<f64>,
    temporal_history: Vec<f64>,
    scene_change_threshold: f64,
}

impl ComplexityAnalyzer {
    /// Create a new complexity analyzer.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            block_size: 16,
            prev_frame: None,
            spatial_history: Vec::new(),
            temporal_history: Vec::new(),
            scene_change_threshold: 0.4,
        }
    }

    /// Set the scene change detection threshold.
    pub fn set_scene_change_threshold(&mut self, threshold: f64) {
        self.scene_change_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Analyze a video frame and compute complexity metrics.
    #[must_use]
    pub fn analyze(&mut self, frame: &VideoFrame, frame_index: u64) -> FrameComplexity {
        let mut complexity = FrameComplexity::new(frame_index, frame.frame_type);

        // Get luma plane
        if let Some(luma_plane) = frame.planes.first() {
            let luma_data = luma_plane.data.as_ref();
            let stride = luma_plane.stride;

            // Calculate spatial complexity
            complexity.spatial_complexity = self.compute_spatial_complexity(luma_data, stride);
            complexity.variance = self.compute_variance(luma_data, stride);

            // Calculate temporal complexity
            if let Some(prev) = &self.prev_frame {
                complexity.sad = self.compute_sad(luma_data, prev, stride);
                complexity.temporal_complexity = self.compute_temporal_complexity(complexity.sad);
                complexity.is_scene_change = self.detect_scene_change(complexity.sad);
            } else {
                complexity.temporal_complexity = 1.0;
                complexity.is_scene_change = true;
            }

            // Store current frame for next iteration
            self.prev_frame = Some(luma_data.to_vec());

            // Update history
            self.spatial_history.push(complexity.spatial_complexity);
            self.temporal_history.push(complexity.temporal_complexity);
            if self.spatial_history.len() > 100 {
                self.spatial_history.remove(0);
                self.temporal_history.remove(0);
            }

            // Calculate combined complexity
            complexity.combined_complexity = self.compute_combined_complexity(&complexity);

            // Estimate encoding difficulty
            complexity.encoding_difficulty = self.estimate_difficulty(&complexity);
        }

        complexity
    }

    /// Compute spatial complexity using block-based variance.
    fn compute_spatial_complexity(&self, luma: &[u8], stride: usize) -> f64 {
        let blocks_x = (self.width as usize) / self.block_size;
        let blocks_y = (self.height as usize) / self.block_size;

        if blocks_x == 0 || blocks_y == 0 {
            return 0.5;
        }

        let mut total_variance = 0.0;
        let mut block_count = 0;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let variance = self.compute_block_variance(luma, stride, bx, by);
                total_variance += variance;
                block_count += 1;
            }
        }

        if block_count == 0 {
            return 0.5;
        }

        let avg_variance = total_variance / block_count as f64;
        // Normalize to 0-1 range (assuming max variance ~1000)
        (avg_variance / 1000.0).min(1.0)
    }

    /// Compute block variance.
    fn compute_block_variance(&self, luma: &[u8], stride: usize, bx: usize, by: usize) -> f64 {
        let start_x = bx * self.block_size;
        let start_y = by * self.block_size;

        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        let mut count = 0u64;

        for y in 0..self.block_size {
            let row_y = start_y + y;
            if row_y >= self.height as usize {
                break;
            }

            for x in 0..self.block_size {
                let col_x = start_x + x;
                if col_x >= self.width as usize {
                    break;
                }

                let idx = row_y * stride + col_x;
                if idx < luma.len() {
                    let pixel = luma[idx] as u64;
                    sum += pixel;
                    sum_sq += pixel * pixel;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }

        let mean = sum as f64 / count as f64;
        let mean_sq = sum_sq as f64 / count as f64;
        (mean_sq - mean * mean).max(0.0)
    }

    /// Compute overall frame variance.
    fn compute_variance(&self, luma: &[u8], stride: usize) -> f64 {
        let height = self.height as usize;
        let width = self.width as usize;

        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        let mut count = 0u64;

        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                if idx < luma.len() {
                    let pixel = luma[idx] as u64;
                    sum += pixel;
                    sum_sq += pixel * pixel;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }

        let mean = sum as f64 / count as f64;
        let mean_sq = sum_sq as f64 / count as f64;
        (mean_sq - mean * mean).max(0.0)
    }

    /// Compute Sum of Absolute Differences between frames.
    fn compute_sad(&self, current: &[u8], previous: &[u8], stride: usize) -> u64 {
        let height = self.height as usize;
        let width = self.width as usize;
        let mut sad = 0u64;

        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                if idx < current.len() && idx < previous.len() {
                    let diff = (current[idx] as i32 - previous[idx] as i32).abs();
                    sad += diff as u64;
                }
            }
        }

        sad
    }

    /// Compute temporal complexity from SAD value.
    fn compute_temporal_complexity(&self, sad: u64) -> f64 {
        let pixels = (self.width as u64) * (self.height as u64);
        if pixels == 0 {
            return 0.5;
        }

        // Average SAD per pixel
        let avg_sad = sad as f64 / pixels as f64;
        // Normalize to 0-1 range (assuming max avg SAD ~50)
        (avg_sad / 50.0).min(1.0)
    }

    /// Detect scene changes based on SAD threshold.
    fn detect_scene_change(&self, sad: u64) -> bool {
        let pixels = (self.width as u64) * (self.height as u64);
        if pixels == 0 {
            return false;
        }

        let avg_sad = sad as f64 / pixels as f64;
        let normalized_sad = (avg_sad / 50.0).min(1.0);
        normalized_sad > self.scene_change_threshold
    }

    /// Compute combined complexity metric.
    fn compute_combined_complexity(&self, complexity: &FrameComplexity) -> f64 {
        // Weight spatial and temporal components
        let spatial_weight = 0.6;
        let temporal_weight = 0.4;

        spatial_weight * complexity.spatial_complexity
            + temporal_weight * complexity.temporal_complexity
    }

    /// Estimate encoding difficulty based on complexity metrics.
    fn estimate_difficulty(&self, complexity: &FrameComplexity) -> f64 {
        let mut difficulty = 1.0;

        // Base difficulty on combined complexity
        difficulty *= 0.5 + complexity.combined_complexity;

        // Adjust for frame type
        difficulty *= match complexity.frame_type {
            FrameType::Key => 2.0,    // Keyframes are more expensive
            FrameType::Inter => 1.0,  // Inter frames are baseline
            FrameType::BiDir => 0.8,  // B-frames can be cheaper
            FrameType::Switch => 1.5, // Switch frames need extra bits
        };

        // Scene changes require more bits
        if complexity.is_scene_change {
            difficulty *= 1.5;
        }

        // Normalize against historical average
        if !self.spatial_history.is_empty() {
            let avg_spatial: f64 =
                self.spatial_history.iter().sum::<f64>() / self.spatial_history.len() as f64;
            let avg_temporal: f64 =
                self.temporal_history.iter().sum::<f64>() / self.temporal_history.len() as f64;

            let historical_avg = 0.6 * avg_spatial + 0.4 * avg_temporal;
            if historical_avg > 0.01 {
                difficulty *= complexity.combined_complexity / historical_avg;
            }
        }

        difficulty.max(0.1).min(10.0)
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        self.prev_frame = None;
        self.spatial_history.clear();
        self.temporal_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Plane;
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    fn create_test_frame(width: u32, height: u32, value: u8) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        let size = (width * height) as usize;
        let data = vec![value; size];
        frame.planes.push(Plane::new(data, width as usize));
        frame.timestamp = Timestamp::new(0, Rational::new(1, 30));
        frame
    }

    #[test]
    fn test_complexity_analyzer_new() {
        let analyzer = ComplexityAnalyzer::new(1920, 1080);
        assert_eq!(analyzer.width, 1920);
        assert_eq!(analyzer.height, 1080);
        assert_eq!(analyzer.block_size, 16);
    }

    #[test]
    fn test_analyze_solid_frame() {
        let mut analyzer = ComplexityAnalyzer::new(320, 240);
        let frame = create_test_frame(320, 240, 128);

        let complexity = analyzer.analyze(&frame, 0);
        assert_eq!(complexity.frame_index, 0);
        assert!(complexity.spatial_complexity < 0.1); // Solid color = low spatial
        assert!(complexity.variance < 1.0); // Very low variance
    }

    #[test]
    fn test_scene_change_detection() {
        let mut analyzer = ComplexityAnalyzer::new(320, 240);

        // First frame
        let frame1 = create_test_frame(320, 240, 0);
        let complexity1 = analyzer.analyze(&frame1, 0);
        assert!(complexity1.is_scene_change); // First frame is always scene change

        // Very different second frame
        let frame2 = create_test_frame(320, 240, 255);
        let complexity2 = analyzer.analyze(&frame2, 1);
        assert!(complexity2.is_scene_change); // Should detect big difference
    }

    #[test]
    fn test_no_scene_change() {
        let mut analyzer = ComplexityAnalyzer::new(320, 240);

        // First frame
        let frame1 = create_test_frame(320, 240, 128);
        let _ = analyzer.analyze(&frame1, 0);

        // Similar second frame
        let frame2 = create_test_frame(320, 240, 130);
        let complexity2 = analyzer.analyze(&frame2, 1);
        assert!(!complexity2.is_scene_change); // Should not detect scene change
    }

    #[test]
    fn test_encoding_difficulty() {
        let mut analyzer = ComplexityAnalyzer::new(320, 240);
        let frame = create_test_frame(320, 240, 128);

        let complexity = analyzer.analyze(&frame, 0);
        assert!(complexity.encoding_difficulty > 0.0);
        assert!(complexity.encoding_difficulty <= 10.0);
    }
}
