//! Frame complexity estimation.
//!
//! This module provides methods for estimating frame complexity, which is
//! essential for accurate rate control. Complexity metrics include:
//!
//! - Spatial complexity (texture/detail)
//! - Temporal complexity (motion)
//! - Combined metrics for rate control decisions

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]
#![forbid(unsafe_code)]

/// Frame complexity estimator.
///
/// Estimates spatial and temporal complexity of video frames to guide
/// rate control decisions.
#[derive(Clone, Debug)]
pub struct ComplexityEstimator {
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Block size for analysis.
    block_size: u32,
    /// Running average of spatial complexity.
    avg_spatial: f32,
    /// Running average of temporal complexity.
    avg_temporal: f32,
    /// Running average of combined complexity.
    avg_combined: f32,
    /// Weight for exponential moving average.
    ema_weight: f32,
    /// Frame counter.
    frame_count: u64,
    /// Previous frame data for temporal analysis.
    prev_frame_data: Option<FrameData>,
}

/// Stored frame data for temporal analysis.
#[derive(Clone, Debug)]
struct FrameData {
    /// Per-block variance values (reserved for motion-compensated analysis).
    #[allow(dead_code)]
    block_variances: Vec<f32>,
    /// Per-block average values (reserved for motion-compensated analysis).
    #[allow(dead_code)]
    block_averages: Vec<f32>,
    /// Total frame variance.
    total_variance: f32,
}

impl ComplexityEstimator {
    /// Create a new complexity estimator.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            block_size: 16,
            avg_spatial: 1.0,
            avg_temporal: 1.0,
            avg_combined: 1.0,
            ema_weight: 0.1,
            frame_count: 0,
            prev_frame_data: None,
        }
    }

    /// Set the block size for analysis.
    pub fn set_block_size(&mut self, size: u32) {
        self.block_size = size.max(4).min(64);
    }

    /// Estimate complexity of a frame's luma plane.
    #[must_use]
    pub fn estimate(&mut self, luma: &[u8], stride: usize) -> ComplexityResult {
        let spatial = self.estimate_spatial(luma, stride);
        let temporal = self.estimate_temporal(luma, stride);

        // Combined metric (geometric mean)
        let combined = (spatial * temporal).sqrt();

        // Update running averages
        self.update_averages(spatial, temporal, combined);

        // Store frame data for next frame's temporal analysis
        self.store_frame_data(luma, stride);

        self.frame_count += 1;

        ComplexityResult {
            spatial,
            temporal,
            combined,
            normalized: combined / self.avg_combined,
        }
    }

    /// Estimate spatial complexity using variance-based method.
    fn estimate_spatial(&self, luma: &[u8], stride: usize) -> f32 {
        let blocks_x = self.width / self.block_size;
        let blocks_y = self.height / self.block_size;

        if blocks_x == 0 || blocks_y == 0 {
            return 1.0;
        }

        let mut total_variance = 0.0f64;
        let mut block_count = 0u32;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let variance = self.calculate_block_variance(luma, stride, bx, by);
                total_variance += variance as f64;
                block_count += 1;
            }
        }

        if block_count == 0 {
            return 1.0;
        }

        let avg_variance = (total_variance / block_count as f64) as f32;

        // Normalize variance to a reasonable scale
        // High variance = high complexity
        (avg_variance / 100.0).sqrt().max(0.1)
    }

    /// Calculate variance of a single block.
    fn calculate_block_variance(&self, luma: &[u8], stride: usize, bx: u32, by: u32) -> f32 {
        let start_x = (bx * self.block_size) as usize;
        let start_y = (by * self.block_size) as usize;
        let block_size = self.block_size as usize;

        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        let mut count = 0u32;

        for y in 0..block_size {
            let row_start = (start_y + y) * stride + start_x;
            if row_start + block_size > luma.len() {
                continue;
            }

            for x in 0..block_size {
                let pixel = luma[row_start + x] as u64;
                sum += pixel;
                sum_sq += pixel * pixel;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let mean = sum as f32 / count as f32;
        let mean_sq = sum_sq as f32 / count as f32;
        let variance = mean_sq - mean * mean;

        variance.max(0.0)
    }

    /// Estimate temporal complexity using SAD-based method.
    fn estimate_temporal(&self, luma: &[u8], stride: usize) -> f32 {
        let prev_data = match &self.prev_frame_data {
            Some(data) => data,
            None => return 1.0, // First frame, assume average complexity
        };

        // Calculate SAD between current and previous frame
        let sad = self.calculate_frame_sad(luma, stride, prev_data);

        // Normalize SAD to complexity metric
        let pixels = self.width as f64 * self.height as f64;
        if pixels == 0.0 {
            return 1.0;
        }

        let normalized_sad = sad as f64 / pixels;

        // Map to reasonable range
        (normalized_sad as f32 / 10.0).sqrt().max(0.1)
    }

    /// Calculate SAD (Sum of Absolute Differences) against previous frame.
    fn calculate_frame_sad(&self, luma: &[u8], stride: usize, _prev: &FrameData) -> u64 {
        // Simplified: just calculate variance difference
        // In a full implementation, this would do motion-compensated SAD
        let current_variance = self.calculate_total_variance(luma, stride);
        let prev_variance = _prev.total_variance;

        ((current_variance - prev_variance).abs() * 1000.0) as u64
    }

    /// Calculate total frame variance.
    fn calculate_total_variance(&self, luma: &[u8], stride: usize) -> f32 {
        let total_pixels = (self.width * self.height) as usize;
        if total_pixels == 0 || luma.len() < total_pixels {
            return 0.0;
        }

        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        let mut count = 0u64;

        for y in 0..self.height as usize {
            let row_start = y * stride;
            let row_end = (row_start + self.width as usize).min(luma.len());

            for x in row_start..row_end {
                let pixel = luma[x] as u64;
                sum += pixel;
                sum_sq += pixel * pixel;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let mean = sum as f32 / count as f32;
        let mean_sq = sum_sq as f32 / count as f32;
        (mean_sq - mean * mean).max(0.0)
    }

    /// Store frame data for temporal analysis.
    fn store_frame_data(&mut self, luma: &[u8], stride: usize) {
        let blocks_x = self.width / self.block_size;
        let blocks_y = self.height / self.block_size;

        let mut block_variances = Vec::with_capacity((blocks_x * blocks_y) as usize);
        let mut block_averages = Vec::with_capacity((blocks_x * blocks_y) as usize);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let (avg, var) = self.calculate_block_stats(luma, stride, bx, by);
                block_variances.push(var);
                block_averages.push(avg);
            }
        }

        let total_variance = self.calculate_total_variance(luma, stride);

        self.prev_frame_data = Some(FrameData {
            block_variances,
            block_averages,
            total_variance,
        });
    }

    /// Calculate block average and variance.
    fn calculate_block_stats(&self, luma: &[u8], stride: usize, bx: u32, by: u32) -> (f32, f32) {
        let start_x = (bx * self.block_size) as usize;
        let start_y = (by * self.block_size) as usize;
        let block_size = self.block_size as usize;

        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        let mut count = 0u32;

        for y in 0..block_size {
            let row_start = (start_y + y) * stride + start_x;
            if row_start + block_size > luma.len() {
                continue;
            }

            for x in 0..block_size {
                let pixel = luma[row_start + x] as u64;
                sum += pixel;
                sum_sq += pixel * pixel;
                count += 1;
            }
        }

        if count == 0 {
            return (128.0, 0.0);
        }

        let mean = sum as f32 / count as f32;
        let mean_sq = sum_sq as f32 / count as f32;
        let variance = (mean_sq - mean * mean).max(0.0);

        (mean, variance)
    }

    /// Update running averages.
    fn update_averages(&mut self, spatial: f32, temporal: f32, combined: f32) {
        let w = self.ema_weight;
        self.avg_spatial = self.avg_spatial * (1.0 - w) + spatial * w;
        self.avg_temporal = self.avg_temporal * (1.0 - w) + temporal * w;
        self.avg_combined = self.avg_combined * (1.0 - w) + combined * w;
    }

    /// Get average spatial complexity.
    #[must_use]
    pub fn avg_spatial(&self) -> f32 {
        self.avg_spatial
    }

    /// Get average temporal complexity.
    #[must_use]
    pub fn avg_temporal(&self) -> f32 {
        self.avg_temporal
    }

    /// Get average combined complexity.
    #[must_use]
    pub fn avg_combined(&self) -> f32 {
        self.avg_combined
    }

    /// Get frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset the estimator state.
    pub fn reset(&mut self) {
        self.avg_spatial = 1.0;
        self.avg_temporal = 1.0;
        self.avg_combined = 1.0;
        self.frame_count = 0;
        self.prev_frame_data = None;
    }
}

impl Default for ComplexityEstimator {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

/// Result of complexity estimation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ComplexityResult {
    /// Spatial complexity (texture/detail).
    pub spatial: f32,
    /// Temporal complexity (motion).
    pub temporal: f32,
    /// Combined complexity metric.
    pub combined: f32,
    /// Normalized complexity (relative to average).
    pub normalized: f32,
}

impl ComplexityResult {
    /// Create a result with default complexity.
    #[must_use]
    pub fn default_complexity() -> Self {
        Self {
            spatial: 1.0,
            temporal: 1.0,
            combined: 1.0,
            normalized: 1.0,
        }
    }

    /// Check if this is a high complexity frame.
    #[must_use]
    pub fn is_high_complexity(&self) -> bool {
        self.normalized > 1.5
    }

    /// Check if this is a low complexity frame.
    #[must_use]
    pub fn is_low_complexity(&self) -> bool {
        self.normalized < 0.7
    }

    /// Get suggested QP adjustment based on complexity.
    #[must_use]
    pub fn suggested_qp_offset(&self) -> f32 {
        // Higher complexity -> higher QP offset (use more compression)
        // Lower complexity -> lower QP offset (better quality)
        let log_ratio = self.normalized.ln();
        (log_ratio * 2.0).clamp(-4.0, 4.0)
    }
}

/// Motion complexity analyzer.
#[derive(Clone, Debug)]
pub struct MotionAnalyzer {
    /// Block size for motion analysis.
    block_size: u32,
    /// Search range for motion estimation (reserved for full motion search).
    #[allow(dead_code)]
    search_range: u32,
    /// Previous frame luma.
    prev_luma: Option<Vec<u8>>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
}

impl MotionAnalyzer {
    /// Create a new motion analyzer.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            block_size: 16,
            search_range: 16,
            prev_luma: None,
            width,
            height,
        }
    }

    /// Analyze motion in a frame.
    pub fn analyze(&mut self, luma: &[u8], stride: usize) -> MotionResult {
        let result = if let Some(ref prev) = self.prev_luma {
            self.calculate_motion(prev, luma, stride)
        } else {
            MotionResult::default()
        };

        // Store current frame for next comparison
        self.store_frame(luma, stride);

        result
    }

    /// Calculate motion metrics between frames.
    fn calculate_motion(&self, prev: &[u8], curr: &[u8], stride: usize) -> MotionResult {
        let blocks_x = self.width / self.block_size;
        let blocks_y = self.height / self.block_size;

        if blocks_x == 0 || blocks_y == 0 {
            return MotionResult::default();
        }

        let mut total_sad = 0u64;
        let mut motion_blocks = 0u32;
        let mut max_motion = 0f32;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let (sad, motion) = self.analyze_block(prev, curr, stride, bx, by);
                total_sad += sad;

                if motion > 2.0 {
                    motion_blocks += 1;
                }
                if motion > max_motion {
                    max_motion = motion;
                }
            }
        }

        let total_blocks = blocks_x * blocks_y;
        let avg_sad = total_sad as f32 / total_blocks as f32;
        let motion_ratio = motion_blocks as f32 / total_blocks as f32;

        MotionResult {
            average_sad: avg_sad,
            motion_ratio,
            max_motion,
            complexity: (avg_sad / 100.0).sqrt() * (1.0 + motion_ratio),
        }
    }

    /// Analyze a single block for motion.
    fn analyze_block(
        &self,
        prev: &[u8],
        curr: &[u8],
        stride: usize,
        bx: u32,
        by: u32,
    ) -> (u64, f32) {
        let start_x = (bx * self.block_size) as usize;
        let start_y = (by * self.block_size) as usize;
        let block_size = self.block_size as usize;

        // Calculate SAD at (0,0) position (no motion)
        let mut sad = 0u64;

        for y in 0..block_size {
            let curr_row = (start_y + y) * stride + start_x;
            let prev_row = (start_y + y) * stride + start_x;

            if curr_row + block_size > curr.len() || prev_row + block_size > prev.len() {
                continue;
            }

            for x in 0..block_size {
                let diff = (curr[curr_row + x] as i32 - prev[prev_row + x] as i32).unsigned_abs();
                sad += diff as u64;
            }
        }

        let pixels = (block_size * block_size) as f32;
        let avg_diff = sad as f32 / pixels;

        (sad, avg_diff)
    }

    /// Store frame for next comparison.
    fn store_frame(&mut self, luma: &[u8], stride: usize) {
        let height = self.height as usize;
        let width = self.width as usize;

        let mut stored = vec![0u8; width * height];

        for y in 0..height {
            let src_start = y * stride;
            let dst_start = y * width;
            let copy_len = width.min(luma.len().saturating_sub(src_start));

            if copy_len > 0 {
                stored[dst_start..dst_start + copy_len]
                    .copy_from_slice(&luma[src_start..src_start + copy_len]);
            }
        }

        self.prev_luma = Some(stored);
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.prev_luma = None;
    }
}

impl Default for MotionAnalyzer {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

/// Motion analysis result.
#[derive(Clone, Copy, Debug, Default)]
pub struct MotionResult {
    /// Average SAD (Sum of Absolute Differences).
    pub average_sad: f32,
    /// Ratio of blocks with significant motion.
    pub motion_ratio: f32,
    /// Maximum motion detected in any block.
    pub max_motion: f32,
    /// Overall motion complexity metric.
    pub complexity: f32,
}

impl MotionResult {
    /// Check if this indicates a high motion frame.
    #[must_use]
    pub fn is_high_motion(&self) -> bool {
        self.motion_ratio > 0.5 || self.max_motion > 20.0
    }

    /// Check if this indicates a static/low motion frame.
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.motion_ratio < 0.1 && self.max_motion < 5.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    fn create_gradient_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                frame.push(((x + y) % 256) as u8);
            }
        }
        frame
    }

    #[test]
    fn test_complexity_estimator_creation() {
        let estimator = ComplexityEstimator::new(1920, 1080);
        assert_eq!(estimator.frame_count(), 0);
    }

    #[test]
    fn test_uniform_frame_low_complexity() {
        let mut estimator = ComplexityEstimator::new(64, 64);
        let frame = create_test_frame(64, 64, 128);

        let result = estimator.estimate(&frame, 64);

        // Uniform frame should have low spatial complexity
        assert!(result.spatial < 0.5);
    }

    #[test]
    fn test_gradient_frame_higher_complexity() {
        let mut estimator = ComplexityEstimator::new(64, 64);
        let frame = create_gradient_frame(64, 64);

        let result = estimator.estimate(&frame, 64);

        // Gradient has some texture, should have measurable complexity
        assert!(result.spatial > 0.0);
    }

    #[test]
    fn test_temporal_complexity_static() {
        let mut estimator = ComplexityEstimator::new(64, 64);
        let frame = create_test_frame(64, 64, 128);

        // First frame
        let _ = estimator.estimate(&frame, 64);

        // Same frame again - should have low temporal complexity
        let result = estimator.estimate(&frame, 64);
        assert!(result.temporal <= 1.0);
    }

    #[test]
    fn test_complexity_result_methods() {
        let high = ComplexityResult {
            spatial: 2.0,
            temporal: 2.0,
            combined: 2.0,
            normalized: 2.0,
        };
        assert!(high.is_high_complexity());
        assert!(!high.is_low_complexity());

        let low = ComplexityResult {
            spatial: 0.5,
            temporal: 0.5,
            combined: 0.5,
            normalized: 0.5,
        };
        assert!(low.is_low_complexity());
        assert!(!low.is_high_complexity());
    }

    #[test]
    fn test_suggested_qp_offset() {
        let high = ComplexityResult {
            spatial: 1.0,
            temporal: 1.0,
            combined: 1.0,
            normalized: 2.0,
        };
        assert!(high.suggested_qp_offset() > 0.0);

        let low = ComplexityResult {
            spatial: 1.0,
            temporal: 1.0,
            combined: 1.0,
            normalized: 0.5,
        };
        assert!(low.suggested_qp_offset() < 0.0);
    }

    #[test]
    fn test_motion_analyzer_creation() {
        let analyzer = MotionAnalyzer::new(1920, 1080);
        assert!(analyzer.prev_luma.is_none());
    }

    #[test]
    fn test_motion_analyzer_static() {
        let mut analyzer = MotionAnalyzer::new(64, 64);
        let frame = create_test_frame(64, 64, 128);

        // First frame
        let _ = analyzer.analyze(&frame, 64);

        // Same frame - no motion
        let result = analyzer.analyze(&frame, 64);
        assert!(result.is_static());
    }

    #[test]
    fn test_motion_analyzer_reset() {
        let mut analyzer = MotionAnalyzer::new(64, 64);
        let frame = create_test_frame(64, 64, 128);

        let _ = analyzer.analyze(&frame, 64);
        assert!(analyzer.prev_luma.is_some());

        analyzer.reset();
        assert!(analyzer.prev_luma.is_none());
    }

    #[test]
    fn test_estimator_reset() {
        let mut estimator = ComplexityEstimator::new(64, 64);
        let frame = create_test_frame(64, 64, 128);

        let _ = estimator.estimate(&frame, 64);
        assert_eq!(estimator.frame_count(), 1);

        estimator.reset();
        assert_eq!(estimator.frame_count(), 0);
    }

    #[test]
    fn test_running_averages() {
        let mut estimator = ComplexityEstimator::new(64, 64);
        let frame = create_gradient_frame(64, 64);

        for _ in 0..10 {
            let _ = estimator.estimate(&frame, 64);
        }

        // Averages should converge
        assert!(estimator.avg_spatial() > 0.0);
        assert!(estimator.avg_combined() > 0.0);
    }
}
