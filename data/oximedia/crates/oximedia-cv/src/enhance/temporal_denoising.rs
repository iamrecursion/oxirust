//! Temporal denoising coherence for video denoising.
//!
//! Provides frame-to-frame consistent denoising by maintaining a temporal
//! buffer and using motion-compensated blending to reduce flickering artifacts.
//!
//! # Features
//!
//! - Motion-compensated temporal blending
//! - Configurable buffer depth for temporal window
//! - Per-pixel motion estimation using block matching
//! - Adaptive blending weights based on motion magnitude
//! - Exponential moving average (EMA) temporal filter
//!
//! # Example
//!
//! ```
//! use oximedia_cv::enhance::temporal_denoising::{TemporalDenoiser, TemporalDenoisingConfig};
//!
//! let config = TemporalDenoisingConfig::default();
//! let mut denoiser = TemporalDenoiser::new(config);
//!
//! let frame1 = vec![128u8; 64 * 64 * 3];
//! let result1 = denoiser.process_frame(&frame1, 64, 64).expect("denoising should succeed");
//!
//! let frame2 = vec![130u8; 64 * 64 * 3];
//! let result2 = denoiser.process_frame(&frame2, 64, 64).expect("denoising should succeed");
//! ```

use crate::error::{CvError, CvResult};

/// Configuration for temporal denoising.
#[derive(Debug, Clone)]
pub struct TemporalDenoisingConfig {
    /// Number of frames to keep in the temporal buffer.
    pub buffer_depth: usize,
    /// Temporal blending strength (0.0 = no temporal, 1.0 = max temporal).
    pub temporal_strength: f64,
    /// Block size for motion estimation (pixels).
    pub block_size: usize,
    /// Search range for motion estimation (pixels).
    pub search_range: usize,
    /// Motion threshold: pixels with motion above this are blended less.
    pub motion_threshold: f64,
    /// Minimum weight for temporal blending (even for high-motion areas).
    pub min_temporal_weight: f64,
    /// Enable motion compensation.
    pub enable_motion_compensation: bool,
    /// Noise floor: pixel differences below this are considered noise.
    pub noise_floor: f64,
}

impl Default for TemporalDenoisingConfig {
    fn default() -> Self {
        Self {
            buffer_depth: 5,
            temporal_strength: 0.7,
            block_size: 8,
            search_range: 16,
            motion_threshold: 30.0,
            min_temporal_weight: 0.1,
            enable_motion_compensation: true,
            noise_floor: 10.0,
        }
    }
}

impl TemporalDenoisingConfig {
    /// Create a new temporal denoising configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set buffer depth.
    #[must_use]
    pub const fn with_buffer_depth(mut self, depth: usize) -> Self {
        self.buffer_depth = depth;
        self
    }

    /// Set temporal blending strength.
    #[must_use]
    pub fn with_temporal_strength(mut self, strength: f64) -> Self {
        self.temporal_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set block size for motion estimation.
    #[must_use]
    pub const fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Set search range for motion estimation.
    #[must_use]
    pub const fn with_search_range(mut self, range: usize) -> Self {
        self.search_range = range;
        self
    }

    /// Set motion threshold.
    #[must_use]
    pub fn with_motion_threshold(mut self, threshold: f64) -> Self {
        self.motion_threshold = threshold.max(1.0);
        self
    }

    /// Enable or disable motion compensation.
    #[must_use]
    pub const fn with_motion_compensation(mut self, enable: bool) -> Self {
        self.enable_motion_compensation = enable;
        self
    }

    /// Set noise floor.
    #[must_use]
    pub fn with_noise_floor(mut self, floor: f64) -> Self {
        self.noise_floor = floor.max(0.0);
        self
    }
}

/// Stored frame in the temporal buffer.
#[derive(Debug, Clone)]
struct TemporalFrame {
    /// RGB pixel data.
    data: Vec<u8>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
}

/// Motion vector for a block.
#[derive(Debug, Clone, Copy)]
struct MotionVector {
    /// Horizontal displacement.
    dx: i32,
    /// Vertical displacement.
    dy: i32,
    /// Sum of absolute differences (match cost).
    sad: u64,
}

/// Temporal denoiser for video frame sequences.
///
/// Maintains a buffer of recent frames and uses motion-compensated
/// temporal blending to produce temporally consistent output.
pub struct TemporalDenoiser {
    /// Configuration.
    config: TemporalDenoisingConfig,
    /// Ring buffer of recent frames.
    frame_buffer: Vec<TemporalFrame>,
    /// Accumulated result (EMA).
    accumulator: Option<Vec<f64>>,
    /// Frame count processed.
    frame_count: u64,
}

impl TemporalDenoiser {
    /// Create a new temporal denoiser.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::temporal_denoising::{TemporalDenoiser, TemporalDenoisingConfig};
    ///
    /// let denoiser = TemporalDenoiser::new(TemporalDenoisingConfig::default());
    /// ```
    #[must_use]
    pub fn new(config: TemporalDenoisingConfig) -> Self {
        Self {
            config,
            frame_buffer: Vec::new(),
            accumulator: None,
            frame_count: 0,
        }
    }

    /// Process a new frame with temporal coherence.
    ///
    /// The first frame is returned as-is. Subsequent frames are blended
    /// with the temporal buffer to reduce noise while maintaining consistency.
    ///
    /// # Arguments
    ///
    /// * `frame` - RGB image data
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or data is insufficient.
    pub fn process_frame(&mut self, frame: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected = (width as usize) * (height as usize) * 3;
        if frame.len() != expected {
            return Err(CvError::insufficient_data(expected, frame.len()));
        }

        let result = if self.frame_buffer.is_empty() {
            // First frame: initialize accumulator
            self.accumulator = Some(frame.iter().map(|&v| v as f64).collect());
            frame.to_vec()
        } else {
            // Compute motion-compensated temporal blend
            self.temporal_blend(frame, width, height)?
        };

        // Add frame to buffer
        self.push_frame(frame, width, height);
        self.frame_count += 1;

        Ok(result)
    }

    /// Push a frame into the ring buffer.
    fn push_frame(&mut self, data: &[u8], width: u32, height: u32) {
        let frame = TemporalFrame {
            data: data.to_vec(),
            width,
            height,
        };

        if self.frame_buffer.len() >= self.config.buffer_depth {
            self.frame_buffer.remove(0);
        }
        self.frame_buffer.push(frame);
    }

    /// Perform motion-compensated temporal blending.
    fn temporal_blend(&mut self, current: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        let prev = self
            .frame_buffer
            .last()
            .ok_or_else(|| CvError::tracking_error("No previous frame in buffer"))?;

        // Compute per-pixel motion weights
        let motion_weights = if self.config.enable_motion_compensation
            && prev.width == width
            && prev.height == height
        {
            self.estimate_motion_weights(&prev.data, current, width, height)
        } else {
            vec![1.0; (width * height) as usize]
        };

        // Temporal EMA blending
        let accumulator = self
            .accumulator
            .as_ref()
            .ok_or_else(|| CvError::tracking_error("Accumulator not initialized"))?;

        let pixel_count = (width as usize) * (height as usize);
        let strength = self.config.temporal_strength;
        let min_weight = self.config.min_temporal_weight;

        let mut new_accumulator = vec![0.0; pixel_count * 3];
        let mut output = vec![0u8; pixel_count * 3];

        for i in 0..pixel_count {
            let motion_w = motion_weights[i];
            // Adaptive temporal weight: less blending for high-motion areas
            let temporal_w = (strength * motion_w).max(min_weight);

            for c in 0..3 {
                let idx = i * 3 + c;
                let curr_val = current[idx] as f64;
                let acc_val = accumulator[idx];

                // EMA: output = alpha * current + (1 - alpha) * accumulated
                let blended = temporal_w * acc_val + (1.0 - temporal_w) * curr_val;
                new_accumulator[idx] = blended;
                output[idx] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }

        self.accumulator = Some(new_accumulator);
        Ok(output)
    }

    /// Estimate per-pixel motion weights using block matching.
    ///
    /// Returns a weight map where 1.0 = static (high temporal blending)
    /// and 0.0 = fast motion (low temporal blending).
    fn estimate_motion_weights(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<f64> {
        let w = width as usize;
        let h = height as usize;
        let block = self.config.block_size.max(2);
        let search = self.config.search_range as i32;
        let threshold = self.config.motion_threshold;
        let noise_floor = self.config.noise_floor;

        let blocks_x = (w + block - 1) / block;
        let blocks_y = (h + block - 1) / block;

        // Compute motion vectors per block
        let mut block_motions = vec![
            MotionVector {
                dx: 0,
                dy: 0,
                sad: 0,
            };
            blocks_x * blocks_y
        ];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let bx0 = bx * block;
                let by0 = by * block;
                let bw = block.min(w - bx0);
                let bh = block.min(h - by0);

                let mv = self.block_match_sad(prev, curr, w, h, bx0, by0, bw, bh, search);
                block_motions[by * blocks_x + bx] = mv;
            }
        }

        // Expand block motions to per-pixel weights
        let mut weights = vec![1.0; w * h];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mv = &block_motions[by * blocks_x + bx];
                let motion_magnitude = ((mv.dx * mv.dx + mv.dy * mv.dy) as f64).sqrt();

                // Also consider SAD as motion indicator
                let block_pixels = (block * block) as f64;
                let avg_sad = mv.sad as f64 / block_pixels.max(1.0) / 3.0;

                // Combined motion score
                let motion_score = motion_magnitude + (avg_sad / noise_floor.max(1.0));

                // Convert to weight: high motion -> low weight
                let weight = if motion_score > threshold {
                    0.0
                } else {
                    1.0 - (motion_score / threshold).clamp(0.0, 1.0)
                };

                // Apply to pixels in this block
                let bx0 = bx * block;
                let by0 = by * block;
                let bw = block.min(w - bx0);
                let bh = block.min(h - by0);

                for dy in 0..bh {
                    for dx in 0..bw {
                        weights[(by0 + dy) * w + (bx0 + dx)] = weight;
                    }
                }
            }
        }

        weights
    }

    /// Block matching using Sum of Absolute Differences.
    #[allow(clippy::too_many_arguments)]
    fn block_match_sad(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
        bw: usize,
        bh: usize,
        search_range: i32,
    ) -> MotionVector {
        // Initialize with zero-motion (0,0) candidate to prefer it when SADs are equal
        let zero_sad = self.compute_sad(curr, prev, width, bx, by, bx, by, bw, bh);
        let mut best = MotionVector {
            dx: 0,
            dy: 0,
            sad: zero_sad,
        };

        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                if dx == 0 && dy == 0 {
                    continue; // Already evaluated above
                }
                let ref_x = bx as i32 + dx;
                let ref_y = by as i32 + dy;

                if ref_x < 0
                    || ref_y < 0
                    || (ref_x as usize + bw) > width
                    || (ref_y as usize + bh) > height
                {
                    continue;
                }

                let sad = self.compute_sad(
                    curr,
                    prev,
                    width,
                    bx,
                    by,
                    ref_x as usize,
                    ref_y as usize,
                    bw,
                    bh,
                );

                if sad < best.sad {
                    best = MotionVector { dx, dy, sad };
                }
            }
        }

        best
    }

    /// Compute SAD between two blocks.
    #[allow(clippy::too_many_arguments)]
    fn compute_sad(
        &self,
        img1: &[u8],
        img2: &[u8],
        width: usize,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        bw: usize,
        bh: usize,
    ) -> u64 {
        let mut sad: u64 = 0;

        for dy in 0..bh {
            for dx in 0..bw {
                let idx1 = ((y1 + dy) * width + (x1 + dx)) * 3;
                let idx2 = ((y2 + dy) * width + (x2 + dx)) * 3;

                if idx1 + 2 < img1.len() && idx2 + 2 < img2.len() {
                    for c in 0..3 {
                        sad += (img1[idx1 + c] as i64 - img2[idx2 + c] as i64).unsigned_abs();
                    }
                }
            }
        }

        sad
    }

    /// Get the number of frames processed.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the current buffer depth.
    #[must_use]
    pub fn current_buffer_size(&self) -> usize {
        self.frame_buffer.len()
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &TemporalDenoisingConfig {
        &self.config
    }

    /// Reset the denoiser, clearing all buffers.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.accumulator = None;
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_denoising_config_default() {
        let config = TemporalDenoisingConfig::default();
        assert_eq!(config.buffer_depth, 5);
        assert_eq!(config.temporal_strength, 0.7);
        assert_eq!(config.block_size, 8);
        assert_eq!(config.search_range, 16);
        assert!(config.enable_motion_compensation);
    }

    #[test]
    fn test_temporal_denoising_config_builder() {
        let config = TemporalDenoisingConfig::new()
            .with_buffer_depth(10)
            .with_temporal_strength(0.5)
            .with_block_size(16)
            .with_search_range(32)
            .with_motion_threshold(50.0)
            .with_motion_compensation(false)
            .with_noise_floor(5.0);

        assert_eq!(config.buffer_depth, 10);
        assert_eq!(config.temporal_strength, 0.5);
        assert_eq!(config.block_size, 16);
        assert_eq!(config.search_range, 32);
        assert_eq!(config.motion_threshold, 50.0);
        assert!(!config.enable_motion_compensation);
        assert_eq!(config.noise_floor, 5.0);
    }

    #[test]
    fn test_temporal_denoiser_new() {
        let denoiser = TemporalDenoiser::new(TemporalDenoisingConfig::default());
        assert_eq!(denoiser.frame_count(), 0);
        assert_eq!(denoiser.current_buffer_size(), 0);
    }

    #[test]
    fn test_process_first_frame() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoisingConfig::default());
        let frame = vec![128u8; 32 * 32 * 3];
        let result = denoiser.process_frame(&frame, 32, 32);
        assert!(result.is_ok());
        let output = result.expect("should succeed");
        // First frame should be returned as-is
        assert_eq!(output, frame);
        assert_eq!(denoiser.frame_count(), 1);
        assert_eq!(denoiser.current_buffer_size(), 1);
    }

    #[test]
    fn test_process_multiple_frames() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoisingConfig::default());
        let w = 32u32;
        let h = 32u32;

        for i in 0..10 {
            let val = (128 + i) as u8;
            let frame = vec![val; (w * h * 3) as usize];
            let result = denoiser.process_frame(&frame, w, h);
            assert!(result.is_ok());
        }

        assert_eq!(denoiser.frame_count(), 10);
        assert_eq!(denoiser.current_buffer_size(), 5); // buffer_depth = 5
    }

    #[test]
    fn test_temporal_blending_reduces_noise() {
        let config = TemporalDenoisingConfig::new()
            .with_temporal_strength(0.8)
            .with_motion_compensation(false);
        let mut denoiser = TemporalDenoiser::new(config);
        let w = 16u32;
        let h = 16u32;

        // Process several frames with slight noise
        let base_val = 128u8;
        let noise_vals = [130u8, 126u8, 132u8, 124u8, 128u8];

        let first_frame = vec![base_val; (w * h * 3) as usize];
        let _ = denoiser.process_frame(&first_frame, w, h);

        let mut last_output = first_frame.clone();
        for &nv in &noise_vals {
            let noisy = vec![nv; (w * h * 3) as usize];
            let output = denoiser
                .process_frame(&noisy, w, h)
                .expect("should succeed");
            last_output = output;
        }

        // The temporal average should be close to the base value
        let avg_val = last_output.iter().map(|&v| v as f64).sum::<f64>() / last_output.len() as f64;
        // Should be within a reasonable range of the mean input
        assert!(
            (avg_val - 128.0).abs() < 10.0,
            "avg_val={avg_val} should be near 128"
        );
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoisingConfig::default());
        let result = denoiser.process_frame(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoisingConfig::default());
        let result = denoiser.process_frame(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoisingConfig::default());
        let frame = vec![128u8; 32 * 32 * 3];
        let _ = denoiser.process_frame(&frame, 32, 32);
        let _ = denoiser.process_frame(&frame, 32, 32);

        denoiser.reset();
        assert_eq!(denoiser.frame_count(), 0);
        assert_eq!(denoiser.current_buffer_size(), 0);
    }

    #[test]
    fn test_static_scene_high_temporal_weight() {
        let config = TemporalDenoisingConfig::new()
            .with_temporal_strength(0.9)
            .with_motion_compensation(true)
            .with_block_size(4)
            .with_search_range(2);
        let mut denoiser = TemporalDenoiser::new(config);
        let w = 16u32;
        let h = 16u32;
        let frame = vec![100u8; (w * h * 3) as usize];

        // Process identical frames
        let _ = denoiser.process_frame(&frame, w, h);
        let output = denoiser
            .process_frame(&frame, w, h)
            .expect("should succeed");

        // Static scene: output should match input closely
        for (a, b) in output.iter().zip(frame.iter()) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "Static scene should preserve values"
            );
        }
    }

    #[test]
    fn test_motion_estimation_static() {
        let config = TemporalDenoisingConfig::new()
            .with_block_size(4)
            .with_search_range(4);
        let denoiser = TemporalDenoiser::new(config);

        let frame = vec![100u8; 32 * 32 * 3];
        let weights = denoiser.estimate_motion_weights(&frame, &frame, 32, 32);

        // Static scene: all weights should be 1.0 (no motion)
        for &w in &weights {
            assert!(
                (w - 1.0).abs() < 0.01,
                "Static scene should have weight 1.0, got {w}"
            );
        }
    }

    #[test]
    fn test_motion_estimation_moving() {
        let config = TemporalDenoisingConfig::new()
            .with_block_size(4)
            .with_search_range(4)
            .with_motion_threshold(10.0);
        let denoiser = TemporalDenoiser::new(config);

        let w = 32u32;
        let h = 32u32;
        let frame1 = vec![100u8; (w * h * 3) as usize];
        let mut frame2 = vec![100u8; (w * h * 3) as usize];
        // Add a large bright region in frame2 that wasn't in frame1
        for y in 0..16u32 {
            for x in 0..16u32 {
                let idx = ((y * w + x) * 3) as usize;
                frame2[idx] = 255;
                frame2[idx + 1] = 255;
                frame2[idx + 2] = 255;
            }
        }

        let weights = denoiser.estimate_motion_weights(&frame1, &frame2, w, h);

        // Moving region should have lower weights
        let moving_weight = weights[0]; // top-left corner, in the changed region
        let static_weight = weights[(20 * w as usize) + 20]; // in the unchanged region
        assert!(
            moving_weight < static_weight,
            "Moving area weight ({moving_weight}) should be less than static ({static_weight})"
        );
    }

    #[test]
    fn test_buffer_depth_limit() {
        let config = TemporalDenoisingConfig::new().with_buffer_depth(3);
        let mut denoiser = TemporalDenoiser::new(config);
        let frame = vec![128u8; 16 * 16 * 3];

        for _ in 0..10 {
            let _ = denoiser.process_frame(&frame, 16, 16);
        }

        assert_eq!(denoiser.current_buffer_size(), 3);
    }

    #[test]
    fn test_compute_sad_identical() {
        let config = TemporalDenoisingConfig::default();
        let denoiser = TemporalDenoiser::new(config);

        let frame = vec![100u8; 16 * 16 * 3];
        let sad = denoiser.compute_sad(&frame, &frame, 16, 0, 0, 0, 0, 4, 4);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_compute_sad_different() {
        let config = TemporalDenoisingConfig::default();
        let denoiser = TemporalDenoiser::new(config);

        let frame1 = vec![100u8; 16 * 16 * 3];
        let frame2 = vec![110u8; 16 * 16 * 3];
        let sad = denoiser.compute_sad(&frame1, &frame2, 16, 0, 0, 0, 0, 4, 4);
        // 4x4 block, 3 channels, each diff = 10
        assert_eq!(sad, 4 * 4 * 3 * 10);
    }

    #[test]
    fn test_temporal_strength_zero() {
        let config = TemporalDenoisingConfig::new()
            .with_temporal_strength(0.0)
            .with_motion_compensation(false);
        let mut denoiser = TemporalDenoiser::new(config);
        let w = 16u32;
        let h = 16u32;

        let frame1 = vec![100u8; (w * h * 3) as usize];
        let _ = denoiser.process_frame(&frame1, w, h);

        let frame2 = vec![200u8; (w * h * 3) as usize];
        let output = denoiser
            .process_frame(&frame2, w, h)
            .expect("should succeed");

        // With strength=0 and min_weight=0.1, output should be mostly frame2
        let avg: f64 = output.iter().map(|&v| v as f64).sum::<f64>() / output.len() as f64;
        assert!(
            avg > 150.0,
            "With low temporal, output should be close to current frame"
        );
    }
}
