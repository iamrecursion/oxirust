//! Look-ahead buffer and frame analysis.
//!
//! The look-ahead system analyzes future frames to make better encoding
//! decisions for the current frame, including scene change detection and
//! adaptive quantization.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_arguments)]

use crate::frame::{FrameType, VideoFrame};
use crate::multipass::complexity::{ComplexityAnalyzer, FrameComplexity};
use std::collections::VecDeque;

/// Configuration for the look-ahead system.
#[derive(Clone, Debug)]
pub struct LookaheadConfig {
    /// Number of frames to look ahead (10-250).
    pub window_size: usize,
    /// Minimum keyframe interval.
    pub min_keyint: u32,
    /// Maximum keyframe interval.
    pub max_keyint: u32,
    /// Scene change threshold (0.0-1.0).
    pub scene_change_threshold: f64,
    /// Enable adaptive quantization.
    pub enable_aq: bool,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            window_size: 40,
            min_keyint: 10,
            max_keyint: 250,
            scene_change_threshold: 0.4,
            enable_aq: true,
        }
    }
}

impl LookaheadConfig {
    /// Create a new lookahead configuration.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.clamp(10, 250),
            ..Default::default()
        }
    }

    /// Set keyframe interval range.
    #[must_use]
    pub fn with_keyint_range(mut self, min: u32, max: u32) -> Self {
        self.min_keyint = min;
        self.max_keyint = max;
        self
    }

    /// Set scene change threshold.
    #[must_use]
    pub fn with_scene_threshold(mut self, threshold: f64) -> Self {
        self.scene_change_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

/// A frame stored in the lookahead buffer.
#[derive(Clone, Debug)]
pub struct LookaheadFrame {
    /// The actual video frame.
    pub frame: VideoFrame,
    /// Frame complexity analysis.
    pub complexity: FrameComplexity,
    /// Assigned frame type (may be updated during lookahead).
    pub assigned_type: FrameType,
    /// QP offset from base (for adaptive quantization).
    pub qp_offset: i32,
}

impl LookaheadFrame {
    /// Create a new lookahead frame.
    #[must_use]
    pub fn new(frame: VideoFrame, complexity: FrameComplexity) -> Self {
        Self {
            assigned_type: frame.frame_type,
            frame,
            complexity,
            qp_offset: 0,
        }
    }
}

/// Look-ahead buffer for analyzing future frames.
pub struct LookaheadBuffer {
    config: LookaheadConfig,
    buffer: VecDeque<LookaheadFrame>,
    complexity_analyzer: ComplexityAnalyzer,
    frames_since_keyframe: u32,
    total_frames_analyzed: u64,
}

impl LookaheadBuffer {
    /// Create a new lookahead buffer.
    #[must_use]
    pub fn new(config: LookaheadConfig, width: u32, height: u32) -> Self {
        let mut analyzer = ComplexityAnalyzer::new(width, height);
        analyzer.set_scene_change_threshold(config.scene_change_threshold);

        let window_size = config.window_size;

        Self {
            config,
            buffer: VecDeque::with_capacity(window_size),
            complexity_analyzer: analyzer,
            frames_since_keyframe: 0,
            total_frames_analyzed: 0,
        }
    }

    /// Add a frame to the lookahead buffer.
    pub fn add_frame(&mut self, frame: VideoFrame) {
        let complexity = self
            .complexity_analyzer
            .analyze(&frame, self.total_frames_analyzed);
        let mut lookahead_frame = LookaheadFrame::new(frame, complexity);

        // Detect scene changes and force keyframes
        if lookahead_frame.complexity.is_scene_change
            && self.frames_since_keyframe >= self.config.min_keyint
        {
            lookahead_frame.assigned_type = FrameType::Key;
            self.frames_since_keyframe = 0;
        } else if self.frames_since_keyframe >= self.config.max_keyint {
            // Force keyframe at max interval
            lookahead_frame.assigned_type = FrameType::Key;
            self.frames_since_keyframe = 0;
        } else {
            lookahead_frame.assigned_type = FrameType::Inter;
            self.frames_since_keyframe += 1;
        }

        // Calculate adaptive QP offset
        if self.config.enable_aq {
            lookahead_frame.qp_offset = self.calculate_aq_offset(&lookahead_frame);
        }

        self.buffer.push_back(lookahead_frame);
        self.total_frames_analyzed += 1;

        // Limit buffer size
        while self.buffer.len() > self.config.window_size {
            self.buffer.pop_front();
        }
    }

    /// Get the next frame to encode (removes from buffer).
    pub fn get_next_frame(&mut self) -> Option<LookaheadFrame> {
        self.buffer.pop_front()
    }

    /// Peek at the next frame without removing it.
    #[must_use]
    pub fn peek_next(&self) -> Option<&LookaheadFrame> {
        self.buffer.front()
    }

    /// Get number of frames in buffer.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.config.window_size
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Analyze future frames and get encoding recommendations.
    #[must_use]
    pub fn analyze_window(&self) -> LookaheadAnalysis {
        if self.buffer.is_empty() {
            return LookaheadAnalysis::default();
        }

        let mut analysis = LookaheadAnalysis::default();
        analysis.total_frames = self.buffer.len();

        // Analyze complexity distribution
        let complexities: Vec<f64> = self
            .buffer
            .iter()
            .map(|f| f.complexity.combined_complexity)
            .collect();

        analysis.avg_complexity = complexities.iter().sum::<f64>() / complexities.len() as f64;

        analysis.min_complexity = complexities
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        analysis.max_complexity = complexities
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Count scene changes
        analysis.scene_changes = self
            .buffer
            .iter()
            .filter(|f| f.complexity.is_scene_change)
            .count();

        // Count keyframes
        analysis.keyframes = self
            .buffer
            .iter()
            .filter(|f| f.assigned_type == FrameType::Key)
            .count();

        // Find next scene change distance
        analysis.next_scene_change = self
            .buffer
            .iter()
            .position(|f| f.complexity.is_scene_change)
            .map(|pos| pos as u32);

        // Calculate complexity variance
        let variance: f64 = complexities
            .iter()
            .map(|c| (c - analysis.avg_complexity).powi(2))
            .sum::<f64>()
            / complexities.len() as f64;
        analysis.complexity_variance = variance;

        analysis
    }

    /// Calculate adaptive QP offset based on frame complexity.
    fn calculate_aq_offset(&self, frame: &LookaheadFrame) -> i32 {
        // Calculate relative complexity compared to recent history
        let relative_complexity = if self.buffer.is_empty() {
            1.0
        } else {
            let avg_complexity: f64 = self
                .buffer
                .iter()
                .map(|f| f.complexity.combined_complexity)
                .sum::<f64>()
                / self.buffer.len() as f64;

            if avg_complexity > 0.01 {
                frame.complexity.combined_complexity / avg_complexity
            } else {
                1.0
            }
        };

        // Convert to QP offset (-10 to +10)
        // Lower complexity -> higher QP (worse quality, fewer bits)
        // Higher complexity -> lower QP (better quality, more bits)
        let offset = (10.0 * (1.0 - relative_complexity)).clamp(-10.0, 10.0);
        offset as i32
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &LookaheadConfig {
        &self.config
    }

    /// Reset the lookahead buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.complexity_analyzer.reset();
        self.frames_since_keyframe = 0;
        self.total_frames_analyzed = 0;
    }

    /// Flush remaining frames from the buffer.
    pub fn flush(&mut self) -> Vec<LookaheadFrame> {
        self.buffer.drain(..).collect()
    }
}

/// Analysis results from the lookahead window.
#[derive(Clone, Debug, Default)]
pub struct LookaheadAnalysis {
    /// Total frames in lookahead window.
    pub total_frames: usize,
    /// Average complexity across window.
    pub avg_complexity: f64,
    /// Minimum complexity in window.
    pub min_complexity: f64,
    /// Maximum complexity in window.
    pub max_complexity: f64,
    /// Variance in complexity.
    pub complexity_variance: f64,
    /// Number of scene changes detected.
    pub scene_changes: usize,
    /// Number of keyframes in window.
    pub keyframes: usize,
    /// Distance to next scene change (if any).
    pub next_scene_change: Option<u32>,
}

impl LookaheadAnalysis {
    /// Check if complexity is stable (low variance).
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.complexity_variance < 0.1
    }

    /// Get complexity range.
    #[must_use]
    pub fn complexity_range(&self) -> f64 {
        self.max_complexity - self.min_complexity
    }
}

/// Scene change detector for video frames.
pub struct SceneChangeDetector {
    threshold: f64,
    width: u32,
    height: u32,
    prev_frame: Option<Vec<u8>>,
}

impl SceneChangeDetector {
    /// Create a new scene change detector.
    #[must_use]
    pub fn new(width: u32, height: u32, threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            width,
            height,
            prev_frame: None,
        }
    }

    /// Detect if the current frame is a scene change.
    #[must_use]
    pub fn detect(&mut self, frame: &VideoFrame) -> bool {
        if let Some(luma_plane) = frame.planes.first() {
            let luma_data = luma_plane.data.as_ref();

            if let Some(prev) = &self.prev_frame {
                let sad = self.compute_sad(luma_data, prev);
                let pixels = (self.width as u64) * (self.height as u64);
                let avg_sad = if pixels > 0 {
                    sad as f64 / pixels as f64
                } else {
                    0.0
                };

                let normalized_sad = (avg_sad / 50.0).min(1.0);
                let is_scene_change = normalized_sad > self.threshold;

                self.prev_frame = Some(luma_data.to_vec());
                is_scene_change
            } else {
                self.prev_frame = Some(luma_data.to_vec());
                true // First frame is always a scene change
            }
        } else {
            false
        }
    }

    /// Compute Sum of Absolute Differences.
    fn compute_sad(&self, current: &[u8], previous: &[u8]) -> u64 {
        let mut sad = 0u64;
        let len = current.len().min(previous.len());

        for i in 0..len {
            let diff = (current[i] as i32 - previous[i] as i32).abs();
            sad += diff as u64;
        }

        sad
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.prev_frame = None;
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
    fn test_lookahead_config_new() {
        let config = LookaheadConfig::new(50);
        assert_eq!(config.window_size, 50);
        assert_eq!(config.max_keyint, 250);
    }

    #[test]
    fn test_lookahead_config_clamp() {
        let config = LookaheadConfig::new(5); // Below minimum
        assert_eq!(config.window_size, 10); // Clamped to min

        let config2 = LookaheadConfig::new(300); // Above maximum
        assert_eq!(config2.window_size, 250); // Clamped to max
    }

    #[test]
    fn test_lookahead_buffer_new() {
        let config = LookaheadConfig::new(40);
        let buffer = LookaheadBuffer::new(config, 1920, 1080);
        assert_eq!(buffer.buffer_size(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_lookahead_add_frame() {
        let config = LookaheadConfig::new(40);
        let mut buffer = LookaheadBuffer::new(config, 320, 240);

        let frame = create_test_frame(320, 240, 128);
        buffer.add_frame(frame);

        assert_eq!(buffer.buffer_size(), 1);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_lookahead_get_next_frame() {
        let config = LookaheadConfig::new(40);
        let mut buffer = LookaheadBuffer::new(config, 320, 240);

        let frame = create_test_frame(320, 240, 128);
        buffer.add_frame(frame);

        let next = buffer.get_next_frame();
        assert!(next.is_some());
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_lookahead_buffer_full() {
        let config = LookaheadConfig::new(10);
        let mut buffer = LookaheadBuffer::new(config, 320, 240);

        for i in 0..15 {
            let frame = create_test_frame(320, 240, i as u8);
            buffer.add_frame(frame);
        }

        // Buffer should be limited to window size
        assert_eq!(buffer.buffer_size(), 10);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_lookahead_analyze_window() {
        let config = LookaheadConfig::new(40);
        let mut buffer = LookaheadBuffer::new(config, 320, 240);

        for i in 0..10 {
            let frame = create_test_frame(320, 240, i as u8);
            buffer.add_frame(frame);
        }

        let analysis = buffer.analyze_window();
        assert_eq!(analysis.total_frames, 10);
        assert!(analysis.avg_complexity >= 0.0);
    }

    #[test]
    fn test_scene_change_detector() {
        let mut detector = SceneChangeDetector::new(320, 240, 0.4);

        let frame1 = create_test_frame(320, 240, 0);
        assert!(detector.detect(&frame1)); // First frame

        let frame2 = create_test_frame(320, 240, 255);
        assert!(detector.detect(&frame2)); // Big change

        let frame3 = create_test_frame(320, 240, 250);
        assert!(!detector.detect(&frame3)); // Small change
    }
}
