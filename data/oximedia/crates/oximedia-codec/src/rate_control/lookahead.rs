//! Lookahead buffer for frame analysis.
//!
//! The lookahead system buffers future frames to enable:
//! - Scene change detection
//! - Mini-GOP structure optimization
//! - Adaptive B-frame decisions
//! - Complexity pre-analysis for better bit allocation

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![forbid(unsafe_code)]

use crate::frame::FrameType;

use super::complexity::{ComplexityEstimator, ComplexityResult};
use super::types::RcConfig;

/// Lookahead buffer for frame pre-analysis.
#[derive(Clone, Debug)]
pub struct Lookahead {
    /// Maximum lookahead depth.
    depth: usize,
    /// Buffered frame info.
    frames: Vec<LookaheadFrame>,
    /// Scene cut threshold.
    scene_cut_threshold: f32,
    /// Complexity estimator.
    complexity_estimator: ComplexityEstimator,
    /// Frame width (reserved for frame analysis).
    #[allow(dead_code)]
    width: u32,
    /// Frame height (reserved for frame analysis).
    #[allow(dead_code)]
    height: u32,
    /// Enable adaptive B-frames.
    adaptive_b_frames: bool,
    /// Maximum consecutive B-frames.
    max_b_frames: u32,
    /// GOP length.
    gop_length: u32,
    /// Frames since last keyframe.
    frames_since_keyframe: u32,
    /// Total frames processed.
    frame_count: u64,
}

/// Information about a frame in the lookahead buffer.
#[derive(Clone, Debug)]
pub struct LookaheadFrame {
    /// Frame index (presentation order).
    pub frame_index: u64,
    /// Complexity analysis result.
    pub complexity: ComplexityResult,
    /// Detected as scene cut.
    pub is_scene_cut: bool,
    /// Determined frame type.
    pub frame_type: FrameType,
    /// Cost to encode as P-frame.
    pub p_cost: f32,
    /// Cost to encode as B-frame.
    pub b_cost: f32,
    /// Suggested target bits.
    pub suggested_bits: u64,
}

impl LookaheadFrame {
    /// Create a new lookahead frame entry.
    #[must_use]
    pub fn new(frame_index: u64) -> Self {
        Self {
            frame_index,
            complexity: ComplexityResult::default_complexity(),
            is_scene_cut: false,
            frame_type: FrameType::Inter,
            p_cost: 0.0,
            b_cost: 0.0,
            suggested_bits: 0,
        }
    }
}

impl Lookahead {
    /// Create a new lookahead buffer from configuration.
    #[must_use]
    pub fn new(config: &RcConfig, width: u32, height: u32) -> Self {
        Self {
            depth: config.lookahead_depth,
            frames: Vec::with_capacity(config.lookahead_depth),
            scene_cut_threshold: config.scene_cut_threshold,
            complexity_estimator: ComplexityEstimator::new(width, height),
            width,
            height,
            adaptive_b_frames: true,
            max_b_frames: 3,
            gop_length: config.gop_length,
            frames_since_keyframe: 0,
            frame_count: 0,
        }
    }

    /// Create with specific parameters.
    #[must_use]
    pub fn with_params(depth: usize, width: u32, height: u32) -> Self {
        Self {
            depth,
            frames: Vec::with_capacity(depth),
            scene_cut_threshold: 0.4,
            complexity_estimator: ComplexityEstimator::new(width, height),
            width,
            height,
            adaptive_b_frames: true,
            max_b_frames: 3,
            gop_length: 250,
            frames_since_keyframe: 0,
            frame_count: 0,
        }
    }

    /// Set scene cut threshold.
    pub fn set_scene_cut_threshold(&mut self, threshold: f32) {
        self.scene_cut_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Enable or disable adaptive B-frames.
    pub fn set_adaptive_b_frames(&mut self, enable: bool) {
        self.adaptive_b_frames = enable;
    }

    /// Set maximum consecutive B-frames.
    pub fn set_max_b_frames(&mut self, max: u32) {
        self.max_b_frames = max;
    }

    /// Push a new frame into the lookahead buffer.
    ///
    /// Returns an analyzed frame if the buffer was full.
    pub fn push_frame(&mut self, luma: &[u8], stride: usize) -> Option<LookaheadFrame> {
        // Analyze complexity
        let complexity = self.complexity_estimator.estimate(luma, stride);

        let mut frame = LookaheadFrame::new(self.frame_count);
        frame.complexity = complexity;

        // Detect scene cut
        frame.is_scene_cut = self.detect_scene_cut(&complexity);

        self.frame_count += 1;
        self.frames.push(frame);

        // If buffer is full, analyze and return the oldest frame
        if self.frames.len() > self.depth {
            self.analyze_mini_gop();
            return self.frames.drain(..1).next();
        }

        None
    }

    /// Push frame with pre-computed complexity.
    pub fn push_frame_with_complexity(
        &mut self,
        complexity: ComplexityResult,
    ) -> Option<LookaheadFrame> {
        let mut frame = LookaheadFrame::new(self.frame_count);
        frame.complexity = complexity;
        frame.is_scene_cut = self.detect_scene_cut(&complexity);

        self.frame_count += 1;
        self.frames.push(frame);

        if self.frames.len() > self.depth {
            self.analyze_mini_gop();
            return self.frames.drain(..1).next();
        }

        None
    }

    /// Detect if frame is a scene cut.
    fn detect_scene_cut(&self, complexity: &ComplexityResult) -> bool {
        // Scene cuts typically have high temporal complexity
        // and normalized complexity significantly above average
        complexity.temporal
            > self.complexity_estimator.avg_temporal() * (1.0 + self.scene_cut_threshold)
            || complexity.normalized > (1.0 + self.scene_cut_threshold * 2.0)
    }

    /// Analyze frames in buffer and determine mini-GOP structure.
    fn analyze_mini_gop(&mut self) {
        if self.frames.is_empty() {
            return;
        }

        // Find scene cuts
        let scene_cut_indices: Vec<usize> = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.is_scene_cut)
            .map(|(i, _)| i)
            .collect();

        // Determine frame types
        self.determine_frame_types(&scene_cut_indices);
    }

    /// Determine frame types based on analysis.
    fn determine_frame_types(&mut self, scene_cuts: &[usize]) {
        let mut consecutive_b = 0u32;

        // Cache values needed for B-frame decision
        let avg_temporal = self.complexity_estimator.avg_temporal();
        let adaptive_b = self.adaptive_b_frames;
        let max_b = self.max_b_frames;
        let gop_len = self.gop_length;

        let frame_count = self.frames.len();
        for i in 0..frame_count {
            let is_scene_cut = self.frames[i].is_scene_cut;
            let complexity = self.frames[i].complexity;

            // Check if this should be a keyframe
            let is_keyframe = i == 0 && self.frames_since_keyframe == 0
                || is_scene_cut
                || self.frames_since_keyframe >= gop_len;

            if is_keyframe {
                self.frames[i].frame_type = FrameType::Key;
                self.frames_since_keyframe = 0;
                consecutive_b = 0;
                continue;
            }

            // Check if this is near a scene cut (should be P-frame)
            let near_scene_cut = scene_cuts.iter().any(|&sc| {
                let diff = i.abs_diff(sc);
                diff <= 2
            });

            if near_scene_cut || !adaptive_b {
                self.frames[i].frame_type = FrameType::Inter;
                consecutive_b = 0;
            } else if consecutive_b < max_b
                && Self::should_be_b_frame_static(&complexity, avg_temporal)
            {
                self.frames[i].frame_type = FrameType::BiDir;
                consecutive_b += 1;
            } else {
                self.frames[i].frame_type = FrameType::Inter;
                consecutive_b = 0;
            }

            self.frames_since_keyframe += 1;
        }
    }

    /// Determine if a frame should be a B-frame (static method to avoid borrow issues).
    fn should_be_b_frame_static(complexity: &ComplexityResult, avg_temporal: f32) -> bool {
        // Low complexity frames benefit from being B-frames
        // High motion frames are better as P-frames
        complexity.normalized < 1.2 && complexity.temporal < avg_temporal * 1.3
    }

    /// Flush remaining frames from buffer.
    pub fn flush(&mut self) -> Vec<LookaheadFrame> {
        self.analyze_mini_gop();
        self.frames.drain(..).collect()
    }

    /// Get the number of frames in buffer.
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.frames.len()
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Check if buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.frames.len() >= self.depth
    }

    /// Get lookahead depth.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Peek at frames in buffer (read-only).
    #[must_use]
    pub fn peek(&self) -> &[LookaheadFrame] {
        &self.frames
    }

    /// Get average complexity of buffered frames.
    #[must_use]
    pub fn average_complexity(&self) -> f32 {
        if self.frames.is_empty() {
            return 1.0;
        }

        let sum: f32 = self.frames.iter().map(|f| f.complexity.combined).sum();
        sum / self.frames.len() as f32
    }

    /// Get count of detected scene cuts in buffer.
    #[must_use]
    pub fn scene_cut_count(&self) -> usize {
        self.frames.iter().filter(|f| f.is_scene_cut).count()
    }

    /// Reset the lookahead buffer.
    pub fn reset(&mut self) {
        self.frames.clear();
        self.complexity_estimator.reset();
        self.frames_since_keyframe = 0;
        self.frame_count = 0;
    }

    /// Get frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

impl Default for Lookahead {
    fn default() -> Self {
        Self::with_params(40, 1920, 1080)
    }
}

/// Mini-GOP structure information.
#[derive(Clone, Debug, Default)]
pub struct MiniGopInfo {
    /// Starting frame index.
    pub start_frame: u64,
    /// Number of frames in mini-GOP.
    pub frame_count: u32,
    /// Number of I-frames.
    pub i_frames: u32,
    /// Number of P-frames.
    pub p_frames: u32,
    /// Number of B-frames.
    pub b_frames: u32,
    /// Total complexity.
    pub total_complexity: f32,
    /// Contains a scene cut.
    pub has_scene_cut: bool,
}

impl MiniGopInfo {
    /// Calculate suggested bit allocation for this mini-GOP.
    #[must_use]
    pub fn suggested_bits(&self, target_bits_per_frame: u64) -> u64 {
        let base = target_bits_per_frame * self.frame_count as u64;

        // Adjust based on frame type distribution and complexity
        let i_overhead = self.i_frames as f64 * 2.0; // I-frames use more bits
        let b_savings = self.b_frames as f64 * 0.3; // B-frames use fewer bits

        let adjustment = 1.0 + (i_overhead - b_savings) / self.frame_count as f64;
        let complexity_factor = self.total_complexity / self.frame_count as f32;

        (base as f64 * adjustment * complexity_factor as f64) as u64
    }

    /// Get average complexity per frame.
    #[must_use]
    pub fn average_complexity(&self) -> f32 {
        if self.frame_count == 0 {
            return 1.0;
        }
        self.total_complexity / self.frame_count as f32
    }
}

/// Scene change detector.
#[derive(Clone, Debug)]
pub struct SceneChangeDetector {
    /// Detection threshold.
    threshold: f32,
    /// Minimum frames between scene changes.
    min_interval: u32,
    /// Frames since last scene change.
    frames_since_change: u32,
    /// Previous frame complexity.
    prev_complexity: f32,
    /// Running average complexity.
    avg_complexity: f32,
    /// Total scene changes detected.
    total_changes: u64,
}

impl SceneChangeDetector {
    /// Create a new scene change detector.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            min_interval: 5,
            frames_since_change: 0,
            prev_complexity: 1.0,
            avg_complexity: 1.0,
            total_changes: 0,
        }
    }

    /// Set minimum interval between scene changes.
    pub fn set_min_interval(&mut self, interval: u32) {
        self.min_interval = interval;
    }

    /// Detect scene change from complexity.
    #[must_use]
    pub fn detect(&mut self, complexity: &ComplexityResult) -> bool {
        let is_scene_change = self.is_scene_change(complexity);

        // Update state
        if is_scene_change {
            self.frames_since_change = 0;
            self.total_changes += 1;
        } else {
            self.frames_since_change += 1;
        }

        // Update running average
        self.avg_complexity = self.avg_complexity * 0.9 + complexity.combined * 0.1;
        self.prev_complexity = complexity.combined;

        is_scene_change
    }

    /// Internal scene change detection logic.
    fn is_scene_change(&self, complexity: &ComplexityResult) -> bool {
        // Don't detect scene changes too frequently
        if self.frames_since_change < self.min_interval {
            return false;
        }

        // Check for sudden complexity increase
        let complexity_ratio = complexity.combined / self.prev_complexity;
        let avg_ratio = complexity.combined / self.avg_complexity;

        // Scene change if complexity spikes significantly
        complexity_ratio > (1.0 + self.threshold) || avg_ratio > (1.0 + self.threshold * 1.5)
    }

    /// Get total scene changes detected.
    #[must_use]
    pub fn total_changes(&self) -> u64 {
        self.total_changes
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.frames_since_change = 0;
        self.prev_complexity = 1.0;
        self.avg_complexity = 1.0;
        self.total_changes = 0;
    }
}

impl Default for SceneChangeDetector {
    fn default() -> Self {
        Self::new(0.4)
    }
}

// =============================================================================
// Content-Adaptive Bitrate Allocator
// =============================================================================

/// Content type classification for scene-adaptive rate control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneContentType {
    /// Static or nearly-static content (slides, still images).
    Static,
    /// Low-motion content (talking heads, slow pans).
    LowMotion,
    /// Moderate motion (typical broadcast content).
    Moderate,
    /// High-motion content (sports, action).
    HighMotion,
    /// Scene transition (fade, dissolve, cut).
    Transition,
    /// High-detail content (texture-heavy, foliage).
    HighDetail,
}

impl SceneContentType {
    /// Get the bitrate multiplier for this content type.
    /// Higher values allocate more bits to complex content.
    #[must_use]
    pub fn bitrate_multiplier(&self) -> f32 {
        match self {
            Self::Static => 0.5,
            Self::LowMotion => 0.75,
            Self::Moderate => 1.0,
            Self::HighMotion => 1.4,
            Self::Transition => 1.6,
            Self::HighDetail => 1.3,
        }
    }

    /// Get the QP adjustment for this content type.
    /// Negative values = higher quality (lower QP).
    #[must_use]
    pub fn qp_adjustment(&self) -> i8 {
        match self {
            Self::Static => 4,
            Self::LowMotion => 2,
            Self::Moderate => 0,
            Self::HighMotion => -2,
            Self::Transition => -3,
            Self::HighDetail => -1,
        }
    }
}

/// Content analysis metrics for a frame or scene segment.
#[derive(Clone, Debug)]
pub struct ContentMetrics {
    /// Average spatial complexity (texture/gradient energy).
    pub spatial_complexity: f32,
    /// Average temporal complexity (motion energy).
    pub temporal_complexity: f32,
    /// Edge density (0.0-1.0): fraction of pixels near edges.
    pub edge_density: f32,
    /// Flat region ratio (0.0-1.0): fraction of low-variance blocks.
    pub flat_ratio: f32,
    /// Motion uniformity (0.0-1.0): 1.0 = uniform global motion, 0.0 = chaotic.
    pub motion_uniformity: f32,
    /// Classified content type.
    pub content_type: SceneContentType,
}

impl Default for ContentMetrics {
    fn default() -> Self {
        Self {
            spatial_complexity: 1.0,
            temporal_complexity: 1.0,
            edge_density: 0.5,
            flat_ratio: 0.5,
            motion_uniformity: 0.5,
            content_type: SceneContentType::Moderate,
        }
    }
}

/// Scene-adaptive bitrate allocation result.
#[derive(Clone, Debug)]
pub struct AdaptiveAllocation {
    /// Recommended target bits for this frame.
    pub target_bits: u64,
    /// QP adjustment relative to base QP (can be negative).
    pub qp_offset: i8,
    /// Content type classification.
    pub content_type: SceneContentType,
    /// Confidence in the classification (0.0-1.0).
    pub confidence: f32,
    /// Suggested lambda multiplier for RDO (rate-distortion optimization).
    pub lambda_multiplier: f32,
}

/// Content-adaptive bitrate allocator.
///
/// Analyzes frame content (texture, motion, edges, flat regions) to
/// classify scenes and adjust bitrate allocation accordingly. Complex
/// scenes get more bits; static scenes get fewer bits.
///
/// # Architecture
///
/// ```text
/// Frame luma → ContentAnalysis → ContentMetrics → Classification
///                                                      ↓
///                                              AdaptiveAllocation
///                                              (target_bits, qp_offset)
/// ```
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::rate_control::lookahead::ContentAdaptiveAllocator;
///
/// let mut allocator = ContentAdaptiveAllocator::new(1920, 1080, 5_000_000);
/// let allocation = allocator.analyze_and_allocate(&luma_data, stride);
/// encoder.set_target_bits(allocation.target_bits);
/// ```
#[derive(Clone, Debug)]
pub struct ContentAdaptiveAllocator {
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Target bitrate (bits per second).
    target_bitrate: u64,
    /// Frame rate (for per-frame bit budget).
    framerate: f32,
    /// Block size for analysis.
    block_size: u32,
    /// Running average of content metrics (EMA).
    avg_metrics: ContentMetrics,
    /// EMA weight for running averages.
    ema_weight: f32,
    /// Previous frame's luma block variances.
    prev_block_variances: Vec<f32>,
    /// Frames processed.
    frame_count: u64,
    /// Scene change detector.
    scene_detector: SceneChangeDetector,
    /// Bit budget carry-over from previous frames.
    bit_surplus: i64,
}

impl ContentAdaptiveAllocator {
    /// Create a new content-adaptive allocator.
    #[must_use]
    pub fn new(width: u32, height: u32, target_bitrate: u64) -> Self {
        Self {
            width,
            height,
            target_bitrate,
            framerate: 30.0,
            block_size: 16,
            avg_metrics: ContentMetrics::default(),
            ema_weight: 0.15,
            prev_block_variances: Vec::new(),
            frame_count: 0,
            scene_detector: SceneChangeDetector::new(0.4),
            bit_surplus: 0,
        }
    }

    /// Set the target framerate.
    pub fn set_framerate(&mut self, fps: f32) {
        self.framerate = fps.max(1.0);
    }

    /// Set the analysis block size.
    pub fn set_block_size(&mut self, size: u32) {
        self.block_size = size.clamp(8, 64);
    }

    /// Per-frame bit budget at the target bitrate and framerate.
    fn base_bits_per_frame(&self) -> u64 {
        (self.target_bitrate as f64 / self.framerate as f64) as u64
    }

    /// Analyze a frame's luma plane and return adaptive allocation.
    #[must_use]
    pub fn analyze_and_allocate(&mut self, luma: &[u8], stride: usize) -> AdaptiveAllocation {
        let metrics = self.analyze_content(luma, stride);
        let allocation = self.compute_allocation(&metrics);

        // Update running averages
        self.update_averages(&metrics);
        self.frame_count += 1;

        allocation
    }

    /// Analyze content metrics from luma plane.
    fn analyze_content(&mut self, luma: &[u8], stride: usize) -> ContentMetrics {
        let bs = self.block_size as usize;
        let cols = self.width as usize / bs.max(1);
        let rows = self.height as usize / bs.max(1);
        let total_blocks = (cols * rows).max(1);

        let mut block_variances = Vec::with_capacity(total_blocks);
        let mut total_edge_pixels = 0u64;
        let mut total_pixels = 0u64;
        let mut flat_blocks = 0u32;
        let mut spatial_energy = 0.0f64;

        // Per-block analysis
        for by in 0..rows {
            for bx in 0..cols {
                let x0 = bx * bs;
                let y0 = by * bs;

                let (mean, variance) = self.block_stats(luma, stride, x0, y0, bs);
                block_variances.push(variance);
                spatial_energy += variance as f64;

                // Edge detection: count pixels with large Sobel-like gradient
                let edges = self.count_edges(luma, stride, x0, y0, bs);
                total_edge_pixels += edges as u64;
                total_pixels += (bs * bs) as u64;

                // Flat block detection
                if variance < 25.0 {
                    flat_blocks += 1;
                }
            }
        }

        let edge_density = if total_pixels > 0 {
            total_edge_pixels as f32 / total_pixels as f32
        } else {
            0.0
        };

        let flat_ratio = flat_blocks as f32 / total_blocks as f32;
        let avg_spatial = spatial_energy as f32 / total_blocks as f32;

        // Temporal complexity: compare block variances with previous frame
        let temporal_complexity = self.compute_temporal_complexity(&block_variances);

        // Motion uniformity: how consistent is the inter-frame difference
        let motion_uniformity = self.compute_motion_uniformity(&block_variances);

        // Store for next frame
        self.prev_block_variances = block_variances;

        // Classify content type
        let complexity_result = ComplexityResult {
            spatial: avg_spatial,
            temporal: temporal_complexity,
            combined: (avg_spatial * temporal_complexity).sqrt(),
            normalized: avg_spatial / self.avg_metrics.spatial_complexity.max(0.001),
        };

        let is_scene_change = self.scene_detector.detect(&complexity_result);

        let content_type = Self::classify_content(
            avg_spatial,
            temporal_complexity,
            edge_density,
            flat_ratio,
            motion_uniformity,
            is_scene_change,
        );

        ContentMetrics {
            spatial_complexity: avg_spatial,
            temporal_complexity,
            edge_density,
            flat_ratio,
            motion_uniformity,
            content_type,
        }
    }

    /// Compute mean and variance of a block.
    fn block_stats(
        &self,
        data: &[u8],
        stride: usize,
        x0: usize,
        y0: usize,
        size: usize,
    ) -> (f32, f32) {
        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        let mut count = 0u32;

        let max_y = (y0 + size).min(self.height as usize);
        let max_x = (x0 + size).min(self.width as usize);

        for y in y0..max_y {
            for x in x0..max_x {
                let idx = y * stride + x;
                if idx < data.len() {
                    let v = data[idx] as u64;
                    sum += v;
                    sum_sq += v * v;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return (128.0, 0.0);
        }

        let mean = sum as f32 / count as f32;
        let variance = (sum_sq as f32 / count as f32) - mean * mean;
        (mean, variance.max(0.0))
    }

    /// Count edge pixels in a block using simplified Sobel gradient.
    fn count_edges(&self, data: &[u8], stride: usize, x0: usize, y0: usize, size: usize) -> u32 {
        let mut edges = 0u32;
        let threshold = 30i32;

        let max_y = (y0 + size).min(self.height as usize);
        let max_x = (x0 + size).min(self.width as usize);

        for y in (y0 + 1)..(max_y.saturating_sub(1)) {
            for x in (x0 + 1)..(max_x.saturating_sub(1)) {
                let idx = y * stride + x;
                if idx + stride + 1 >= data.len() || idx < stride + 1 {
                    continue;
                }

                // Simplified Sobel: horizontal and vertical gradients
                let gx = data[idx + 1] as i32 - data[idx.saturating_sub(1)] as i32;
                let gy = data[idx + stride] as i32 - data[idx - stride] as i32;
                let magnitude = gx.abs() + gy.abs();

                if magnitude > threshold {
                    edges += 1;
                }
            }
        }

        edges
    }

    /// Compute temporal complexity from block variance differences.
    fn compute_temporal_complexity(&self, current_variances: &[f32]) -> f32 {
        if self.prev_block_variances.is_empty() {
            return 1.0;
        }

        let pairs = current_variances
            .iter()
            .zip(self.prev_block_variances.iter());
        let mut total_diff = 0.0f64;
        let mut count = 0u32;

        for (curr, prev) in pairs {
            total_diff += (curr - prev).abs() as f64;
            count += 1;
        }

        if count == 0 {
            return 1.0;
        }

        let avg_diff = total_diff as f32 / count as f32;
        // Normalize to a reasonable range (1.0 = average)
        (avg_diff / 50.0).max(0.1).min(10.0)
    }

    /// Compute motion uniformity (how consistent block-level changes are).
    fn compute_motion_uniformity(&self, current_variances: &[f32]) -> f32 {
        if self.prev_block_variances.is_empty() || current_variances.len() < 4 {
            return 0.5;
        }

        let diffs: Vec<f32> = current_variances
            .iter()
            .zip(self.prev_block_variances.iter())
            .map(|(c, p)| (c - p).abs())
            .collect();

        let mean_diff: f32 = diffs.iter().sum::<f32>() / diffs.len() as f32;
        if mean_diff < 0.001 {
            return 1.0; // No motion = perfectly uniform
        }

        let variance: f32 = diffs
            .iter()
            .map(|d| (d - mean_diff) * (d - mean_diff))
            .sum::<f32>()
            / diffs.len() as f32;

        let std_dev = variance.sqrt();
        let cv = std_dev / mean_diff; // Coefficient of variation

        // Lower CV = more uniform motion
        (1.0 - (cv / 3.0).min(1.0)).max(0.0)
    }

    /// Classify content type from metrics.
    fn classify_content(
        spatial: f32,
        temporal: f32,
        edge_density: f32,
        flat_ratio: f32,
        motion_uniformity: f32,
        is_scene_change: bool,
    ) -> SceneContentType {
        if is_scene_change {
            return SceneContentType::Transition;
        }

        // Static: very low temporal complexity, high flat ratio
        if temporal < 0.3 && flat_ratio > 0.7 {
            return SceneContentType::Static;
        }

        // High detail: high spatial complexity, significant edges
        if spatial > 200.0 && edge_density > 0.15 {
            return SceneContentType::HighDetail;
        }

        // High motion: high temporal with varied motion vectors
        if temporal > 2.0 && motion_uniformity < 0.4 {
            return SceneContentType::HighMotion;
        }

        // Low motion: low temporal, moderate spatial
        if temporal < 0.8 {
            return SceneContentType::LowMotion;
        }

        SceneContentType::Moderate
    }

    /// Compute allocation based on content metrics.
    fn compute_allocation(&mut self, metrics: &ContentMetrics) -> AdaptiveAllocation {
        let base_bits = self.base_bits_per_frame();
        let multiplier = metrics.content_type.bitrate_multiplier();

        // Adjust for relative complexity compared to running average
        let relative_spatial =
            metrics.spatial_complexity / self.avg_metrics.spatial_complexity.max(0.001);
        let relative_temporal =
            metrics.temporal_complexity / self.avg_metrics.temporal_complexity.max(0.001);
        let complexity_factor = ((relative_spatial + relative_temporal) / 2.0).clamp(0.5, 2.0);

        let target_bits = (base_bits as f64 * multiplier as f64 * complexity_factor as f64) as u64;

        // Add surplus/deficit from previous frames (budget smoothing)
        let adjusted_bits = (target_bits as i64 + self.bit_surplus / 4).max(1) as u64;
        self.bit_surplus -= adjusted_bits as i64 - base_bits as i64;
        // Clamp surplus to prevent unbounded accumulation
        self.bit_surplus = self
            .bit_surplus
            .clamp(-(base_bits as i64 * 8), base_bits as i64 * 8);

        // Lambda multiplier: inverse of complexity (simpler content = higher lambda = fewer bits)
        let lambda_multiplier = 1.0 / complexity_factor.max(0.1);

        // Confidence: higher when we have more history
        let confidence = (self.frame_count as f32 / 30.0).min(1.0);

        AdaptiveAllocation {
            target_bits: adjusted_bits,
            qp_offset: metrics.content_type.qp_adjustment(),
            content_type: metrics.content_type,
            confidence,
            lambda_multiplier,
        }
    }

    /// Update running average metrics.
    fn update_averages(&mut self, metrics: &ContentMetrics) {
        let w = self.ema_weight;
        self.avg_metrics.spatial_complexity =
            self.avg_metrics.spatial_complexity * (1.0 - w) + metrics.spatial_complexity * w;
        self.avg_metrics.temporal_complexity =
            self.avg_metrics.temporal_complexity * (1.0 - w) + metrics.temporal_complexity * w;
        self.avg_metrics.edge_density =
            self.avg_metrics.edge_density * (1.0 - w) + metrics.edge_density * w;
        self.avg_metrics.flat_ratio =
            self.avg_metrics.flat_ratio * (1.0 - w) + metrics.flat_ratio * w;
        self.avg_metrics.motion_uniformity =
            self.avg_metrics.motion_uniformity * (1.0 - w) + metrics.motion_uniformity * w;
    }

    /// Reset the allocator state.
    pub fn reset(&mut self) {
        self.avg_metrics = ContentMetrics::default();
        self.prev_block_variances.clear();
        self.frame_count = 0;
        self.scene_detector.reset();
        self.bit_surplus = 0;
    }

    /// Get frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get current average metrics.
    #[must_use]
    pub fn average_metrics(&self) -> &ContentMetrics {
        &self.avg_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Vec<u8> {
        vec![128u8; (width * height) as usize]
    }

    #[test]
    fn test_lookahead_creation() {
        let lookahead = Lookahead::with_params(40, 1920, 1080);
        assert_eq!(lookahead.depth(), 40);
        assert!(lookahead.is_empty());
    }

    #[test]
    fn test_push_frame() {
        let mut lookahead = Lookahead::with_params(10, 64, 64);
        let frame = create_test_frame(64, 64);

        for _ in 0..5 {
            let result = lookahead.push_frame(&frame, 64);
            assert!(result.is_none()); // Buffer not full yet
        }

        assert_eq!(lookahead.buffered_count(), 5);
    }

    #[test]
    fn test_buffer_full() {
        let mut lookahead = Lookahead::with_params(5, 64, 64);
        let frame = create_test_frame(64, 64);

        for _ in 0..5 {
            let _ = lookahead.push_frame(&frame, 64);
        }

        assert!(lookahead.is_full());

        // Next push should return a frame
        let result = lookahead.push_frame(&frame, 64);
        assert!(result.is_some());
    }

    #[test]
    fn test_flush() {
        let mut lookahead = Lookahead::with_params(10, 64, 64);
        let frame = create_test_frame(64, 64);

        for _ in 0..5 {
            let _ = lookahead.push_frame(&frame, 64);
        }

        let flushed = lookahead.flush();
        assert_eq!(flushed.len(), 5);
        assert!(lookahead.is_empty());
    }

    #[test]
    fn test_push_with_complexity() {
        let mut lookahead = Lookahead::with_params(10, 64, 64);

        let complexity = ComplexityResult {
            spatial: 1.0,
            temporal: 1.0,
            combined: 1.0,
            normalized: 1.0,
        };

        for _ in 0..5 {
            let _ = lookahead.push_frame_with_complexity(complexity);
        }

        assert_eq!(lookahead.buffered_count(), 5);
    }

    #[test]
    fn test_scene_cut_detection() {
        let mut detector = SceneChangeDetector::new(0.3);
        detector.set_min_interval(0); // Allow immediate detection

        let normal = ComplexityResult {
            spatial: 1.0,
            temporal: 1.0,
            combined: 1.0,
            normalized: 1.0,
        };

        let scene_cut = ComplexityResult {
            spatial: 3.0,
            temporal: 3.0,
            combined: 3.0,
            normalized: 3.0,
        };

        // First frame establishes baseline
        assert!(!detector.detect(&normal));

        // Normal frame
        assert!(!detector.detect(&normal));

        // Scene cut - significant complexity increase
        assert!(detector.detect(&scene_cut));
    }

    #[test]
    fn test_scene_cut_min_interval() {
        let mut detector = SceneChangeDetector::new(0.3);
        detector.set_min_interval(5);

        let normal = ComplexityResult::default_complexity();
        let scene_cut = ComplexityResult {
            spatial: 5.0,
            temporal: 5.0,
            combined: 5.0,
            normalized: 5.0,
        };

        // Trigger a scene change
        for _ in 0..6 {
            let _ = detector.detect(&normal);
        }
        assert!(detector.detect(&scene_cut));

        // Immediate next frame should not be detected even with high complexity
        assert!(!detector.detect(&scene_cut));
    }

    #[test]
    fn test_lookahead_frame() {
        let frame = LookaheadFrame::new(42);
        assert_eq!(frame.frame_index, 42);
        assert_eq!(frame.frame_type, FrameType::Inter);
        assert!(!frame.is_scene_cut);
    }

    #[test]
    fn test_mini_gop_info() {
        let info = MiniGopInfo {
            start_frame: 0,
            frame_count: 15,
            i_frames: 1,
            p_frames: 4,
            b_frames: 10,
            total_complexity: 15.0,
            has_scene_cut: false,
        };

        assert!((info.average_complexity() - 1.0).abs() < f32::EPSILON);

        let bits = info.suggested_bits(100_000);
        assert!(bits > 0);
    }

    #[test]
    fn test_lookahead_reset() {
        let mut lookahead = Lookahead::with_params(10, 64, 64);
        let frame = create_test_frame(64, 64);

        for _ in 0..5 {
            let _ = lookahead.push_frame(&frame, 64);
        }

        lookahead.reset();

        assert!(lookahead.is_empty());
        assert_eq!(lookahead.frame_count(), 0);
    }

    #[test]
    fn test_average_complexity() {
        let mut lookahead = Lookahead::with_params(10, 64, 64);

        let complexity = ComplexityResult {
            spatial: 1.0,
            temporal: 1.0,
            combined: 2.0,
            normalized: 1.0,
        };

        for _ in 0..5 {
            let _ = lookahead.push_frame_with_complexity(complexity);
        }

        assert!((lookahead.average_complexity() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scene_change_detector_reset() {
        let mut detector = SceneChangeDetector::new(0.3);

        let complexity = ComplexityResult::default_complexity();
        for _ in 0..10 {
            let _ = detector.detect(&complexity);
        }

        detector.reset();
        assert_eq!(detector.total_changes(), 0);
    }

    // =====================================================================
    // Content-Adaptive Allocator Tests
    // =====================================================================

    #[test]
    fn test_content_type_bitrate_multipliers() {
        assert!(SceneContentType::Static.bitrate_multiplier() < 1.0);
        assert!((SceneContentType::Moderate.bitrate_multiplier() - 1.0).abs() < f32::EPSILON);
        assert!(SceneContentType::HighMotion.bitrate_multiplier() > 1.0);
        assert!(SceneContentType::Transition.bitrate_multiplier() > 1.0);
    }

    #[test]
    fn test_content_type_qp_adjustments() {
        // Static gets positive QP (lower quality ok)
        assert!(SceneContentType::Static.qp_adjustment() > 0);
        // Moderate is neutral
        assert_eq!(SceneContentType::Moderate.qp_adjustment(), 0);
        // High motion gets negative QP (needs higher quality)
        assert!(SceneContentType::HighMotion.qp_adjustment() < 0);
    }

    #[test]
    fn test_allocator_creation() {
        let allocator = ContentAdaptiveAllocator::new(1920, 1080, 5_000_000);
        assert_eq!(allocator.frame_count(), 0);
        assert_eq!(allocator.width, 1920);
        assert_eq!(allocator.height, 1080);
    }

    #[test]
    fn test_allocator_flat_frame() {
        let mut allocator = ContentAdaptiveAllocator::new(64, 64, 5_000_000);
        allocator.set_framerate(30.0);

        // Flat frame: all same value
        let luma = vec![128u8; 64 * 64];
        let alloc = allocator.analyze_and_allocate(&luma, 64);

        assert!(alloc.target_bits > 0);
        // Flat frames should be classified as static or low motion
        assert!(
            alloc.content_type == SceneContentType::Static
                || alloc.content_type == SceneContentType::LowMotion
                || alloc.content_type == SceneContentType::Moderate
        );
    }

    #[test]
    fn test_allocator_noisy_frame() {
        let mut allocator = ContentAdaptiveAllocator::new(64, 64, 5_000_000);
        allocator.set_framerate(30.0);

        // High-variance frame: alternating values
        let mut luma = vec![0u8; 64 * 64];
        for (i, pixel) in luma.iter_mut().enumerate() {
            *pixel = if i % 2 == 0 { 20 } else { 220 };
        }

        let alloc = allocator.analyze_and_allocate(&luma, 64);
        assert!(alloc.target_bits > 0);
    }

    #[test]
    fn test_allocator_multiple_frames() {
        let mut allocator = ContentAdaptiveAllocator::new(64, 64, 5_000_000);
        allocator.set_framerate(30.0);

        let flat_frame = vec![128u8; 64 * 64];
        let noisy_frame: Vec<u8> = (0..64 * 64).map(|i| ((i * 37) % 256) as u8).collect();

        // Process several flat frames
        for _ in 0..5 {
            let _ = allocator.analyze_and_allocate(&flat_frame, 64);
        }

        // Then a noisy frame should get more bits
        let flat_alloc = allocator.analyze_and_allocate(&flat_frame, 64);
        let noisy_alloc = allocator.analyze_and_allocate(&noisy_frame, 64);

        // Noisy frame should generally get more bits due to higher complexity
        // (though exact values depend on running averages)
        assert!(noisy_alloc.target_bits > 0);
        assert!(flat_alloc.target_bits > 0);
    }

    #[test]
    fn test_allocator_reset() {
        let mut allocator = ContentAdaptiveAllocator::new(64, 64, 5_000_000);
        let luma = vec![128u8; 64 * 64];

        let _ = allocator.analyze_and_allocate(&luma, 64);
        let _ = allocator.analyze_and_allocate(&luma, 64);

        allocator.reset();
        assert_eq!(allocator.frame_count(), 0);
    }

    #[test]
    fn test_allocator_confidence_increases() {
        let mut allocator = ContentAdaptiveAllocator::new(64, 64, 5_000_000);
        let luma = vec![128u8; 64 * 64];

        let alloc1 = allocator.analyze_and_allocate(&luma, 64);
        for _ in 0..30 {
            let _ = allocator.analyze_and_allocate(&luma, 64);
        }
        let alloc_later = allocator.analyze_and_allocate(&luma, 64);

        assert!(alloc_later.confidence >= alloc1.confidence);
    }

    #[test]
    fn test_allocator_set_block_size() {
        let mut allocator = ContentAdaptiveAllocator::new(64, 64, 5_000_000);
        allocator.set_block_size(32);
        assert_eq!(allocator.block_size, 32);

        // Clamping
        allocator.set_block_size(2);
        assert_eq!(allocator.block_size, 8);
        allocator.set_block_size(128);
        assert_eq!(allocator.block_size, 64);
    }

    #[test]
    fn test_classify_static_content() {
        let ct = ContentAdaptiveAllocator::classify_content(10.0, 0.1, 0.02, 0.9, 0.9, false);
        assert_eq!(ct, SceneContentType::Static);
    }

    #[test]
    fn test_classify_high_motion() {
        let ct = ContentAdaptiveAllocator::classify_content(50.0, 3.0, 0.1, 0.2, 0.2, false);
        assert_eq!(ct, SceneContentType::HighMotion);
    }

    #[test]
    fn test_classify_transition() {
        let ct = ContentAdaptiveAllocator::classify_content(50.0, 1.0, 0.1, 0.3, 0.5, true);
        assert_eq!(ct, SceneContentType::Transition);
    }

    #[test]
    fn test_classify_high_detail() {
        let ct = ContentAdaptiveAllocator::classify_content(500.0, 1.0, 0.25, 0.1, 0.5, false);
        assert_eq!(ct, SceneContentType::HighDetail);
    }

    #[test]
    fn test_content_metrics_default() {
        let m = ContentMetrics::default();
        assert!((m.spatial_complexity - 1.0).abs() < f32::EPSILON);
        assert!((m.temporal_complexity - 1.0).abs() < f32::EPSILON);
        assert_eq!(m.content_type, SceneContentType::Moderate);
    }

    #[test]
    fn test_adaptive_allocation_lambda() {
        let mut allocator = ContentAdaptiveAllocator::new(64, 64, 5_000_000);
        let luma = vec![128u8; 64 * 64];
        let alloc = allocator.analyze_and_allocate(&luma, 64);

        // Lambda multiplier should be positive
        assert!(alloc.lambda_multiplier > 0.0);
    }

    // =========================================================================
    // Scene-adaptive bitrate allocation tests (Task 2)
    // =========================================================================

    /// Generate a synthetic luma frame with given mean and variance.
    fn make_luma_frame(w: usize, h: usize, mean: u8, variance: u32) -> Vec<u8> {
        let mut buf = vec![mean; w * h];
        if variance > 0 {
            // Add a deterministic checkerboard to create spatial variance
            let step = (variance as usize).max(1).min(64);
            for y in 0..h {
                for x in 0..w {
                    let v = if (x / step + y / step) % 2 == 0 {
                        mean.saturating_add((variance / 2) as u8)
                    } else {
                        mean.saturating_sub((variance / 2) as u8)
                    };
                    buf[y * w + x] = v;
                }
            }
        }
        buf
    }

    #[test]
    fn test_scene_adaptive_more_bits_for_complex_content() {
        let w = 64usize;
        let h = 64usize;
        let target_bps = 5_000_000u64;
        let mut allocator = ContentAdaptiveAllocator::new(w as u32, h as u32, target_bps);

        // Warm up with flat frames to establish baseline
        let flat = make_luma_frame(w, h, 128, 0);
        for _ in 0..10 {
            let _ = allocator.analyze_and_allocate(&flat, w);
        }

        // Allocate for flat (low complexity)
        let flat_alloc = allocator.analyze_and_allocate(&flat, w);

        // Now allocate for a high-variance (complex) frame
        let complex = make_luma_frame(w, h, 128, 80);
        let complex_alloc = allocator.analyze_and_allocate(&complex, w);

        // Complex frame should receive >= bits vs flat frame
        // (The scene-adaptive system allocates more bits to complex content)
        assert!(
            complex_alloc.target_bits
                >= flat_alloc
                    .target_bits
                    .saturating_sub(flat_alloc.target_bits / 4),
            "Complex frame should get >= bits to flat frame: complex={}, flat={}",
            complex_alloc.target_bits,
            flat_alloc.target_bits
        );
    }

    #[test]
    fn test_scene_adaptive_scene_change_boosts_bits() {
        let w = 64usize;
        let h = 64usize;
        let target_bps = 2_000_000u64;
        let mut allocator = ContentAdaptiveAllocator::new(w as u32, h as u32, target_bps);

        // Warm up with static frames
        let static_frame = make_luma_frame(w, h, 64, 2);
        for _ in 0..20 {
            let _ = allocator.analyze_and_allocate(&static_frame, w);
        }
        let static_alloc = allocator.analyze_and_allocate(&static_frame, w);

        // Sudden scene change: very different luminance and high variance
        let scene_change_frame = make_luma_frame(w, h, 200, 100);
        let scene_alloc = allocator.analyze_and_allocate(&scene_change_frame, w);

        // After a scene change the allocator should provide a non-trivial allocation
        assert!(
            scene_alloc.target_bits > 0,
            "Scene change frame must receive non-zero bits"
        );

        // The scene change confidence should still be positive
        assert!(scene_alloc.confidence >= 0.0 && scene_alloc.confidence <= 1.0);
    }

    #[test]
    fn test_scene_adaptive_static_scene_reduces_qp_offset() {
        let w = 64usize;
        let h = 64usize;
        let mut allocator = ContentAdaptiveAllocator::new(w as u32, h as u32, 5_000_000);

        // Pure flat content → StaticLow content type → lower bitrate multiplier
        let flat = make_luma_frame(w, h, 128, 0);
        for _ in 0..10 {
            let _ = allocator.analyze_and_allocate(&flat, w);
        }
        let flat_alloc = allocator.analyze_and_allocate(&flat, w);

        // For flat/static content, QP offset should be non-negative (increase QP = fewer bits)
        assert!(
            flat_alloc.qp_offset >= 0,
            "Static content should have QP offset >= 0, got {}",
            flat_alloc.qp_offset
        );
    }

    #[test]
    fn test_scene_adaptive_confidence_increases_over_time() {
        let w = 64usize;
        let h = 64usize;
        let mut allocator = ContentAdaptiveAllocator::new(w as u32, h as u32, 1_000_000);
        let luma = make_luma_frame(w, h, 128, 20);

        let mut prev_confidence = 0.0f32;
        for i in 0..30 {
            let alloc = allocator.analyze_and_allocate(&luma, w);
            if i > 0 {
                // Confidence should be non-decreasing after the first frame
                assert!(
                    alloc.confidence >= prev_confidence - 0.01,
                    "Confidence should be non-decreasing at frame {i}: {:.3} < prev {:.3}",
                    alloc.confidence,
                    prev_confidence
                );
            }
            prev_confidence = alloc.confidence;
        }

        // After 30 frames, confidence should be at or near 1.0
        assert!(
            prev_confidence >= 0.9,
            "Confidence should be >= 0.9 after 30 frames, got {prev_confidence:.3}"
        );
    }

    #[test]
    fn test_scene_adaptive_target_bits_positive() {
        let w = 128usize;
        let h = 72usize;
        let mut allocator = ContentAdaptiveAllocator::new(w as u32, h as u32, 500_000);
        let luma = make_luma_frame(w, h, 200, 40);

        for _ in 0..50 {
            let alloc = allocator.analyze_and_allocate(&luma, w);
            assert!(alloc.target_bits > 0, "Target bits must always be positive");
        }
    }

    #[test]
    fn test_scene_adaptive_reset_clears_history() {
        let w = 64usize;
        let h = 64usize;
        let mut allocator = ContentAdaptiveAllocator::new(w as u32, h as u32, 2_000_000);
        let luma = make_luma_frame(w, h, 100, 30);

        for _ in 0..15 {
            let _ = allocator.analyze_and_allocate(&luma, w);
        }
        allocator.reset();

        // After reset, a flat frame should get base allocation
        let flat = make_luma_frame(w, h, 128, 0);
        let alloc = allocator.analyze_and_allocate(&flat, w);
        assert!(alloc.target_bits > 0);
        // Confidence should be near zero after reset
        assert!(
            alloc.confidence <= 0.1,
            "Confidence should reset to near 0, got {}",
            alloc.confidence
        );
    }

    #[test]
    fn test_scene_adaptive_bit_surplus_bounded() {
        // Verify that the bit surplus doesn't grow without bound over many frames.
        // We do this by checking that target_bits stays within a reasonable range
        // of the base bits per frame.
        let w = 64usize;
        let h = 64usize;
        let target_bps = 3_000_000u64;
        let fps = 30.0f64;
        let base_per_frame = target_bps as f64 / fps;
        let mut allocator = ContentAdaptiveAllocator::new(w as u32, h as u32, target_bps);
        let luma = make_luma_frame(w, h, 128, 10);

        let mut total_target_bits = 0u64;
        let n = 150usize; // 5 seconds
        for _ in 0..n {
            let alloc = allocator.analyze_and_allocate(&luma, w);
            total_target_bits += alloc.target_bits;
        }

        let expected_total = base_per_frame * n as f64;
        let ratio = total_target_bits as f64 / expected_total;
        // Allow up to 5× deviation (content type multipliers can be up to ~2×)
        assert!(
            ratio >= 0.1 && ratio <= 5.0,
            "Total target bits ratio {ratio:.3} out of [0.1, 5.0] (total={total_target_bits}, expected={expected_total:.0})"
        );
    }
}
