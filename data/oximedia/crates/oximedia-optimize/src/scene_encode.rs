#![allow(dead_code)]
//! Scene-aware encoding optimization.
//!
//! This module provides scene-level encoding strategies that adjust encoding
//! parameters based on scene characteristics (complexity, motion, texture).
//! It operates at a higher level than per-frame optimization, making decisions
//! about GOP structure, QP offsets, and bitrate allocation on a scene-by-scene
//! basis.
//!
//! The [`LookaheadSceneQp`] analyzer uses lookahead frame data to make
//! scene-cut-aware QP adjustments, boosting quality at scene boundaries
//! and smoothly transitioning between scenes of different complexity.

use std::collections::VecDeque;

/// Scene type classification for encoding decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneType {
    /// Static or near-static scene (e.g., title card, credits).
    Static,
    /// Talking head or slow-paced dialogue scene.
    Dialogue,
    /// Moderate action with some motion.
    Moderate,
    /// Fast action scene with high motion and complexity.
    Action,
    /// Scene with lots of fine detail (e.g., foliage, crowds).
    HighDetail,
    /// Dark or low-light scene.
    DarkScene,
    /// Scene transition / crossfade.
    Transition,
}

/// Metrics describing a scene's visual characteristics.
#[derive(Debug, Clone)]
pub struct SceneMetrics {
    /// Scene index in the stream.
    pub scene_index: u32,
    /// Frame index where the scene starts.
    pub start_frame: u64,
    /// Frame index where the scene ends (exclusive).
    pub end_frame: u64,
    /// Average spatial complexity (0.0 - 1.0).
    pub spatial_complexity: f64,
    /// Average temporal complexity / motion (0.0 - 1.0).
    pub temporal_complexity: f64,
    /// Average luminance (0.0 - 1.0).
    pub avg_luminance: f64,
    /// Luminance variance.
    pub luminance_variance: f64,
    /// Texture density (0.0 - 1.0).
    pub texture_density: f64,
    /// Classified scene type.
    pub scene_type: SceneType,
}

impl SceneMetrics {
    /// Creates new scene metrics.
    #[must_use]
    pub fn new(scene_index: u32, start_frame: u64, end_frame: u64) -> Self {
        Self {
            scene_index,
            start_frame,
            end_frame,
            spatial_complexity: 0.5,
            temporal_complexity: 0.5,
            avg_luminance: 0.5,
            luminance_variance: 0.1,
            texture_density: 0.5,
            scene_type: SceneType::Moderate,
        }
    }

    /// Returns the number of frames in this scene.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns a combined complexity score (0.0 - 1.0).
    #[must_use]
    pub fn combined_complexity(&self) -> f64 {
        (self.spatial_complexity * 0.4
            + self.temporal_complexity * 0.4
            + self.texture_density * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Returns true if this is a dark scene.
    #[must_use]
    pub fn is_dark(&self) -> bool {
        self.avg_luminance < 0.2
    }
}

/// Encoding parameters for a scene.
#[derive(Debug, Clone)]
pub struct SceneEncodeParams {
    /// QP offset relative to base QP (can be negative for higher quality).
    pub qp_offset: f64,
    /// Bitrate allocation weight (1.0 = normal, >1.0 = more bits).
    pub bitrate_weight: f64,
    /// Recommended GOP size in frames.
    pub gop_size: u32,
    /// Whether to force a keyframe at scene start.
    pub force_keyframe: bool,
    /// Minimum QP allowed.
    pub min_qp: f64,
    /// Maximum QP allowed.
    pub max_qp: f64,
    /// B-frame count for this scene.
    pub b_frames: u32,
    /// Whether to enable adaptive quantization.
    pub enable_aq: bool,
    /// AQ strength (0.0 - 2.0).
    pub aq_strength: f64,
}

impl Default for SceneEncodeParams {
    fn default() -> Self {
        Self {
            qp_offset: 0.0,
            bitrate_weight: 1.0,
            gop_size: 250,
            force_keyframe: true,
            min_qp: 0.0,
            max_qp: 51.0,
            b_frames: 3,
            enable_aq: true,
            aq_strength: 1.0,
        }
    }
}

impl SceneEncodeParams {
    /// Creates new default scene encoding parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the effective QP given a base QP.
    #[must_use]
    pub fn effective_qp(&self, base_qp: f64) -> f64 {
        (base_qp + self.qp_offset).clamp(self.min_qp, self.max_qp)
    }
}

/// Scene-based encoding optimizer.
#[derive(Debug)]
pub struct SceneEncoder {
    /// Base QP for the encoding session.
    base_qp: f64,
    /// Target bitrate in bps.
    target_bitrate_bps: u64,
    /// Frame rate.
    frame_rate: f64,
    /// Scene parameters generated so far.
    scene_params: Vec<SceneEncodeParams>,
    /// Scene metrics history.
    scene_history: VecDeque<SceneMetrics>,
    /// Maximum scene history length.
    max_history: usize,
}

impl SceneEncoder {
    /// Creates a new scene encoder with target settings.
    #[must_use]
    pub fn new(base_qp: f64, target_bitrate_bps: u64, frame_rate: f64) -> Self {
        Self {
            base_qp,
            target_bitrate_bps,
            frame_rate,
            scene_params: Vec::new(),
            scene_history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Returns the base QP.
    #[must_use]
    pub fn base_qp(&self) -> f64 {
        self.base_qp
    }

    /// Returns the target bitrate.
    #[must_use]
    pub fn target_bitrate_bps(&self) -> u64 {
        self.target_bitrate_bps
    }

    /// Generates encoding parameters for a scene based on its metrics.
    #[must_use]
    pub fn generate_params(&self, metrics: &SceneMetrics) -> SceneEncodeParams {
        let mut params = SceneEncodeParams::default();

        // Determine QP offset based on scene type and complexity
        params.qp_offset = self.compute_qp_offset(metrics);
        params.bitrate_weight = self.compute_bitrate_weight(metrics);
        params.gop_size = self.compute_gop_size(metrics);
        params.b_frames = self.compute_b_frames(metrics);
        params.aq_strength = self.compute_aq_strength(metrics);

        // Always force keyframe at scene boundaries
        params.force_keyframe = true;

        params
    }

    /// Processes a scene: records metrics and generates params.
    pub fn process_scene(&mut self, metrics: SceneMetrics) -> SceneEncodeParams {
        let params = self.generate_params(&metrics);
        self.scene_params.push(params.clone());
        self.scene_history.push_back(metrics);
        if self.scene_history.len() > self.max_history {
            self.scene_history.pop_front();
        }
        params
    }

    /// Returns scene parameters generated so far.
    #[must_use]
    pub fn scene_params(&self) -> &[SceneEncodeParams] {
        &self.scene_params
    }

    /// Returns the number of scenes processed.
    #[must_use]
    pub fn scenes_processed(&self) -> usize {
        self.scene_params.len()
    }

    /// Returns the scene history.
    #[must_use]
    pub fn scene_history(&self) -> &VecDeque<SceneMetrics> {
        &self.scene_history
    }

    /// Computes the QP offset for a scene.
    fn compute_qp_offset(&self, metrics: &SceneMetrics) -> f64 {
        match metrics.scene_type {
            SceneType::Static => -4.0,    // boost quality for static
            SceneType::Dialogue => -2.0,  // slightly boost dialogue
            SceneType::Moderate => 0.0,   // neutral
            SceneType::Action => 2.0,     // relax for action
            SceneType::HighDetail => 1.0, // slight relax for heavy detail
            SceneType::DarkScene => -3.0, // boost dark scenes (artifacts visible)
            SceneType::Transition => 3.0, // relax during transitions
        }
    }

    /// Computes the bitrate weight for a scene.
    fn compute_bitrate_weight(&self, metrics: &SceneMetrics) -> f64 {
        let complexity = metrics.combined_complexity();
        // Map complexity [0,1] to weight [0.6, 1.8]
        0.6 + complexity * 1.2
    }

    /// Computes GOP size based on temporal characteristics.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn compute_gop_size(&self, metrics: &SceneMetrics) -> u32 {
        let frame_count = metrics.frame_count();
        // GOP should not exceed scene length
        let max_gop = frame_count.min(300) as u32;

        match metrics.scene_type {
            SceneType::Static => max_gop.min(300),
            SceneType::Dialogue => max_gop.min(250),
            SceneType::Moderate => max_gop.min(200),
            SceneType::Action => max_gop.min(120),
            SceneType::HighDetail => max_gop.min(150),
            SceneType::DarkScene => max_gop.min(250),
            SceneType::Transition => max_gop.min(60),
        }
    }

    /// Computes B-frame count for a scene.
    fn compute_b_frames(&self, metrics: &SceneMetrics) -> u32 {
        match metrics.scene_type {
            SceneType::Static => 5,
            SceneType::Dialogue => 4,
            SceneType::Moderate => 3,
            SceneType::Action => 2,
            SceneType::HighDetail => 3,
            SceneType::DarkScene => 4,
            SceneType::Transition => 1,
        }
    }

    /// Computes adaptive quantization strength.
    fn compute_aq_strength(&self, metrics: &SceneMetrics) -> f64 {
        if metrics.is_dark() {
            // Stronger AQ for dark scenes to reduce banding
            1.5
        } else if metrics.spatial_complexity > 0.7 {
            1.2
        } else {
            1.0
        }
    }

    /// Returns the average complexity across all processed scenes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_scene_complexity(&self) -> f64 {
        if self.scene_history.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .scene_history
            .iter()
            .map(|m| m.combined_complexity())
            .sum();
        total / self.scene_history.len() as f64
    }
}

/// Lookahead-based scene-aware QP adjustment.
///
/// Uses a window of lookahead frames to detect scene cuts and adjust QP
/// based on upcoming content complexity. This prevents quality drops at
/// scene boundaries by pre-allocating bits and smoothly transitioning
/// QP between scenes of different complexity.
#[derive(Debug)]
pub struct LookaheadSceneQp {
    /// Number of lookahead frames.
    lookahead_depth: usize,
    /// Base QP for the encoding session.
    base_qp: f64,
    /// Lookahead buffer of frame complexities.
    complexity_buffer: VecDeque<FrameLookaheadInfo>,
    /// QP adjustment history for smoothing.
    qp_history: VecDeque<f64>,
    /// Maximum QP delta from base.
    max_qp_delta: f64,
    /// QP smoothing factor (0.0-1.0, higher = more smoothing).
    smoothing: f64,
    /// Scene cut boost: extra QP reduction at scene boundaries.
    scene_cut_boost: f64,
    /// Total frames processed.
    frames_processed: u64,
}

/// Per-frame information used by the lookahead QP adjuster.
#[derive(Debug, Clone)]
pub struct FrameLookaheadInfo {
    /// Frame index.
    pub frame_index: u64,
    /// Spatial complexity (0.0-1.0).
    pub spatial_complexity: f64,
    /// Temporal complexity / motion (0.0-1.0).
    pub temporal_complexity: f64,
    /// Whether this frame is a detected scene cut.
    pub is_scene_cut: bool,
    /// Average luminance.
    pub avg_luminance: f64,
}

impl FrameLookaheadInfo {
    /// Creates a new frame info entry.
    pub fn new(frame_index: u64) -> Self {
        Self {
            frame_index,
            spatial_complexity: 0.5,
            temporal_complexity: 0.5,
            is_scene_cut: false,
            avg_luminance: 0.5,
        }
    }

    /// Returns the combined complexity of this frame.
    pub fn combined_complexity(&self) -> f64 {
        (self.spatial_complexity * 0.5 + self.temporal_complexity * 0.5).clamp(0.0, 1.0)
    }
}

/// Result of lookahead QP analysis for a single frame.
#[derive(Debug, Clone)]
pub struct LookaheadQpResult {
    /// Recommended QP for this frame.
    pub recommended_qp: f64,
    /// QP delta from base.
    pub qp_delta: f64,
    /// Whether this frame is at/near a scene cut.
    pub near_scene_cut: bool,
    /// Distance to the next scene cut in frames (0 = this frame is a cut).
    pub distance_to_next_cut: Option<usize>,
    /// Upcoming complexity average (from lookahead).
    pub upcoming_complexity: f64,
}

impl LookaheadSceneQp {
    /// Creates a new lookahead scene QP adjuster.
    pub fn new(lookahead_depth: usize, base_qp: f64) -> Self {
        Self {
            lookahead_depth: lookahead_depth.max(1),
            base_qp,
            complexity_buffer: VecDeque::new(),
            qp_history: VecDeque::new(),
            max_qp_delta: 6.0,
            smoothing: 0.3,
            scene_cut_boost: 3.0,
            frames_processed: 0,
        }
    }

    /// Sets the maximum QP delta.
    pub fn set_max_qp_delta(&mut self, delta: f64) {
        self.max_qp_delta = delta.max(0.0);
    }

    /// Sets the scene cut QP boost.
    pub fn set_scene_cut_boost(&mut self, boost: f64) {
        self.scene_cut_boost = boost.max(0.0);
    }

    /// Sets the QP smoothing factor.
    pub fn set_smoothing(&mut self, smoothing: f64) {
        self.smoothing = smoothing.clamp(0.0, 1.0);
    }

    /// Feeds a frame's lookahead info into the buffer.
    pub fn feed_frame(&mut self, info: FrameLookaheadInfo) {
        self.complexity_buffer.push_back(info);
        // Keep buffer at lookahead depth + some margin
        while self.complexity_buffer.len() > self.lookahead_depth * 2 {
            self.complexity_buffer.pop_front();
        }
    }

    /// Analyzes the current frame considering lookahead data and returns QP recommendation.
    ///
    /// The algorithm:
    /// 1. Computes upcoming complexity from lookahead frames
    /// 2. Detects proximity to scene cuts
    /// 3. Boosts QP at scene boundaries (lower QP = more bits for I-frames)
    /// 4. Smoothly transitions QP between scenes of different complexity
    /// 5. Applies temporal smoothing to prevent QP oscillation
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_frame(&mut self, current: &FrameLookaheadInfo) -> LookaheadQpResult {
        // Find scene cuts in the lookahead window
        let distance_to_cut = self.find_next_scene_cut();
        let near_scene_cut =
            current.is_scene_cut || distance_to_cut.map(|d| d < 3).unwrap_or(false);

        // Compute upcoming complexity from lookahead
        let upcoming_complexity = self.compute_upcoming_complexity();

        // Base QP delta from complexity
        let complexity_delta = self.complexity_to_qp_delta(upcoming_complexity);

        // Scene cut adjustment
        let scene_cut_delta = if current.is_scene_cut {
            // Strong boost at the scene cut itself (first frame of new scene)
            -self.scene_cut_boost
        } else if let Some(dist) = distance_to_cut {
            if dist <= 2 {
                // Ramp down bits before the cut (let the I-frame have more)
                let ramp_factor = dist as f64 / 3.0;
                self.scene_cut_boost * 0.3 * ramp_factor
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Dark scene adjustment
        let dark_delta = if current.avg_luminance < 0.15 {
            -1.5 // Boost dark scenes
        } else {
            0.0
        };

        // Combine deltas
        let raw_delta = complexity_delta + scene_cut_delta + dark_delta;
        let clamped_delta = raw_delta.clamp(-self.max_qp_delta, self.max_qp_delta);

        // Apply temporal smoothing
        let smoothed_delta = if let Some(&last_qp) = self.qp_history.back() {
            let last_delta = last_qp - self.base_qp;
            if current.is_scene_cut {
                // No smoothing at scene cuts
                clamped_delta
            } else {
                last_delta * self.smoothing + clamped_delta * (1.0 - self.smoothing)
            }
        } else {
            clamped_delta
        };

        let recommended_qp = (self.base_qp + smoothed_delta).clamp(1.0, 51.0);

        // Record history
        self.qp_history.push_back(recommended_qp);
        if self.qp_history.len() > self.lookahead_depth {
            self.qp_history.pop_front();
        }
        self.frames_processed += 1;

        LookaheadQpResult {
            recommended_qp,
            qp_delta: smoothed_delta,
            near_scene_cut,
            distance_to_next_cut: distance_to_cut,
            upcoming_complexity,
        }
    }

    /// Returns the number of frames processed.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Resets the lookahead state.
    pub fn reset(&mut self) {
        self.complexity_buffer.clear();
        self.qp_history.clear();
        self.frames_processed = 0;
    }

    /// Finds the distance to the next scene cut in the lookahead buffer.
    fn find_next_scene_cut(&self) -> Option<usize> {
        for (i, info) in self.complexity_buffer.iter().enumerate() {
            if info.is_scene_cut && i > 0 {
                return Some(i);
            }
        }
        None
    }

    /// Computes the average complexity of upcoming frames in the buffer.
    #[allow(clippy::cast_precision_loss)]
    fn compute_upcoming_complexity(&self) -> f64 {
        if self.complexity_buffer.is_empty() {
            return 0.5;
        }
        let count = self.complexity_buffer.len().min(self.lookahead_depth);
        let sum: f64 = self
            .complexity_buffer
            .iter()
            .take(count)
            .map(|f| f.combined_complexity())
            .sum();
        sum / count as f64
    }

    /// Maps complexity to QP delta using a sigmoid-like curve.
    fn complexity_to_qp_delta(&self, complexity: f64) -> f64 {
        // More complex = lower QP (more bits)
        // Map [0,1] complexity to [-max_delta, +max_delta] delta
        let centered = complexity - 0.5;
        -centered * self.max_qp_delta * 2.0
    }
}

/// Bitrate distribution plan across scenes.
#[derive(Debug, Clone)]
pub struct SceneBitratePlan {
    /// Scene index to allocated bitrate in bps.
    allocations: Vec<SceneBitrateAlloc>,
}

/// A single scene's bitrate allocation.
#[derive(Debug, Clone)]
pub struct SceneBitrateAlloc {
    /// Scene index.
    pub scene_index: u32,
    /// Allocated bitrate in bps.
    pub allocated_bps: u64,
    /// Frame count in the scene.
    pub frame_count: u64,
    /// Allocated bits for the entire scene.
    pub total_bits: u64,
}

impl SceneBitratePlan {
    /// Creates a new bitrate plan.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allocations: Vec::new(),
        }
    }

    /// Distributes a total bitrate budget across scenes weighted by encode params.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn distribute(
        scenes: &[SceneMetrics],
        params: &[SceneEncodeParams],
        total_bitrate_bps: u64,
        fps: f64,
    ) -> Self {
        if scenes.is_empty() || params.is_empty() || fps <= 0.0 {
            return Self::new();
        }

        let len = scenes.len().min(params.len());

        // Compute weighted frame counts
        let weighted_total: f64 = (0..len)
            .map(|i| scenes[i].frame_count() as f64 * params[i].bitrate_weight)
            .sum();

        if weighted_total <= 0.0 {
            return Self::new();
        }

        let total_frames: u64 = scenes[..len].iter().map(|s| s.frame_count()).sum();
        let total_bits = (total_bitrate_bps as f64 * total_frames as f64 / fps) as u64;

        let mut allocations = Vec::with_capacity(len);
        for i in 0..len {
            let weight = scenes[i].frame_count() as f64 * params[i].bitrate_weight / weighted_total;
            let scene_bits = (total_bits as f64 * weight) as u64;
            let scene_bps = if scenes[i].frame_count() > 0 {
                (scene_bits as f64 * fps / scenes[i].frame_count() as f64) as u64
            } else {
                0
            };

            allocations.push(SceneBitrateAlloc {
                scene_index: scenes[i].scene_index,
                allocated_bps: scene_bps,
                frame_count: scenes[i].frame_count(),
                total_bits: scene_bits,
            });
        }

        Self { allocations }
    }

    /// Returns the allocations.
    #[must_use]
    pub fn allocations(&self) -> &[SceneBitrateAlloc] {
        &self.allocations
    }

    /// Returns the number of scene allocations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.allocations.len()
    }

    /// Returns true if no allocations exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    /// Returns total allocated bits.
    #[must_use]
    pub fn total_bits(&self) -> u64 {
        self.allocations.iter().map(|a| a.total_bits).sum()
    }
}

impl Default for SceneBitratePlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene(index: u32, start: u64, end: u64, st: SceneType) -> SceneMetrics {
        let mut m = SceneMetrics::new(index, start, end);
        m.scene_type = st;
        m.spatial_complexity = match st {
            SceneType::Static => 0.1,
            SceneType::Dialogue => 0.3,
            SceneType::Moderate => 0.5,
            SceneType::Action => 0.8,
            SceneType::HighDetail => 0.9,
            SceneType::DarkScene => 0.4,
            SceneType::Transition => 0.6,
        };
        m.temporal_complexity = match st {
            SceneType::Static => 0.05,
            SceneType::Dialogue => 0.2,
            SceneType::Moderate => 0.5,
            SceneType::Action => 0.9,
            SceneType::HighDetail => 0.4,
            SceneType::DarkScene => 0.3,
            SceneType::Transition => 0.7,
        };
        if st == SceneType::DarkScene {
            m.avg_luminance = 0.1;
        }
        m
    }

    #[test]
    fn test_scene_metrics_frame_count() {
        let m = SceneMetrics::new(0, 10, 50);
        assert_eq!(m.frame_count(), 40);
    }

    #[test]
    fn test_scene_metrics_combined_complexity() {
        let mut m = SceneMetrics::new(0, 0, 100);
        m.spatial_complexity = 0.8;
        m.temporal_complexity = 0.6;
        m.texture_density = 0.5;
        let cc = m.combined_complexity();
        // 0.8*0.4 + 0.6*0.4 + 0.5*0.2 = 0.32 + 0.24 + 0.10 = 0.66
        assert!((cc - 0.66).abs() < 1e-6);
    }

    #[test]
    fn test_scene_metrics_is_dark() {
        let mut m = SceneMetrics::new(0, 0, 100);
        m.avg_luminance = 0.1;
        assert!(m.is_dark());
        m.avg_luminance = 0.5;
        assert!(!m.is_dark());
    }

    #[test]
    fn test_scene_encode_params_default() {
        let p = SceneEncodeParams::default();
        assert!((p.qp_offset - 0.0).abs() < 1e-9);
        assert!((p.bitrate_weight - 1.0).abs() < 1e-9);
        assert_eq!(p.gop_size, 250);
    }

    #[test]
    fn test_effective_qp() {
        let p = SceneEncodeParams {
            qp_offset: -3.0,
            min_qp: 10.0,
            max_qp: 45.0,
            ..SceneEncodeParams::default()
        };
        assert!((p.effective_qp(28.0) - 25.0).abs() < 1e-9);
        // Clamp to min
        assert!((p.effective_qp(11.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_scene_encoder_new() {
        let enc = SceneEncoder::new(26.0, 5_000_000, 30.0);
        assert!((enc.base_qp() - 26.0).abs() < 1e-9);
        assert_eq!(enc.target_bitrate_bps(), 5_000_000);
    }

    #[test]
    fn test_generate_params_static() {
        let enc = SceneEncoder::new(26.0, 5_000_000, 30.0);
        let scene = make_scene(0, 0, 300, SceneType::Static);
        let params = enc.generate_params(&scene);
        assert!(params.qp_offset < 0.0); // Should boost quality
        assert!(params.gop_size <= 300);
    }

    #[test]
    fn test_generate_params_action() {
        let enc = SceneEncoder::new(26.0, 5_000_000, 30.0);
        let scene = make_scene(0, 0, 200, SceneType::Action);
        let params = enc.generate_params(&scene);
        assert!(params.qp_offset > 0.0); // Should relax
        assert!(params.b_frames <= 3);
    }

    #[test]
    fn test_generate_params_dark() {
        let enc = SceneEncoder::new(26.0, 5_000_000, 30.0);
        let scene = make_scene(0, 0, 200, SceneType::DarkScene);
        let params = enc.generate_params(&scene);
        assert!(params.aq_strength > 1.0); // Stronger AQ for dark
    }

    #[test]
    fn test_process_scene() {
        let mut enc = SceneEncoder::new(26.0, 5_000_000, 30.0);
        let scene = make_scene(0, 0, 100, SceneType::Moderate);
        let _params = enc.process_scene(scene);
        assert_eq!(enc.scenes_processed(), 1);
    }

    #[test]
    fn test_avg_scene_complexity() {
        let mut enc = SceneEncoder::new(26.0, 5_000_000, 30.0);
        enc.process_scene(make_scene(0, 0, 100, SceneType::Static));
        enc.process_scene(make_scene(1, 100, 200, SceneType::Action));
        let avg = enc.avg_scene_complexity();
        assert!(avg > 0.0 && avg < 1.0);
    }

    #[test]
    fn test_bitrate_plan_distribute() {
        let scenes = vec![
            make_scene(0, 0, 100, SceneType::Static),
            make_scene(1, 100, 300, SceneType::Action),
        ];
        let enc = SceneEncoder::new(26.0, 5_000_000, 30.0);
        let params: Vec<_> = scenes.iter().map(|s| enc.generate_params(s)).collect();

        let plan = SceneBitratePlan::distribute(&scenes, &params, 5_000_000, 30.0);
        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());
        assert!(plan.total_bits() > 0);
    }

    #[test]
    fn test_bitrate_plan_empty() {
        let plan = SceneBitratePlan::distribute(&[], &[], 5_000_000, 30.0);
        assert!(plan.is_empty());
        assert_eq!(plan.total_bits(), 0);
    }

    #[test]
    fn test_bitrate_plan_action_gets_more_bits() {
        let scenes = vec![
            make_scene(0, 0, 100, SceneType::Static),
            make_scene(1, 100, 200, SceneType::Action),
        ];
        let enc = SceneEncoder::new(26.0, 5_000_000, 30.0);
        let params: Vec<_> = scenes.iter().map(|s| enc.generate_params(s)).collect();

        let plan = SceneBitratePlan::distribute(&scenes, &params, 5_000_000, 30.0);
        let allocs = plan.allocations();
        // Action scene should get higher bitrate due to higher weight
        assert!(allocs[1].allocated_bps > allocs[0].allocated_bps);
    }

    // --- New tests for LookaheadSceneQp ---

    #[test]
    fn test_lookahead_scene_qp_creation() {
        let la = LookaheadSceneQp::new(20, 28.0);
        assert_eq!(la.frames_processed(), 0);
        assert!((la.base_qp - 28.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lookahead_scene_qp_no_lookahead() {
        let mut la = LookaheadSceneQp::new(10, 28.0);
        let frame = FrameLookaheadInfo::new(0);
        let result = la.analyze_frame(&frame);
        // With no lookahead data and neutral complexity, should be near base QP
        assert!(
            (result.recommended_qp - 28.0).abs() < 7.0,
            "QP should be near base: {}",
            result.recommended_qp
        );
    }

    #[test]
    fn test_lookahead_scene_cut_boost() {
        let mut la = LookaheadSceneQp::new(10, 28.0);
        la.set_scene_cut_boost(4.0);

        // Feed some normal frames
        for i in 0..5 {
            let mut info = FrameLookaheadInfo::new(i);
            info.spatial_complexity = 0.5;
            info.temporal_complexity = 0.5;
            la.feed_frame(info.clone());
            la.analyze_frame(&info);
        }

        // Now a scene cut frame
        let mut cut_frame = FrameLookaheadInfo::new(5);
        cut_frame.is_scene_cut = true;
        cut_frame.spatial_complexity = 0.6;
        la.feed_frame(cut_frame.clone());
        let result = la.analyze_frame(&cut_frame);

        assert!(result.near_scene_cut, "Should detect scene cut");
        // Scene cut should lower QP (boost quality)
        assert!(
            result.qp_delta < 0.0,
            "Scene cut should produce negative QP delta: {}",
            result.qp_delta
        );
    }

    #[test]
    fn test_lookahead_upcoming_complexity_high() {
        let mut la = LookaheadSceneQp::new(10, 28.0);

        // Feed high complexity frames into lookahead
        for i in 0..10 {
            let mut info = FrameLookaheadInfo::new(i);
            info.spatial_complexity = 0.9;
            info.temporal_complexity = 0.8;
            la.feed_frame(info);
        }

        let frame = FrameLookaheadInfo::new(0);
        let result = la.analyze_frame(&frame);
        // High upcoming complexity should produce negative delta (more bits)
        assert!(
            result.upcoming_complexity > 0.7,
            "Upcoming complexity should be high: {}",
            result.upcoming_complexity
        );
        assert!(
            result.qp_delta < 0.0,
            "High complexity should lower QP: {}",
            result.qp_delta
        );
    }

    #[test]
    fn test_lookahead_upcoming_complexity_low() {
        let mut la = LookaheadSceneQp::new(10, 28.0);

        // Feed low complexity frames into lookahead
        for i in 0..10 {
            let mut info = FrameLookaheadInfo::new(i);
            info.spatial_complexity = 0.1;
            info.temporal_complexity = 0.1;
            la.feed_frame(info);
        }

        let frame = FrameLookaheadInfo::new(0);
        let result = la.analyze_frame(&frame);
        // Low upcoming complexity should produce positive delta (save bits)
        assert!(
            result.upcoming_complexity < 0.3,
            "Upcoming complexity should be low: {}",
            result.upcoming_complexity
        );
        assert!(
            result.qp_delta > 0.0,
            "Low complexity should raise QP: {}",
            result.qp_delta
        );
    }

    #[test]
    fn test_lookahead_distance_to_cut() {
        let mut la = LookaheadSceneQp::new(10, 28.0);

        // Feed 5 normal frames, then a scene cut
        for i in 0..5 {
            la.feed_frame(FrameLookaheadInfo::new(i));
        }
        let mut cut = FrameLookaheadInfo::new(5);
        cut.is_scene_cut = true;
        la.feed_frame(cut);

        let frame = FrameLookaheadInfo::new(0);
        let result = la.analyze_frame(&frame);
        assert!(
            result.distance_to_next_cut.is_some(),
            "Should find upcoming scene cut"
        );
    }

    #[test]
    fn test_lookahead_dark_scene_boost() {
        let mut la = LookaheadSceneQp::new(10, 28.0);

        let mut dark_frame = FrameLookaheadInfo::new(0);
        dark_frame.avg_luminance = 0.1;
        dark_frame.spatial_complexity = 0.5;
        dark_frame.temporal_complexity = 0.5;
        la.feed_frame(dark_frame.clone());

        let result = la.analyze_frame(&dark_frame);
        // Dark scene should get quality boost (lower QP)
        assert!(
            result.recommended_qp < 28.0 + 1.0,
            "Dark scene should get lower QP: {}",
            result.recommended_qp
        );
    }

    #[test]
    fn test_lookahead_qp_clamped() {
        let mut la = LookaheadSceneQp::new(10, 28.0);
        la.set_max_qp_delta(3.0);

        // Very high complexity
        for i in 0..10 {
            let mut info = FrameLookaheadInfo::new(i);
            info.spatial_complexity = 1.0;
            info.temporal_complexity = 1.0;
            la.feed_frame(info);
        }

        let frame = FrameLookaheadInfo::new(0);
        let result = la.analyze_frame(&frame);
        assert!(
            result.qp_delta >= -3.0 && result.qp_delta <= 3.0,
            "QP delta should be clamped to max: {}",
            result.qp_delta
        );
        assert!(result.recommended_qp >= 1.0 && result.recommended_qp <= 51.0);
    }

    #[test]
    fn test_lookahead_smoothing() {
        let mut la = LookaheadSceneQp::new(10, 28.0);
        la.set_smoothing(0.5);

        // Process a static frame
        let mut info1 = FrameLookaheadInfo::new(0);
        info1.spatial_complexity = 0.1;
        info1.temporal_complexity = 0.1;
        la.feed_frame(info1.clone());
        let r1 = la.analyze_frame(&info1);

        // Then a complex frame (non-scene-cut)
        let mut info2 = FrameLookaheadInfo::new(1);
        info2.spatial_complexity = 0.9;
        info2.temporal_complexity = 0.9;
        la.feed_frame(info2.clone());
        let r2 = la.analyze_frame(&info2);

        // Due to smoothing, the jump should be dampened
        let delta_change = (r2.qp_delta - r1.qp_delta).abs();
        assert!(
            delta_change < 12.0,
            "Smoothing should dampen QP changes: delta_change={}",
            delta_change
        );
    }

    #[test]
    fn test_lookahead_reset() {
        let mut la = LookaheadSceneQp::new(10, 28.0);
        la.feed_frame(FrameLookaheadInfo::new(0));
        la.analyze_frame(&FrameLookaheadInfo::new(0));
        la.reset();
        assert_eq!(la.frames_processed(), 0);
    }

    #[test]
    fn test_frame_lookahead_info_combined_complexity() {
        let mut info = FrameLookaheadInfo::new(0);
        info.spatial_complexity = 0.8;
        info.temporal_complexity = 0.4;
        let cc = info.combined_complexity();
        assert!((cc - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_lookahead_multiple_scene_cuts() {
        let mut la = LookaheadSceneQp::new(20, 28.0);

        // Simulate: normal -> cut -> normal -> cut
        for i in 0..5 {
            la.feed_frame(FrameLookaheadInfo::new(i));
        }
        let mut cut1 = FrameLookaheadInfo::new(5);
        cut1.is_scene_cut = true;
        la.feed_frame(cut1);
        for i in 6..10 {
            la.feed_frame(FrameLookaheadInfo::new(i));
        }
        let mut cut2 = FrameLookaheadInfo::new(10);
        cut2.is_scene_cut = true;
        la.feed_frame(cut2);

        let frame = FrameLookaheadInfo::new(0);
        let result = la.analyze_frame(&frame);
        // Should find the first upcoming cut
        assert!(result.distance_to_next_cut.is_some());
    }
}
