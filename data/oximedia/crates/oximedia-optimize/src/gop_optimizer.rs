#![allow(dead_code)]
//! GOP structure optimization and analysis.
//!
//! This module provides tools for optimizing the Group of Pictures (GOP)
//! structure in video encoding. It analyzes scene content, motion patterns,
//! and complexity to determine optimal GOP lengths, B-frame patterns, and
//! key frame placement. Supports both fixed and adaptive GOP strategies.
//!
//! Content-adaptive GOP selection uses a classifier that maps content features
//! (motion, complexity, temporal correlation, scene type) to optimal GOP
//! parameters, including length, B-frame pattern, and reference distance.

use std::collections::VecDeque;

use crate::ContentType;

/// GOP structure pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GopPattern {
    /// I-P only (no B-frames).
    IpOnly,
    /// I-B-P with one B-frame between references.
    Ibp,
    /// I-B-B-P with two B-frames between references.
    Ibbp,
    /// I-B-B-B-P with three B-frames between references.
    Ibbbp,
    /// Hierarchical B-frame structure.
    Hierarchical,
}

impl GopPattern {
    /// Returns the number of B-frames between reference frames.
    #[must_use]
    pub fn b_frame_count(&self) -> u32 {
        match self {
            Self::IpOnly => 0,
            Self::Ibp => 1,
            Self::Ibbp => 2,
            Self::Ibbbp => 3,
            Self::Hierarchical => 3,
        }
    }

    /// Returns the reference distance (distance between anchor frames).
    #[must_use]
    pub fn reference_distance(&self) -> u32 {
        self.b_frame_count() + 1
    }
}

/// Scene analysis result for GOP placement decisions.
#[derive(Debug, Clone)]
pub struct SceneInfo {
    /// Frame index where this scene starts.
    pub start_frame: u64,
    /// Estimated scene complexity (0.0 to 1.0).
    pub complexity: f64,
    /// Average motion magnitude.
    pub motion_magnitude: f64,
    /// Whether this is a scene change boundary.
    pub is_scene_change: bool,
    /// Estimated temporal correlation with the next frame.
    pub temporal_correlation: f64,
}

impl SceneInfo {
    /// Creates a new scene info entry.
    #[must_use]
    pub fn new(start_frame: u64) -> Self {
        Self {
            start_frame,
            complexity: 0.5,
            motion_magnitude: 0.0,
            is_scene_change: false,
            temporal_correlation: 0.8,
        }
    }

    /// Returns true if the scene has high motion.
    #[must_use]
    pub fn is_high_motion(&self) -> bool {
        self.motion_magnitude > 15.0
    }

    /// Returns true if the scene has low complexity (easy to encode).
    #[must_use]
    pub fn is_low_complexity(&self) -> bool {
        self.complexity < 0.3
    }
}

/// Configuration for the GOP optimizer.
#[derive(Debug, Clone)]
pub struct GopOptimizerConfig {
    /// Minimum GOP length in frames.
    pub min_gop_length: u32,
    /// Maximum GOP length in frames.
    pub max_gop_length: u32,
    /// Default GOP pattern.
    pub default_pattern: GopPattern,
    /// Scene change detection threshold (0.0 to 1.0).
    pub scene_change_threshold: f64,
    /// Whether to allow adaptive GOP length.
    pub adaptive_gop: bool,
    /// Whether to force key frames at scene changes.
    pub keyframe_at_scene_change: bool,
    /// Minimum scene duration in frames before a key frame.
    pub min_scene_frames: u32,
    /// Content type hint for adaptive selection.
    pub content_type: ContentType,
}

impl Default for GopOptimizerConfig {
    fn default() -> Self {
        Self {
            min_gop_length: 12,
            max_gop_length: 250,
            default_pattern: GopPattern::Ibbp,
            scene_change_threshold: 0.5,
            adaptive_gop: true,
            keyframe_at_scene_change: true,
            min_scene_frames: 8,
            content_type: ContentType::Generic,
        }
    }
}

/// A planned GOP structure.
#[derive(Debug, Clone)]
pub struct GopPlan {
    /// Start frame index.
    pub start_frame: u64,
    /// Length of the GOP in frames.
    pub length: u32,
    /// Pattern to use.
    pub pattern: GopPattern,
    /// Whether this GOP starts with a forced keyframe.
    pub forced_keyframe: bool,
    /// Estimated average complexity of this GOP.
    pub avg_complexity: f64,
}

impl GopPlan {
    /// Creates a new GOP plan.
    #[must_use]
    pub fn new(start_frame: u64, length: u32, pattern: GopPattern) -> Self {
        Self {
            start_frame,
            length,
            pattern,
            forced_keyframe: false,
            avg_complexity: 0.5,
        }
    }

    /// Returns the frame index of the last frame in this GOP.
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.start_frame + u64::from(self.length) - 1
    }

    /// Returns the number of reference frames (I + P frames).
    #[must_use]
    pub fn reference_frame_count(&self) -> u32 {
        let ref_dist = self.pattern.reference_distance();
        if ref_dist == 0 {
            return self.length;
        }
        // 1 (I-frame) + number of P-frames
        1 + (self.length.saturating_sub(1)) / ref_dist
    }

    /// Returns the estimated number of B-frames.
    #[must_use]
    pub fn b_frame_count(&self) -> u32 {
        self.length.saturating_sub(self.reference_frame_count())
    }
}

/// Content-adaptive GOP selection classifier.
///
/// Maps content features to optimal GOP parameters based on content type
/// and measured scene characteristics.
#[derive(Debug, Clone)]
pub struct ContentAdaptiveGop {
    /// Content type hint.
    content_type: ContentType,
    /// Minimum GOP length.
    min_length: u32,
    /// Maximum GOP length.
    max_length: u32,
}

impl ContentAdaptiveGop {
    /// Creates a new content-adaptive GOP selector.
    #[must_use]
    pub fn new(content_type: ContentType, min_length: u32, max_length: u32) -> Self {
        Self {
            content_type,
            min_length,
            max_length,
        }
    }

    /// Selects GOP structure based on content type and scene features.
    ///
    /// Uses a decision tree that considers:
    /// - Content type (animation, film, screen, generic)
    /// - Average motion magnitude
    /// - Average complexity
    /// - Temporal correlation between frames
    #[must_use]
    pub fn select_gop(
        &self,
        avg_complexity: f64,
        avg_motion: f64,
        avg_temporal_corr: f64,
    ) -> ContentGopDecision {
        // First, apply content-type-specific base rules
        let (base_pattern, base_length) = self.content_type_base_rules();

        // Then adapt based on measured features
        let pattern =
            self.adapt_pattern(base_pattern, avg_motion, avg_complexity, avg_temporal_corr);
        let length = self.adapt_length(base_length, avg_motion, avg_complexity, avg_temporal_corr);

        ContentGopDecision {
            pattern,
            gop_length: length.clamp(self.min_length, self.max_length),
            use_hierarchical_refs: self.should_use_hierarchical(avg_temporal_corr, pattern),
            recommended_ref_frames: self.recommend_ref_frames(pattern, avg_motion),
        }
    }

    /// Returns base GOP rules for the content type.
    fn content_type_base_rules(&self) -> (GopPattern, u32) {
        match self.content_type {
            ContentType::Animation => {
                // Animation: sharp edges, flat areas, high temporal correlation
                // Use longer GOPs with more B-frames for compression
                (GopPattern::Ibbbp, 120)
            }
            ContentType::Film => {
                // Film: grain, natural textures, moderate motion
                // Standard pattern, moderate GOP length
                (GopPattern::Ibbp, 96)
            }
            ContentType::Screen => {
                // Screen: text, graphics, often static
                // Fewer B-frames (text can be damaged), longer GOPs for static
                (GopPattern::Ibp, 200)
            }
            ContentType::Generic => {
                // Generic: balanced defaults
                (GopPattern::Ibbp, 72)
            }
        }
    }

    /// Adapts the GOP pattern based on measured features.
    fn adapt_pattern(
        &self,
        base: GopPattern,
        avg_motion: f64,
        avg_complexity: f64,
        avg_temporal_corr: f64,
    ) -> GopPattern {
        // Very high motion: reduce B-frames to avoid prediction failures
        if avg_motion > 40.0 {
            return GopPattern::IpOnly;
        }
        if avg_motion > 25.0 {
            return match base {
                GopPattern::Ibbbp | GopPattern::Hierarchical => GopPattern::Ibp,
                GopPattern::Ibbp => GopPattern::Ibp,
                other => other,
            };
        }

        // High temporal correlation + low complexity: more B-frames for compression
        if avg_temporal_corr > 0.9 && avg_complexity < 0.3 {
            return match base {
                GopPattern::IpOnly => GopPattern::Ibp,
                GopPattern::Ibp => GopPattern::Ibbp,
                GopPattern::Ibbp => GopPattern::Ibbbp,
                other => other,
            };
        }

        // Low temporal correlation: fewer B-frames
        if avg_temporal_corr < 0.4 {
            return match base {
                GopPattern::Ibbbp | GopPattern::Hierarchical => GopPattern::Ibp,
                GopPattern::Ibbp => GopPattern::Ibp,
                other => other,
            };
        }

        // High complexity with moderate correlation: standard pattern
        if avg_complexity > 0.7 && avg_temporal_corr > 0.5 {
            return GopPattern::Ibbp;
        }

        base
    }

    /// Adapts GOP length based on measured features.
    fn adapt_length(
        &self,
        base: u32,
        avg_motion: f64,
        avg_complexity: f64,
        avg_temporal_corr: f64,
    ) -> u32 {
        let mut length = base;

        // High motion: shorter GOPs for better random access and adaptation
        if avg_motion > 20.0 {
            length = (length as f64 * 0.5) as u32;
        } else if avg_motion > 10.0 {
            length = (length as f64 * 0.75) as u32;
        }

        // Low complexity + high correlation: longer GOPs (more compression)
        if avg_complexity < 0.3 && avg_temporal_corr > 0.85 {
            length = (length as f64 * 1.5).min(self.max_length as f64) as u32;
        }

        // Very high complexity: shorter GOPs for better quality adaptation
        if avg_complexity > 0.8 {
            length = (length as f64 * 0.7) as u32;
        }

        length.max(self.min_length)
    }

    /// Determines if hierarchical reference structure should be used.
    fn should_use_hierarchical(&self, avg_temporal_corr: f64, pattern: GopPattern) -> bool {
        // Hierarchical refs benefit when temporal correlation is high
        // and there are enough B-frames to build the hierarchy
        avg_temporal_corr > 0.7 && pattern.b_frame_count() >= 2
    }

    /// Recommends the number of reference frames.
    fn recommend_ref_frames(&self, pattern: GopPattern, avg_motion: f64) -> u32 {
        let base_refs = match pattern {
            GopPattern::IpOnly => 2,
            GopPattern::Ibp => 3,
            GopPattern::Ibbp => 4,
            GopPattern::Ibbbp | GopPattern::Hierarchical => 5,
        };

        // High motion benefits from more reference frames
        if avg_motion > 15.0 {
            (base_refs + 1).min(8)
        } else {
            base_refs
        }
    }
}

/// Result of content-adaptive GOP selection.
#[derive(Debug, Clone)]
pub struct ContentGopDecision {
    /// Selected GOP pattern.
    pub pattern: GopPattern,
    /// Recommended GOP length in frames.
    pub gop_length: u32,
    /// Whether to use hierarchical reference structure.
    pub use_hierarchical_refs: bool,
    /// Recommended number of reference frames.
    pub recommended_ref_frames: u32,
}

/// GOP optimizer that analyzes scene data and produces GOP plans.
#[derive(Debug)]
pub struct GopOptimizer {
    config: GopOptimizerConfig,
    /// Scene info buffer for lookahead analysis.
    scene_buffer: VecDeque<SceneInfo>,
    /// Planned GOPs.
    plans: Vec<GopPlan>,
    /// Current frame position.
    current_frame: u64,
    /// Frames since last key frame.
    frames_since_keyframe: u32,
    /// Content-adaptive GOP selector (if enabled).
    adaptive_selector: Option<ContentAdaptiveGop>,
}

impl GopOptimizer {
    /// Creates a new GOP optimizer.
    #[must_use]
    pub fn new(config: GopOptimizerConfig) -> Self {
        let adaptive_selector = if config.adaptive_gop {
            Some(ContentAdaptiveGop::new(
                config.content_type,
                config.min_gop_length,
                config.max_gop_length,
            ))
        } else {
            None
        };

        Self {
            config,
            scene_buffer: VecDeque::new(),
            plans: Vec::new(),
            current_frame: 0,
            frames_since_keyframe: 0,
            adaptive_selector,
        }
    }

    /// Feeds scene information for the next frame.
    pub fn feed_scene_info(&mut self, info: SceneInfo) {
        self.scene_buffer.push_back(info);
    }

    /// Decides whether to place a key frame at the current position.
    #[must_use]
    pub fn should_place_keyframe(&self, info: &SceneInfo) -> bool {
        // Force keyframe if max GOP length reached
        if self.frames_since_keyframe >= self.config.max_gop_length {
            return true;
        }

        // Don't place keyframe if minimum not reached
        if self.frames_since_keyframe < self.config.min_gop_length {
            return false;
        }

        // Place at scene change if enabled
        if self.config.keyframe_at_scene_change
            && info.is_scene_change
            && self.frames_since_keyframe >= self.config.min_scene_frames
        {
            return true;
        }

        false
    }

    /// Selects the best GOP pattern based on scene characteristics.
    ///
    /// When content-adaptive mode is enabled, delegates to [`ContentAdaptiveGop`].
    /// Otherwise falls back to the simple heuristic.
    #[must_use]
    pub fn select_pattern(&self, avg_complexity: f64, avg_motion: f64) -> GopPattern {
        if !self.config.adaptive_gop {
            return self.config.default_pattern;
        }

        // High motion: fewer B-frames for better prediction
        if avg_motion > 20.0 {
            return GopPattern::Ibp;
        }

        // Very high motion: no B-frames
        if avg_motion > 40.0 {
            return GopPattern::IpOnly;
        }

        // Low complexity: can afford more B-frames for compression
        if avg_complexity < 0.3 {
            return GopPattern::Ibbbp;
        }

        // High complexity: moderate B-frames
        if avg_complexity > 0.7 {
            return GopPattern::Ibp;
        }

        // Default: standard pattern
        self.config.default_pattern
    }

    /// Plans a GOP starting at the current position with content-adaptive selection.
    ///
    /// Uses the [`ContentAdaptiveGop`] classifier when available, falling back to
    /// simple heuristics otherwise.
    pub fn plan_gop(&mut self, scene_infos: &[SceneInfo]) -> GopPlan {
        let start_frame = self.current_frame;

        // Calculate average complexity, motion, and temporal correlation
        #[allow(clippy::cast_precision_loss)]
        let (avg_complexity, avg_motion, avg_temporal_corr) = if scene_infos.is_empty() {
            (0.5, 5.0, 0.8)
        } else {
            let c =
                scene_infos.iter().map(|s| s.complexity).sum::<f64>() / scene_infos.len() as f64;
            let m = scene_infos.iter().map(|s| s.motion_magnitude).sum::<f64>()
                / scene_infos.len() as f64;
            let t = scene_infos
                .iter()
                .map(|s| s.temporal_correlation)
                .sum::<f64>()
                / scene_infos.len() as f64;
            (c, m, t)
        };

        // Use content-adaptive selector if available
        let (pattern, adaptive_length) = if let Some(ref selector) = self.adaptive_selector {
            let decision = selector.select_gop(avg_complexity, avg_motion, avg_temporal_corr);
            (decision.pattern, Some(decision.gop_length))
        } else {
            (self.select_pattern(avg_complexity, avg_motion), None)
        };

        // Determine GOP length
        let mut length = adaptive_length.unwrap_or(self.config.max_gop_length);
        for (i, info) in scene_infos.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let frame_offset = i as u32;
            if frame_offset >= self.config.min_gop_length
                && info.is_scene_change
                && self.config.keyframe_at_scene_change
            {
                length = frame_offset;
                break;
            }
        }
        length = length.clamp(self.config.min_gop_length, self.config.max_gop_length);

        let mut plan = GopPlan::new(start_frame, length, pattern);
        plan.avg_complexity = avg_complexity;
        plan.forced_keyframe = self.frames_since_keyframe >= self.config.max_gop_length;

        self.current_frame += u64::from(length);
        self.frames_since_keyframe = 0;
        self.plans.push(plan.clone());

        plan
    }

    /// Returns all planned GOPs.
    #[must_use]
    pub fn plans(&self) -> &[GopPlan] {
        &self.plans
    }

    /// Returns the current frame position.
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Returns the content-adaptive selector if enabled.
    #[must_use]
    pub fn adaptive_selector(&self) -> Option<&ContentAdaptiveGop> {
        self.adaptive_selector.as_ref()
    }

    /// Resets the optimizer state.
    pub fn reset(&mut self) {
        self.scene_buffer.clear();
        self.plans.clear();
        self.current_frame = 0;
        self.frames_since_keyframe = 0;
    }
}

/// Analyzes GOP efficiency based on actual encoding results.
#[derive(Debug, Clone)]
pub struct GopEfficiencyAnalysis {
    /// GOP index.
    pub gop_index: u32,
    /// Planned length vs actual length.
    pub planned_length: u32,
    /// Compression ratio for this GOP.
    pub compression_ratio: f64,
    /// Average bits per frame.
    pub avg_bits_per_frame: f64,
    /// I-frame to P-frame size ratio.
    pub i_to_p_ratio: f64,
    /// B-frame to P-frame size ratio.
    pub b_to_p_ratio: f64,
}

impl GopEfficiencyAnalysis {
    /// Creates a new efficiency analysis.
    #[must_use]
    pub fn new(gop_index: u32, planned_length: u32) -> Self {
        Self {
            gop_index,
            planned_length,
            compression_ratio: 0.0,
            avg_bits_per_frame: 0.0,
            i_to_p_ratio: 0.0,
            b_to_p_ratio: 0.0,
        }
    }

    /// Computes the efficiency analysis from frame sizes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_frame_sizes(
        gop_index: u32,
        i_frame_bits: u64,
        p_frame_bits: &[u64],
        b_frame_bits: &[u64],
    ) -> Self {
        let total_frames = 1 + p_frame_bits.len() + b_frame_bits.len();
        let total_bits: u64 =
            i_frame_bits + p_frame_bits.iter().sum::<u64>() + b_frame_bits.iter().sum::<u64>();

        let avg_bits_per_frame = if total_frames == 0 {
            0.0
        } else {
            total_bits as f64 / total_frames as f64
        };

        let avg_p = if p_frame_bits.is_empty() {
            1.0
        } else {
            p_frame_bits.iter().sum::<u64>() as f64 / p_frame_bits.len() as f64
        };

        let i_to_p_ratio = if avg_p > 0.0 {
            i_frame_bits as f64 / avg_p
        } else {
            0.0
        };

        let b_to_p_ratio = if b_frame_bits.is_empty() || avg_p <= 0.0 {
            0.0
        } else {
            let avg_b = b_frame_bits.iter().sum::<u64>() as f64 / b_frame_bits.len() as f64;
            avg_b / avg_p
        };

        #[allow(clippy::cast_possible_truncation)]
        Self {
            gop_index,
            planned_length: total_frames as u32,
            compression_ratio: 0.0,
            avg_bits_per_frame,
            i_to_p_ratio,
            b_to_p_ratio,
        }
    }

    /// Returns true if B-frames are providing good compression benefit.
    #[must_use]
    pub fn b_frames_effective(&self) -> bool {
        self.b_to_p_ratio > 0.0 && self.b_to_p_ratio < 0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gop_pattern_b_frame_count() {
        assert_eq!(GopPattern::IpOnly.b_frame_count(), 0);
        assert_eq!(GopPattern::Ibp.b_frame_count(), 1);
        assert_eq!(GopPattern::Ibbp.b_frame_count(), 2);
        assert_eq!(GopPattern::Ibbbp.b_frame_count(), 3);
        assert_eq!(GopPattern::Hierarchical.b_frame_count(), 3);
    }

    #[test]
    fn test_gop_pattern_reference_distance() {
        assert_eq!(GopPattern::IpOnly.reference_distance(), 1);
        assert_eq!(GopPattern::Ibp.reference_distance(), 2);
        assert_eq!(GopPattern::Ibbp.reference_distance(), 3);
    }

    #[test]
    fn test_scene_info_new() {
        let info = SceneInfo::new(100);
        assert_eq!(info.start_frame, 100);
        assert!(!info.is_scene_change);
        assert!((info.complexity - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scene_info_high_motion() {
        let mut info = SceneInfo::new(0);
        info.motion_magnitude = 5.0;
        assert!(!info.is_high_motion());
        info.motion_magnitude = 20.0;
        assert!(info.is_high_motion());
    }

    #[test]
    fn test_scene_info_low_complexity() {
        let mut info = SceneInfo::new(0);
        info.complexity = 0.1;
        assert!(info.is_low_complexity());
        info.complexity = 0.5;
        assert!(!info.is_low_complexity());
    }

    #[test]
    fn test_gop_plan_new() {
        let plan = GopPlan::new(0, 30, GopPattern::Ibbp);
        assert_eq!(plan.start_frame, 0);
        assert_eq!(plan.length, 30);
        assert_eq!(plan.pattern, GopPattern::Ibbp);
        assert!(!plan.forced_keyframe);
    }

    #[test]
    fn test_gop_plan_end_frame() {
        let plan = GopPlan::new(10, 30, GopPattern::Ibbp);
        assert_eq!(plan.end_frame(), 39);
    }

    #[test]
    fn test_gop_plan_reference_count_ip_only() {
        let plan = GopPlan::new(0, 10, GopPattern::IpOnly);
        // All frames are reference frames in IP-only
        assert_eq!(plan.reference_frame_count(), 10);
    }

    #[test]
    fn test_gop_plan_reference_count_ibbp() {
        let plan = GopPlan::new(0, 12, GopPattern::Ibbp);
        // ref_dist = 3, refs = 1 + (12-1)/3 = 1 + 3 = 4
        assert_eq!(plan.reference_frame_count(), 4);
    }

    #[test]
    fn test_gop_plan_b_frame_count() {
        let plan = GopPlan::new(0, 12, GopPattern::Ibbp);
        let refs = plan.reference_frame_count();
        let b_frames = plan.b_frame_count();
        assert_eq!(refs + b_frames, 12);
    }

    #[test]
    fn test_gop_optimizer_should_place_keyframe_max_length() {
        let config = GopOptimizerConfig {
            max_gop_length: 30,
            ..Default::default()
        };
        let mut optimizer = GopOptimizer::new(config);
        optimizer.frames_since_keyframe = 30;
        let info = SceneInfo::new(30);
        assert!(optimizer.should_place_keyframe(&info));
    }

    #[test]
    fn test_gop_optimizer_should_not_place_keyframe_too_early() {
        let config = GopOptimizerConfig {
            min_gop_length: 12,
            ..Default::default()
        };
        let optimizer = GopOptimizer::new(config);
        let mut info = SceneInfo::new(5);
        info.is_scene_change = true;
        assert!(!optimizer.should_place_keyframe(&info));
    }

    #[test]
    fn test_gop_optimizer_select_pattern_high_motion() {
        let config = GopOptimizerConfig::default();
        let optimizer = GopOptimizer::new(config);
        let pattern = optimizer.select_pattern(0.5, 25.0);
        assert_eq!(pattern, GopPattern::Ibp);
    }

    #[test]
    fn test_gop_optimizer_select_pattern_low_complexity() {
        let config = GopOptimizerConfig::default();
        let optimizer = GopOptimizer::new(config);
        let pattern = optimizer.select_pattern(0.2, 5.0);
        assert_eq!(pattern, GopPattern::Ibbbp);
    }

    #[test]
    fn test_gop_optimizer_plan_gop() {
        let config = GopOptimizerConfig {
            min_gop_length: 12,
            max_gop_length: 30,
            ..Default::default()
        };
        let mut optimizer = GopOptimizer::new(config);
        let scenes: Vec<SceneInfo> = (0..30).map(|i| SceneInfo::new(i)).collect();
        let plan = optimizer.plan_gop(&scenes);
        assert_eq!(plan.start_frame, 0);
        assert!(plan.length >= 12);
        assert!(plan.length <= 30);
    }

    #[test]
    fn test_gop_optimizer_reset() {
        let config = GopOptimizerConfig::default();
        let mut optimizer = GopOptimizer::new(config);
        let scenes: Vec<SceneInfo> = (0..20).map(|i| SceneInfo::new(i)).collect();
        let _ = optimizer.plan_gop(&scenes);
        optimizer.reset();
        assert_eq!(optimizer.current_frame(), 0);
        assert!(optimizer.plans().is_empty());
    }

    #[test]
    fn test_gop_efficiency_from_frame_sizes() {
        let i_bits = 50000_u64;
        let p_bits = vec![10000_u64, 12000, 11000];
        let b_bits = vec![4000_u64, 3500, 4500, 3800, 4200, 3900];
        let analysis = GopEfficiencyAnalysis::from_frame_sizes(0, i_bits, &p_bits, &b_bits);
        assert_eq!(analysis.planned_length, 10);
        assert!(analysis.avg_bits_per_frame > 0.0);
        assert!(analysis.i_to_p_ratio > 1.0); // I-frame should be bigger than P
        assert!(analysis.b_to_p_ratio < 1.0); // B-frame should be smaller than P
    }

    #[test]
    fn test_gop_efficiency_b_frames_effective() {
        let analysis = GopEfficiencyAnalysis {
            gop_index: 0,
            planned_length: 10,
            compression_ratio: 50.0,
            avg_bits_per_frame: 5000.0,
            i_to_p_ratio: 4.0,
            b_to_p_ratio: 0.4,
        };
        assert!(analysis.b_frames_effective());
    }

    #[test]
    fn test_gop_efficiency_b_frames_not_effective() {
        let analysis = GopEfficiencyAnalysis {
            gop_index: 0,
            planned_length: 10,
            compression_ratio: 50.0,
            avg_bits_per_frame: 5000.0,
            i_to_p_ratio: 4.0,
            b_to_p_ratio: 0.85,
        };
        assert!(!analysis.b_frames_effective());
    }

    // --- New tests for content-adaptive GOP ---

    #[test]
    fn test_content_adaptive_gop_animation() {
        let selector = ContentAdaptiveGop::new(ContentType::Animation, 12, 250);
        let decision = selector.select_gop(0.2, 3.0, 0.95);
        // Animation + low complexity + high correlation = many B-frames, long GOP
        assert!(
            decision.pattern.b_frame_count() >= 2,
            "Animation should get 2+ B-frames: {:?}",
            decision.pattern
        );
        assert!(
            decision.gop_length > 50,
            "Animation with high corr should get long GOP: {}",
            decision.gop_length
        );
    }

    #[test]
    fn test_content_adaptive_gop_screen() {
        let selector = ContentAdaptiveGop::new(ContentType::Screen, 12, 250);
        let decision = selector.select_gop(0.3, 2.0, 0.9);
        // Screen content: fewer B-frames to preserve text quality
        assert!(
            decision.pattern.b_frame_count() <= 2,
            "Screen content should get few B-frames: {:?}",
            decision.pattern
        );
    }

    #[test]
    fn test_content_adaptive_gop_high_motion() {
        let selector = ContentAdaptiveGop::new(ContentType::Generic, 12, 250);
        let decision = selector.select_gop(0.5, 45.0, 0.4);
        // Very high motion: IP-only or Ibp
        assert!(
            decision.pattern.b_frame_count() <= 1,
            "High motion should get few B-frames: {:?}",
            decision.pattern
        );
        // Shorter GOP
        assert!(
            decision.gop_length < 80,
            "High motion should get shorter GOP: {}",
            decision.gop_length
        );
    }

    #[test]
    fn test_content_adaptive_gop_static() {
        let selector = ContentAdaptiveGop::new(ContentType::Generic, 12, 250);
        let decision = selector.select_gop(0.1, 1.0, 0.98);
        // Static: long GOP, many B-frames
        assert!(
            decision.gop_length > 60,
            "Static content should get long GOP: {}",
            decision.gop_length
        );
        assert!(
            decision.pattern.b_frame_count() >= 2,
            "Static content should get B-frames: {:?}",
            decision.pattern
        );
    }

    #[test]
    fn test_content_adaptive_gop_action_film() {
        let selector = ContentAdaptiveGop::new(ContentType::Film, 12, 250);
        let decision = selector.select_gop(0.7, 22.0, 0.5);
        // Action in film: moderate B-frames, shorter GOP
        assert!(
            decision.pattern.b_frame_count() <= 2,
            "Action film should get moderate B-frames: {:?}",
            decision.pattern
        );
    }

    #[test]
    fn test_content_adaptive_hierarchical_refs() {
        let selector = ContentAdaptiveGop::new(ContentType::Film, 12, 250);
        // High correlation + enough B-frames: should use hierarchical
        let decision = selector.select_gop(0.4, 5.0, 0.85);
        if decision.pattern.b_frame_count() >= 2 {
            assert!(
                decision.use_hierarchical_refs,
                "Should use hierarchical refs with high corr + B-frames"
            );
        }
    }

    #[test]
    fn test_content_adaptive_no_hierarchical_low_corr() {
        let selector = ContentAdaptiveGop::new(ContentType::Generic, 12, 250);
        let decision = selector.select_gop(0.5, 30.0, 0.3);
        assert!(
            !decision.use_hierarchical_refs,
            "Should not use hierarchical refs with low temporal correlation"
        );
    }

    #[test]
    fn test_content_adaptive_ref_frames() {
        let selector = ContentAdaptiveGop::new(ContentType::Generic, 12, 250);
        let decision_low = selector.select_gop(0.5, 5.0, 0.7);
        let decision_high = selector.select_gop(0.5, 25.0, 0.5);
        // High motion should recommend more reference frames
        assert!(
            decision_high.recommended_ref_frames >= decision_low.recommended_ref_frames,
            "High motion should get >= ref frames: {} vs {}",
            decision_high.recommended_ref_frames,
            decision_low.recommended_ref_frames
        );
    }

    #[test]
    fn test_gop_length_clamped() {
        let selector = ContentAdaptiveGop::new(ContentType::Animation, 24, 120);
        let decision = selector.select_gop(0.1, 0.5, 0.99);
        assert!(
            decision.gop_length >= 24,
            "GOP length should be >= min: {}",
            decision.gop_length
        );
        assert!(
            decision.gop_length <= 120,
            "GOP length should be <= max: {}",
            decision.gop_length
        );
    }

    #[test]
    fn test_optimizer_uses_adaptive_selector() {
        let config = GopOptimizerConfig {
            adaptive_gop: true,
            content_type: ContentType::Animation,
            min_gop_length: 12,
            max_gop_length: 120,
            ..Default::default()
        };
        let mut optimizer = GopOptimizer::new(config);
        assert!(optimizer.adaptive_selector().is_some());

        // Plan GOP with static content
        let scenes: Vec<SceneInfo> = (0..30)
            .map(|i| {
                let mut s = SceneInfo::new(i);
                s.complexity = 0.2;
                s.motion_magnitude = 2.0;
                s.temporal_correlation = 0.95;
                s
            })
            .collect();
        let plan = optimizer.plan_gop(&scenes);
        // Should use content-adaptive selection
        assert!(plan.length >= 12);
        assert!(plan.length <= 120);
    }

    #[test]
    fn test_optimizer_no_adaptive_when_disabled() {
        let config = GopOptimizerConfig {
            adaptive_gop: false,
            ..Default::default()
        };
        let optimizer = GopOptimizer::new(config);
        assert!(optimizer.adaptive_selector().is_none());
    }

    #[test]
    fn test_content_gop_decision_fields() {
        let decision = ContentGopDecision {
            pattern: GopPattern::Ibbp,
            gop_length: 72,
            use_hierarchical_refs: true,
            recommended_ref_frames: 4,
        };
        assert_eq!(decision.pattern, GopPattern::Ibbp);
        assert_eq!(decision.gop_length, 72);
        assert!(decision.use_hierarchical_refs);
        assert_eq!(decision.recommended_ref_frames, 4);
    }
}
