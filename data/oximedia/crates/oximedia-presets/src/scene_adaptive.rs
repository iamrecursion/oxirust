//! Per-scene adaptive preset selection.
//!
//! Analyzes content complexity on a per-scene basis and selects the optimal
//! encoding preset for each segment. This allows a single encode job to use
//! different quality/bitrate targets for simple scenes (e.g. static titles)
//! versus complex scenes (e.g. fast action).
//!
//! # Architecture
//!
//! 1. [`SceneComplexity`] — an enum describing how complex a scene is.
//! 2. [`SceneSegment`] — represents a contiguous portion of the timeline with
//!    uniform complexity characteristics.
//! 3. [`SceneAnalysisResult`] — the output of analysing a scene: complexity,
//!    motion level, texture density, and temporal variance.
//! 4. [`SceneAdaptivePreset`] — the main engine: given a library of presets
//!    (one per complexity tier) and a sequence of scene analyses, it selects
//!    the best preset for each scene.
//!
//! The module intentionally works with *preset IDs* so that callers can map
//! back to any `Preset`, `PresetConfig`, or `InheritedConfig`.

#![allow(dead_code)]

use std::collections::HashMap;

// ── Scene complexity classification ─────────────────────────────────────────

/// Broad complexity classification of a scene segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneComplexity {
    /// Very simple scene — static card, fade, solid colour.
    VeryLow,
    /// Low complexity — talking head, minimal motion.
    Low,
    /// Medium complexity — moderate motion, mixed content.
    Medium,
    /// High complexity — fast action, dense texture.
    High,
    /// Very high complexity — rapid motion with heavy grain/detail.
    VeryHigh,
}

impl SceneComplexity {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::VeryLow => "Very Low",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::VeryHigh => "Very High",
        }
    }

    /// Classify complexity from a 0.0-1.0 score.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        let clamped = score.clamp(0.0, 1.0);
        if clamped < 0.15 {
            Self::VeryLow
        } else if clamped < 0.35 {
            Self::Low
        } else if clamped < 0.60 {
            Self::Medium
        } else if clamped < 0.80 {
            Self::High
        } else {
            Self::VeryHigh
        }
    }

    /// Suggested bitrate multiplier relative to a baseline "medium" preset.
    ///
    /// VeryLow scenes need less bitrate; VeryHigh scenes need more.
    #[must_use]
    pub fn bitrate_multiplier(&self) -> f64 {
        match self {
            Self::VeryLow => 0.4,
            Self::Low => 0.65,
            Self::Medium => 1.0,
            Self::High => 1.4,
            Self::VeryHigh => 1.8,
        }
    }
}

// ── Scene segment ───────────────────────────────────────────────────────────

/// A contiguous segment of the timeline.
#[derive(Debug, Clone)]
pub struct SceneSegment {
    /// Start time of the segment in seconds.
    pub start_time: f64,
    /// End time of the segment in seconds.
    pub end_time: f64,
    /// Scene index (0-based).
    pub index: usize,
}

impl SceneSegment {
    /// Create a new scene segment.
    #[must_use]
    pub fn new(index: usize, start_time: f64, end_time: f64) -> Self {
        Self {
            start_time,
            end_time,
            index,
        }
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        (self.end_time - self.start_time).max(0.0)
    }
}

// ── Scene analysis result ───────────────────────────────────────────────────

/// Analysis result for a single scene.
#[derive(Debug, Clone)]
pub struct SceneAnalysisResult {
    /// The scene segment this analysis applies to.
    pub segment: SceneSegment,
    /// Overall complexity score 0.0 (trivial) to 1.0 (extreme).
    pub complexity_score: f64,
    /// Motion level 0.0 (static) to 1.0 (extreme motion).
    pub motion_level: f64,
    /// Texture density 0.0 (smooth) to 1.0 (very detailed).
    pub texture_density: f64,
    /// Temporal variance 0.0 (no change) to 1.0 (rapid flicker).
    pub temporal_variance: f64,
    /// Classified complexity.
    pub complexity: SceneComplexity,
}

impl SceneAnalysisResult {
    /// Create a new analysis result from raw scores.
    ///
    /// The overall `complexity_score` is computed as a weighted combination of
    /// motion, texture, and temporal variance.
    #[must_use]
    pub fn new(
        segment: SceneSegment,
        motion_level: f64,
        texture_density: f64,
        temporal_variance: f64,
    ) -> Self {
        let motion = motion_level.clamp(0.0, 1.0);
        let texture = texture_density.clamp(0.0, 1.0);
        let temporal = temporal_variance.clamp(0.0, 1.0);

        // Weighted combination: motion contributes most to encoding difficulty
        let complexity_score = (0.45 * motion + 0.35 * texture + 0.20 * temporal).clamp(0.0, 1.0);
        let complexity = SceneComplexity::from_score(complexity_score);

        Self {
            segment,
            complexity_score,
            motion_level: motion,
            texture_density: texture,
            temporal_variance: temporal,
            complexity,
        }
    }

    /// Create a simple analysis result from just a complexity score.
    #[must_use]
    pub fn from_score(segment: SceneSegment, score: f64) -> Self {
        let clamped = score.clamp(0.0, 1.0);
        Self {
            segment,
            complexity_score: clamped,
            motion_level: clamped,
            texture_density: clamped,
            temporal_variance: clamped,
            complexity: SceneComplexity::from_score(clamped),
        }
    }
}

// ── Preset selection result ─────────────────────────────────────────────────

/// The result of adaptive preset selection for a single scene.
#[derive(Debug, Clone)]
pub struct AdaptivePresetSelection {
    /// Scene index.
    pub scene_index: usize,
    /// Start time (seconds).
    pub start_time: f64,
    /// End time (seconds).
    pub end_time: f64,
    /// Selected preset ID.
    pub preset_id: String,
    /// The complexity tier that drove the selection.
    pub complexity: SceneComplexity,
    /// Suggested bitrate for this scene (derived from base bitrate * multiplier).
    pub suggested_bitrate: u64,
}

// ── SceneAdaptivePreset engine ──────────────────────────────────────────────

/// Engine that maps scene analysis results to optimal presets.
///
/// The caller registers a preset ID for each complexity tier. The engine
/// then assigns presets to scenes based on their analysed complexity.
pub struct SceneAdaptivePreset {
    /// Mapping from complexity tier to preset ID.
    tier_presets: HashMap<SceneComplexity, String>,
    /// Baseline bitrate (bits/s) for the Medium tier; other tiers are scaled.
    base_bitrate: u64,
    /// Minimum scene duration (seconds) below which scenes inherit their
    /// neighbour's preset to avoid excessive switching.
    min_scene_duration: f64,
}

impl SceneAdaptivePreset {
    /// Create a new adaptive preset engine.
    ///
    /// `base_bitrate` is the target bitrate for Medium-complexity scenes.
    #[must_use]
    pub fn new(base_bitrate: u64) -> Self {
        Self {
            tier_presets: HashMap::new(),
            base_bitrate,
            min_scene_duration: 2.0,
        }
    }

    /// Set the minimum scene duration (seconds). Scenes shorter than this
    /// will inherit the previous scene's preset.
    #[must_use]
    pub fn with_min_scene_duration(mut self, duration: f64) -> Self {
        self.min_scene_duration = duration.max(0.0);
        self
    }

    /// Register a preset ID for a complexity tier.
    pub fn set_tier_preset(&mut self, complexity: SceneComplexity, preset_id: &str) {
        self.tier_presets.insert(complexity, preset_id.to_string());
    }

    /// Builder-style tier registration.
    #[must_use]
    pub fn with_tier(mut self, complexity: SceneComplexity, preset_id: &str) -> Self {
        self.set_tier_preset(complexity, preset_id);
        self
    }

    /// Get the preset ID assigned to a tier.
    #[must_use]
    pub fn get_tier_preset(&self, complexity: &SceneComplexity) -> Option<&str> {
        self.tier_presets.get(complexity).map(String::as_str)
    }

    /// Select presets for a sequence of analysed scenes.
    ///
    /// Returns one `AdaptivePresetSelection` per scene. Scenes shorter than
    /// `min_scene_duration` inherit the previous scene's preset to avoid
    /// rapid switching.
    ///
    /// If no preset is registered for a tier, the engine falls back to the
    /// nearest registered tier (preferring lower complexity).
    #[must_use]
    pub fn select(&self, analyses: &[SceneAnalysisResult]) -> Vec<AdaptivePresetSelection> {
        let mut selections = Vec::with_capacity(analyses.len());
        let mut prev_preset_id: Option<String> = None;

        for analysis in analyses {
            let duration = analysis.segment.duration();

            // If scene is too short, inherit previous preset
            let (preset_id, complexity) = if duration < self.min_scene_duration {
                if let Some(prev) = &prev_preset_id {
                    (prev.clone(), analysis.complexity)
                } else {
                    (
                        self.resolve_preset_for_tier(analysis.complexity),
                        analysis.complexity,
                    )
                }
            } else {
                (
                    self.resolve_preset_for_tier(analysis.complexity),
                    analysis.complexity,
                )
            };

            let suggested_bitrate =
                (self.base_bitrate as f64 * complexity.bitrate_multiplier()) as u64;

            prev_preset_id = Some(preset_id.clone());

            selections.push(AdaptivePresetSelection {
                scene_index: analysis.segment.index,
                start_time: analysis.segment.start_time,
                end_time: analysis.segment.end_time,
                preset_id,
                complexity,
                suggested_bitrate,
            });
        }

        selections
    }

    /// Resolve the best preset ID for a given tier, falling back to neighbours
    /// when the exact tier has no registered preset.
    fn resolve_preset_for_tier(&self, target: SceneComplexity) -> String {
        // Try exact match first
        if let Some(id) = self.tier_presets.get(&target) {
            return id.clone();
        }

        // Fallback order: search lower tiers first, then higher
        let all_tiers = [
            SceneComplexity::VeryLow,
            SceneComplexity::Low,
            SceneComplexity::Medium,
            SceneComplexity::High,
            SceneComplexity::VeryHigh,
        ];

        let target_idx = all_tiers.iter().position(|t| *t == target).unwrap_or(2);

        // Search outward from target index
        for offset in 1..=4 {
            // Try lower
            if target_idx >= offset {
                if let Some(id) = self.tier_presets.get(&all_tiers[target_idx - offset]) {
                    return id.clone();
                }
            }
            // Try higher
            if target_idx + offset < all_tiers.len() {
                if let Some(id) = self.tier_presets.get(&all_tiers[target_idx + offset]) {
                    return id.clone();
                }
            }
        }

        // Absolute fallback
        "default".to_string()
    }

    /// Compute summary statistics for a set of selections.
    #[must_use]
    pub fn summarize(selections: &[AdaptivePresetSelection]) -> AdaptiveSummary {
        let total_duration: f64 = selections
            .iter()
            .map(|s| (s.end_time - s.start_time).max(0.0))
            .sum();

        let mut tier_durations: HashMap<SceneComplexity, f64> = HashMap::new();
        let mut total_bitrate_weighted: f64 = 0.0;

        for sel in selections {
            let dur = (sel.end_time - sel.start_time).max(0.0);
            *tier_durations.entry(sel.complexity).or_default() += dur;
            total_bitrate_weighted += sel.suggested_bitrate as f64 * dur;
        }

        let average_bitrate = if total_duration > 0.0 {
            (total_bitrate_weighted / total_duration) as u64
        } else {
            0
        };

        AdaptiveSummary {
            scene_count: selections.len(),
            total_duration,
            average_bitrate,
            tier_durations,
        }
    }
}

/// Summary statistics for an adaptive encoding plan.
#[derive(Debug, Clone)]
pub struct AdaptiveSummary {
    /// Total number of scenes.
    pub scene_count: usize,
    /// Total duration in seconds.
    pub total_duration: f64,
    /// Weighted average bitrate across all scenes.
    pub average_bitrate: u64,
    /// Duration spent in each complexity tier.
    pub tier_durations: HashMap<SceneComplexity, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(index: usize, start: f64, end: f64) -> SceneSegment {
        SceneSegment::new(index, start, end)
    }

    #[test]
    fn test_scene_complexity_from_score() {
        assert_eq!(SceneComplexity::from_score(0.0), SceneComplexity::VeryLow);
        assert_eq!(SceneComplexity::from_score(0.1), SceneComplexity::VeryLow);
        assert_eq!(SceneComplexity::from_score(0.2), SceneComplexity::Low);
        assert_eq!(SceneComplexity::from_score(0.5), SceneComplexity::Medium);
        assert_eq!(SceneComplexity::from_score(0.7), SceneComplexity::High);
        assert_eq!(SceneComplexity::from_score(0.9), SceneComplexity::VeryHigh);
        assert_eq!(SceneComplexity::from_score(1.0), SceneComplexity::VeryHigh);
    }

    #[test]
    fn test_scene_complexity_from_score_clamps() {
        assert_eq!(SceneComplexity::from_score(-1.0), SceneComplexity::VeryLow);
        assert_eq!(SceneComplexity::from_score(2.0), SceneComplexity::VeryHigh);
    }

    #[test]
    fn test_scene_complexity_labels() {
        assert_eq!(SceneComplexity::VeryLow.label(), "Very Low");
        assert_eq!(SceneComplexity::Medium.label(), "Medium");
        assert_eq!(SceneComplexity::VeryHigh.label(), "Very High");
    }

    #[test]
    fn test_bitrate_multiplier_ordering() {
        assert!(
            SceneComplexity::VeryLow.bitrate_multiplier()
                < SceneComplexity::Low.bitrate_multiplier()
        );
        assert!(
            SceneComplexity::Low.bitrate_multiplier()
                < SceneComplexity::Medium.bitrate_multiplier()
        );
        assert!(
            SceneComplexity::Medium.bitrate_multiplier()
                < SceneComplexity::High.bitrate_multiplier()
        );
        assert!(
            SceneComplexity::High.bitrate_multiplier()
                < SceneComplexity::VeryHigh.bitrate_multiplier()
        );
    }

    #[test]
    fn test_scene_segment_duration() {
        let seg = make_segment(0, 1.0, 5.0);
        assert!((seg.duration() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scene_segment_negative_duration_clamped() {
        let seg = make_segment(0, 5.0, 3.0);
        assert!((seg.duration() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scene_analysis_weighted_score() {
        // motion=1.0, texture=0.0, temporal=0.0 => 0.45*1 + 0.35*0 + 0.20*0 = 0.45
        let analysis = SceneAnalysisResult::new(make_segment(0, 0.0, 5.0), 1.0, 0.0, 0.0);
        assert!((analysis.complexity_score - 0.45).abs() < 0.01);
        assert_eq!(analysis.complexity, SceneComplexity::Medium);
    }

    #[test]
    fn test_scene_analysis_all_max() {
        let analysis = SceneAnalysisResult::new(make_segment(0, 0.0, 5.0), 1.0, 1.0, 1.0);
        assert!((analysis.complexity_score - 1.0).abs() < 0.01);
        assert_eq!(analysis.complexity, SceneComplexity::VeryHigh);
    }

    #[test]
    fn test_scene_analysis_from_score() {
        let analysis = SceneAnalysisResult::from_score(make_segment(0, 0.0, 10.0), 0.3);
        assert_eq!(analysis.complexity, SceneComplexity::Low);
    }

    #[test]
    fn test_adaptive_select_basic() {
        let engine = SceneAdaptivePreset::new(5_000_000)
            .with_tier(SceneComplexity::Low, "preset-low")
            .with_tier(SceneComplexity::Medium, "preset-medium")
            .with_tier(SceneComplexity::High, "preset-high");

        let analyses = vec![
            SceneAnalysisResult::from_score(make_segment(0, 0.0, 10.0), 0.25),
            SceneAnalysisResult::from_score(make_segment(1, 10.0, 20.0), 0.5),
            SceneAnalysisResult::from_score(make_segment(2, 20.0, 30.0), 0.75),
        ];

        let selections = engine.select(&analyses);
        assert_eq!(selections.len(), 3);
        assert_eq!(selections[0].preset_id, "preset-low");
        assert_eq!(selections[1].preset_id, "preset-medium");
        assert_eq!(selections[2].preset_id, "preset-high");
    }

    #[test]
    fn test_adaptive_select_short_scene_inherits() {
        let engine = SceneAdaptivePreset::new(5_000_000)
            .with_min_scene_duration(3.0)
            .with_tier(SceneComplexity::Low, "preset-low")
            .with_tier(SceneComplexity::High, "preset-high");

        let analyses = vec![
            // First scene: 10s, low complexity
            SceneAnalysisResult::from_score(make_segment(0, 0.0, 10.0), 0.25),
            // Second scene: 1s (too short), high complexity — should inherit "preset-low"
            SceneAnalysisResult::from_score(make_segment(1, 10.0, 11.0), 0.75),
            // Third scene: 10s, high complexity
            SceneAnalysisResult::from_score(make_segment(2, 11.0, 21.0), 0.75),
        ];

        let selections = engine.select(&analyses);
        assert_eq!(selections[0].preset_id, "preset-low");
        assert_eq!(selections[1].preset_id, "preset-low"); // inherited
        assert_eq!(selections[2].preset_id, "preset-high");
    }

    #[test]
    fn test_adaptive_fallback_to_nearest_tier() {
        // Only register Medium tier
        let engine =
            SceneAdaptivePreset::new(5_000_000).with_tier(SceneComplexity::Medium, "preset-medium");

        let analyses = vec![
            SceneAnalysisResult::from_score(make_segment(0, 0.0, 10.0), 0.1), // VeryLow
            SceneAnalysisResult::from_score(make_segment(1, 10.0, 20.0), 0.9), // VeryHigh
        ];

        let selections = engine.select(&analyses);
        // Both should fallback to the only registered tier
        assert_eq!(selections[0].preset_id, "preset-medium");
        assert_eq!(selections[1].preset_id, "preset-medium");
    }

    #[test]
    fn test_adaptive_suggested_bitrate() {
        let engine = SceneAdaptivePreset::new(5_000_000)
            .with_tier(SceneComplexity::Low, "preset-low")
            .with_tier(SceneComplexity::High, "preset-high");

        let analyses = vec![
            SceneAnalysisResult::from_score(make_segment(0, 0.0, 10.0), 0.25), // Low
            SceneAnalysisResult::from_score(make_segment(1, 10.0, 20.0), 0.75), // High
        ];

        let selections = engine.select(&analyses);
        // Low: 5_000_000 * 0.65 = 3_250_000
        assert_eq!(selections[0].suggested_bitrate, 3_250_000);
        // High: 5_000_000 * 1.4 = 7_000_000
        assert_eq!(selections[1].suggested_bitrate, 7_000_000);
    }

    #[test]
    fn test_adaptive_summary() {
        let selections = vec![
            AdaptivePresetSelection {
                scene_index: 0,
                start_time: 0.0,
                end_time: 10.0,
                preset_id: "low".to_string(),
                complexity: SceneComplexity::Low,
                suggested_bitrate: 3_000_000,
            },
            AdaptivePresetSelection {
                scene_index: 1,
                start_time: 10.0,
                end_time: 20.0,
                preset_id: "high".to_string(),
                complexity: SceneComplexity::High,
                suggested_bitrate: 7_000_000,
            },
        ];

        let summary = SceneAdaptivePreset::summarize(&selections);
        assert_eq!(summary.scene_count, 2);
        assert!((summary.total_duration - 20.0).abs() < 0.01);
        // avg = (3M*10 + 7M*10)/20 = 5M
        assert_eq!(summary.average_bitrate, 5_000_000);
        assert!((summary.tier_durations[&SceneComplexity::Low] - 10.0).abs() < 0.01);
        assert!((summary.tier_durations[&SceneComplexity::High] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_empty_input() {
        let engine =
            SceneAdaptivePreset::new(5_000_000).with_tier(SceneComplexity::Medium, "preset-medium");
        let selections = engine.select(&[]);
        assert!(selections.is_empty());
    }

    #[test]
    fn test_adaptive_no_tiers_registered() {
        let engine = SceneAdaptivePreset::new(5_000_000);
        let analyses = vec![SceneAnalysisResult::from_score(
            make_segment(0, 0.0, 10.0),
            0.5,
        )];
        let selections = engine.select(&analyses);
        assert_eq!(selections[0].preset_id, "default");
    }

    #[test]
    fn test_get_tier_preset() {
        let engine =
            SceneAdaptivePreset::new(5_000_000).with_tier(SceneComplexity::Medium, "mid-preset");
        assert_eq!(
            engine.get_tier_preset(&SceneComplexity::Medium),
            Some("mid-preset")
        );
        assert_eq!(engine.get_tier_preset(&SceneComplexity::High), None);
    }

    #[test]
    fn test_summary_empty() {
        let summary = SceneAdaptivePreset::summarize(&[]);
        assert_eq!(summary.scene_count, 0);
        assert_eq!(summary.average_bitrate, 0);
        assert!((summary.total_duration - 0.0).abs() < f64::EPSILON);
    }
}
