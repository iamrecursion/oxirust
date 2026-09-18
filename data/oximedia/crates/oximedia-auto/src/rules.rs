//! Rules engine for automated video editing.
//!
//! This module provides a configurable rules system for controlling
//! automated editing behavior:
//!
//! - **Cut rules**: Minimum/maximum shot duration constraints
//! - **Transition rules**: When and how to use transitions
//! - **Music sync**: Synchronize edits with audio
//! - **Aspect ratio**: Adapt content for different formats
//! - **Pacing**: Control editing rhythm and timing
//!
//! # Example
//!
//! ```
//! use oximedia_auto::rules::{RulesEngine, EditRules};
//!
//! let rules = EditRules::default();
//! let engine = RulesEngine::new(rules);
//! ```

use crate::cuts::{CutPoint, CutType};
use crate::error::{AutoError, AutoResult};
use oximedia_core::{Rational, Timestamp};
use std::collections::HashMap;

/// Aspect ratio for video output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AspectRatio {
    /// 16:9 landscape (standard widescreen).
    Landscape16x9,
    /// 4:3 standard (legacy TV).
    Standard4x3,
    /// 1:1 square (Instagram).
    Square1x1,
    /// 9:16 vertical (`TikTok`, Reels).
    Vertical9x16,
    /// 4:5 portrait (Instagram feed).
    Portrait4x5,
    /// 21:9 ultrawide (cinema).
    Cinema21x9,
    /// Custom ratio.
    Custom(u32, u32),
}

impl AspectRatio {
    /// Get the aspect ratio as a floating point value.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Landscape16x9 => 16.0 / 9.0,
            Self::Standard4x3 => 4.0 / 3.0,
            Self::Square1x1 => 1.0,
            Self::Vertical9x16 => 9.0 / 16.0,
            Self::Portrait4x5 => 4.0 / 5.0,
            Self::Cinema21x9 => 21.0 / 9.0,
            Self::Custom(w, h) => *w as f64 / *h as f64,
        }
    }

    /// Get width and height for a given resolution height.
    #[must_use]
    pub fn dimensions_for_height(&self, height: u32) -> (u32, u32) {
        let width = (f64::from(height) * self.as_f64()).round() as u32;
        (width, height)
    }

    /// Check if this is a portrait orientation.
    #[must_use]
    pub fn is_portrait(&self) -> bool {
        self.as_f64() < 1.0
    }

    /// Check if this is a landscape orientation.
    #[must_use]
    pub fn is_landscape(&self) -> bool {
        self.as_f64() > 1.0
    }
}

/// Pacing preset for video editing rhythm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingPreset {
    /// Very fast cuts (action, sports).
    VeryFast,
    /// Fast cuts (music videos, trailers).
    Fast,
    /// Medium pacing (balanced).
    Medium,
    /// Slow cuts (documentaries).
    Slow,
    /// Very slow (contemplative, art).
    VerySlow,
    /// Custom pacing.
    Custom,
}

impl PacingPreset {
    /// Get the average shot duration for this preset in milliseconds.
    #[must_use]
    pub const fn average_shot_duration_ms(&self) -> i64 {
        match self {
            Self::VeryFast => 1000,  // 1 second
            Self::Fast => 2000,      // 2 seconds
            Self::Medium => 4000,    // 4 seconds
            Self::Slow => 6000,      // 6 seconds
            Self::VerySlow => 10000, // 10 seconds
            Self::Custom => 4000,
        }
    }

    /// Get the minimum shot duration for this preset in milliseconds.
    #[must_use]
    pub const fn min_shot_duration_ms(&self) -> i64 {
        match self {
            Self::VeryFast => 500,
            Self::Fast => 1000,
            Self::Medium => 1500,
            Self::Slow => 2000,
            Self::VerySlow => 3000,
            Self::Custom => 1000,
        }
    }

    /// Get the maximum shot duration for this preset in milliseconds.
    #[must_use]
    pub const fn max_shot_duration_ms(&self) -> i64 {
        match self {
            Self::VeryFast => 3000,
            Self::Fast => 5000,
            Self::Medium => 8000,
            Self::Slow => 12000,
            Self::VerySlow => 20000,
            Self::Custom => 10000,
        }
    }
}

/// Music synchronization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusicSyncMode {
    /// No music synchronization.
    None,
    /// Sync cuts to beats.
    Beats,
    /// Sync to musical bars.
    Bars,
    /// Sync to phrases.
    Phrases,
    /// Adaptive sync based on music intensity.
    Adaptive,
}

/// Transition preference settings.
#[derive(Debug, Clone)]
pub struct TransitionPreferences {
    /// Default transition type.
    pub default_type: CutType,
    /// Allow hard cuts.
    pub allow_hard_cuts: bool,
    /// Allow dissolves.
    pub allow_dissolves: bool,
    /// Allow fades.
    pub allow_fades: bool,
    /// Allow wipes.
    pub allow_wipes: bool,
    /// Prefer gradual transitions over hard cuts.
    pub prefer_gradual: bool,
    /// Default transition duration in milliseconds.
    pub default_duration_ms: i64,
    /// Minimum transition duration in milliseconds.
    pub min_duration_ms: i64,
    /// Maximum transition duration in milliseconds.
    pub max_duration_ms: i64,
}

impl Default for TransitionPreferences {
    fn default() -> Self {
        Self {
            default_type: CutType::Hard,
            allow_hard_cuts: true,
            allow_dissolves: true,
            allow_fades: true,
            allow_wipes: false,
            prefer_gradual: false,
            default_duration_ms: 500,
            min_duration_ms: 200,
            max_duration_ms: 2000,
        }
    }
}

impl TransitionPreferences {
    /// Check if a transition type is allowed.
    #[must_use]
    pub const fn is_allowed(&self, cut_type: CutType) -> bool {
        match cut_type {
            CutType::Hard | CutType::Soft => self.allow_hard_cuts,
            CutType::Dissolve => self.allow_dissolves,
            CutType::Fade => self.allow_fades,
            CutType::Wipe => self.allow_wipes,
            CutType::Jump => false, // Jump cuts should be avoided
            CutType::LCut | CutType::JCut => true, // Split edits always allowed
        }
    }
}

/// Shot duration constraints.
#[derive(Debug, Clone, Copy)]
pub struct ShotConstraints {
    /// Minimum shot duration in milliseconds.
    pub min_duration_ms: i64,
    /// Maximum shot duration in milliseconds.
    pub max_duration_ms: i64,
    /// Target average duration in milliseconds.
    pub target_average_ms: i64,
    /// Allow variance from target (0.0 to 1.0).
    pub variance_tolerance: f64,
}

impl Default for ShotConstraints {
    fn default() -> Self {
        Self {
            min_duration_ms: 1000,
            max_duration_ms: 8000,
            target_average_ms: 4000,
            variance_tolerance: 0.3,
        }
    }
}

impl ShotConstraints {
    /// Create constraints from a pacing preset.
    #[must_use]
    pub fn from_preset(preset: PacingPreset) -> Self {
        Self {
            min_duration_ms: preset.min_shot_duration_ms(),
            max_duration_ms: preset.max_shot_duration_ms(),
            target_average_ms: preset.average_shot_duration_ms(),
            variance_tolerance: 0.3,
        }
    }

    /// Check if a duration is within acceptable range.
    #[must_use]
    pub fn is_acceptable(&self, duration_ms: i64) -> bool {
        duration_ms >= self.min_duration_ms && duration_ms <= self.max_duration_ms
    }

    /// Check if a duration is close to the target.
    #[must_use]
    pub fn is_near_target(&self, duration_ms: i64) -> bool {
        let diff = (duration_ms - self.target_average_ms).abs() as f64;
        let tolerance = self.target_average_ms as f64 * self.variance_tolerance;
        diff <= tolerance
    }
}

/// Complete set of editing rules.
#[derive(Debug, Clone)]
pub struct EditRules {
    /// Shot duration constraints.
    pub shot_constraints: ShotConstraints,
    /// Transition preferences.
    pub transition_prefs: TransitionPreferences,
    /// Target aspect ratio.
    pub target_aspect_ratio: AspectRatio,
    /// Music synchronization mode.
    pub music_sync: MusicSyncMode,
    /// Pacing preset.
    pub pacing: PacingPreset,
    /// Enforce minimum shot duration strictly.
    pub strict_min_duration: bool,
    /// Enforce maximum shot duration strictly.
    pub strict_max_duration: bool,
    /// Allow split edits (L-cuts, J-cuts).
    pub allow_split_edits: bool,
    /// Prefer cuts on action.
    pub prefer_action_cuts: bool,
    /// Avoid cutting during dialogue.
    pub avoid_dialogue_cuts: bool,
    /// Custom rules (name -> value).
    pub custom_rules: HashMap<String, String>,
}

impl Default for EditRules {
    fn default() -> Self {
        Self {
            shot_constraints: ShotConstraints::default(),
            transition_prefs: TransitionPreferences::default(),
            target_aspect_ratio: AspectRatio::Landscape16x9,
            music_sync: MusicSyncMode::Beats,
            pacing: PacingPreset::Medium,
            strict_min_duration: true,
            strict_max_duration: false,
            allow_split_edits: true,
            prefer_action_cuts: true,
            avoid_dialogue_cuts: true,
            custom_rules: HashMap::new(),
        }
    }
}

impl EditRules {
    /// Create new editing rules.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create rules for a specific use case.
    #[must_use]
    pub fn for_use_case(use_case: &str) -> Self {
        match use_case.to_lowercase().as_str() {
            "trailer" => Self::trailer_preset(),
            "highlights" => Self::highlights_preset(),
            "social" => Self::social_media_preset(),
            "documentary" => Self::documentary_preset(),
            "music_video" => Self::music_video_preset(),
            _ => Self::default(),
        }
    }

    /// Preset for trailers (fast paced, dynamic).
    #[must_use]
    pub fn trailer_preset() -> Self {
        Self {
            shot_constraints: ShotConstraints::from_preset(PacingPreset::Fast),
            pacing: PacingPreset::Fast,
            music_sync: MusicSyncMode::Beats,
            prefer_action_cuts: true,
            transition_prefs: TransitionPreferences {
                prefer_gradual: true,
                allow_wipes: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Preset for highlight reels (very fast, action-focused).
    #[must_use]
    pub fn highlights_preset() -> Self {
        Self {
            shot_constraints: ShotConstraints::from_preset(PacingPreset::VeryFast),
            pacing: PacingPreset::VeryFast,
            music_sync: MusicSyncMode::Beats,
            prefer_action_cuts: true,
            strict_max_duration: true,
            ..Default::default()
        }
    }

    /// Preset for social media (short, attention-grabbing).
    #[must_use]
    pub fn social_media_preset() -> Self {
        Self {
            shot_constraints: ShotConstraints::from_preset(PacingPreset::Fast),
            pacing: PacingPreset::Fast,
            target_aspect_ratio: AspectRatio::Vertical9x16,
            music_sync: MusicSyncMode::Beats,
            prefer_action_cuts: true,
            strict_max_duration: true,
            ..Default::default()
        }
    }

    /// Preset for documentaries (slower, contemplative).
    #[must_use]
    pub fn documentary_preset() -> Self {
        Self {
            shot_constraints: ShotConstraints::from_preset(PacingPreset::Slow),
            pacing: PacingPreset::Slow,
            music_sync: MusicSyncMode::None,
            prefer_action_cuts: false,
            avoid_dialogue_cuts: true,
            transition_prefs: TransitionPreferences {
                prefer_gradual: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Preset for music videos (synchronized to music).
    #[must_use]
    pub fn music_video_preset() -> Self {
        Self {
            shot_constraints: ShotConstraints::from_preset(PacingPreset::Medium),
            pacing: PacingPreset::Medium,
            music_sync: MusicSyncMode::Bars,
            prefer_action_cuts: true,
            avoid_dialogue_cuts: false,
            transition_prefs: TransitionPreferences {
                allow_wipes: true,
                prefer_gradual: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Set the pacing preset.
    #[must_use]
    pub fn with_pacing(mut self, pacing: PacingPreset) -> Self {
        self.pacing = pacing;
        self.shot_constraints = ShotConstraints::from_preset(pacing);
        self
    }

    /// Set the aspect ratio.
    #[must_use]
    pub const fn with_aspect_ratio(mut self, ratio: AspectRatio) -> Self {
        self.target_aspect_ratio = ratio;
        self
    }

    /// Set the music sync mode.
    #[must_use]
    pub const fn with_music_sync(mut self, sync: MusicSyncMode) -> Self {
        self.music_sync = sync;
        self
    }

    /// Add a custom rule.
    #[must_use]
    pub fn with_custom_rule<S: Into<String>>(mut self, name: S, value: S) -> Self {
        self.custom_rules.insert(name.into(), value.into());
        self
    }

    /// Validate the rules configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.shot_constraints.min_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.shot_constraints.min_duration_ms,
            });
        }

        if self.shot_constraints.max_duration_ms <= self.shot_constraints.min_duration_ms {
            return Err(AutoError::invalid_parameter(
                "max_duration",
                "must be greater than min_duration",
            ));
        }

        if self.transition_prefs.min_duration_ms < 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.transition_prefs.min_duration_ms,
            });
        }

        if self.transition_prefs.max_duration_ms <= self.transition_prefs.min_duration_ms {
            return Err(AutoError::invalid_parameter(
                "max_transition_duration",
                "must be greater than min_transition_duration",
            ));
        }

        if !(0.0..=1.0).contains(&self.shot_constraints.variance_tolerance) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.shot_constraints.variance_tolerance,
                min: 0.0,
                max: 1.0,
            });
        }

        Ok(())
    }
}

/// Rules engine for applying editing rules.
pub struct RulesEngine {
    /// Editing rules.
    rules: EditRules,
}

impl RulesEngine {
    /// Create a new rules engine.
    #[must_use]
    pub fn new(rules: EditRules) -> Self {
        Self { rules }
    }

    /// Create a rules engine with default rules.
    #[must_use]
    pub fn default_engine() -> Self {
        Self::new(EditRules::default())
    }

    /// Apply rules to a set of cut points.
    ///
    /// # Errors
    ///
    /// Returns an error if rules validation fails.
    pub fn apply_rules(&self, cuts: &mut Vec<CutPoint>) -> AutoResult<()> {
        self.rules.validate()?;

        // Filter invalid transitions
        cuts.retain(|cut| self.rules.transition_prefs.is_allowed(cut.cut_type));

        // Enforce shot duration constraints
        self.enforce_shot_durations(cuts)?;

        // Apply transition preferences
        self.apply_transition_preferences(cuts);

        // Sort by timestamp
        cuts.sort_by_key(|c| c.timestamp.pts);

        Ok(())
    }

    /// Enforce shot duration constraints by adding or removing cuts.
    fn enforce_shot_durations(&self, cuts: &mut Vec<CutPoint>) -> AutoResult<()> {
        if cuts.is_empty() {
            return Ok(());
        }

        let mut i = 0;
        while i < cuts.len().saturating_sub(1) {
            let duration = cuts[i + 1].timestamp.pts - cuts[i].timestamp.pts;

            // Check minimum duration
            if self.rules.strict_min_duration
                && duration < self.rules.shot_constraints.min_duration_ms
            {
                // Remove the cut with lower priority
                if cuts[i].priority < cuts[i + 1].priority {
                    cuts.remove(i);
                } else {
                    cuts.remove(i + 1);
                }
                continue;
            }

            // Check maximum duration
            if self.rules.strict_max_duration
                && duration > self.rules.shot_constraints.max_duration_ms
            {
                // Insert a new cut point in the middle
                let mid_time =
                    cuts[i].timestamp.pts + self.rules.shot_constraints.target_average_ms;
                let mid_cut = CutPoint::new(
                    Timestamp::new(mid_time, cuts[i].timestamp.timebase),
                    self.rules.transition_prefs.default_type,
                    0.7,
                )
                .with_reason("Inserted to enforce max shot duration");
                cuts.insert(i + 1, mid_cut);
            }

            i += 1;
        }

        Ok(())
    }

    /// Apply transition preferences to cut points.
    fn apply_transition_preferences(&self, cuts: &mut [CutPoint]) {
        for cut in cuts {
            // Apply default transition type if current is not allowed
            if !self.rules.transition_prefs.is_allowed(cut.cut_type) {
                cut.cut_type = self.rules.transition_prefs.default_type;
            }

            // Enforce transition duration constraints
            if cut.cut_type.is_gradual() {
                cut.transition_duration_ms = cut.transition_duration_ms.clamp(
                    self.rules.transition_prefs.min_duration_ms,
                    self.rules.transition_prefs.max_duration_ms,
                );
            }

            // Prefer gradual transitions if configured
            if self.rules.transition_prefs.prefer_gradual && cut.cut_type.is_hard() {
                if self.rules.transition_prefs.allow_dissolves {
                    cut.cut_type = CutType::Dissolve;
                    cut.transition_duration_ms = self.rules.transition_prefs.default_duration_ms;
                } else if self.rules.transition_prefs.allow_fades {
                    cut.cut_type = CutType::Fade;
                    cut.transition_duration_ms = self.rules.transition_prefs.default_duration_ms;
                }
            }
        }
    }

    /// Calculate target duration for a clip based on rules.
    #[must_use]
    pub fn calculate_target_duration(&self, total_duration_ms: i64, num_clips: usize) -> i64 {
        if num_clips == 0 {
            return 0;
        }

        let average_per_clip = total_duration_ms / num_clips as i64;
        average_per_clip.clamp(
            self.rules.shot_constraints.min_duration_ms,
            self.rules.shot_constraints.max_duration_ms,
        )
    }

    /// Get aspect ratio adaptation parameters.
    #[must_use]
    pub fn get_aspect_ratio_params(&self) -> (AspectRatio, bool) {
        (
            self.rules.target_aspect_ratio,
            self.rules.target_aspect_ratio.is_portrait(),
        )
    }

    /// Check if music synchronization is enabled.
    #[must_use]
    pub const fn should_sync_to_music(&self) -> bool {
        !matches!(self.rules.music_sync, MusicSyncMode::None)
    }

    /// Get the music sync mode.
    #[must_use]
    pub const fn music_sync_mode(&self) -> MusicSyncMode {
        self.rules.music_sync
    }

    /// Get the current rules.
    #[must_use]
    pub const fn rules(&self) -> &EditRules {
        &self.rules
    }

    /// Update rules.
    pub fn set_rules(&mut self, rules: EditRules) {
        self.rules = rules;
    }
}

impl Default for RulesEngine {
    fn default() -> Self {
        Self::default_engine()
    }
}

/// Calculate optimal pacing based on content analysis.
#[allow(dead_code)]
pub fn suggest_pacing(
    avg_motion_intensity: f64,
    has_music: bool,
    content_type: &str,
) -> PacingPreset {
    match content_type.to_lowercase().as_str() {
        "action" | "sports" => PacingPreset::VeryFast,
        "music" if has_music => PacingPreset::Fast,
        "documentary" => PacingPreset::Slow,
        "interview" => PacingPreset::Slow,
        _ => {
            // Use motion intensity to determine pacing
            if avg_motion_intensity > 0.7 {
                PacingPreset::Fast
            } else if avg_motion_intensity > 0.4 {
                PacingPreset::Medium
            } else {
                PacingPreset::Slow
            }
        }
    }
}

/// Adapt cuts for different aspect ratios by adjusting framing.
#[allow(dead_code)]
pub fn adapt_for_aspect_ratio(
    _cuts: &mut [CutPoint],
    source_ratio: AspectRatio,
    target_ratio: AspectRatio,
) -> AutoResult<()> {
    // If converting from landscape to portrait, may need to adjust cut points
    // to ensure important content stays in frame
    if source_ratio.is_landscape() && target_ratio.is_portrait() {
        // Additional logic would go here for smart cropping
    }

    Ok(())
}

/// Synchronize cuts to a specific BPM.
#[allow(dead_code)]
pub fn sync_to_bpm(cuts: &mut [CutPoint], bpm: f64, timebase: Rational) -> AutoResult<()> {
    if bpm <= 0.0 {
        return Err(AutoError::invalid_parameter("bpm", "must be positive"));
    }

    let beat_duration_ms = (60_000.0 / bpm) as i64;

    for cut in cuts {
        // Snap to nearest beat
        let beats_from_start = cut.timestamp.pts / beat_duration_ms;
        let snapped_time = beats_from_start * beat_duration_ms;

        cut.timestamp = Timestamp::new(snapped_time, timebase);
        cut.on_beat = true;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn make_ts(pts: i64) -> Timestamp {
        Timestamp::new(pts, Rational::new(1, 1000))
    }

    fn make_cut(pts: i64) -> CutPoint {
        CutPoint::new(make_ts(pts), CutType::Hard, 0.9)
    }

    /// Cuts spaced closer than `preset.min_shot_duration_ms()` must be
    /// pruned by `apply_rules` so all resulting gaps are >= the minimum.
    #[test]
    fn test_enforce_min_shot_duration() {
        // VeryFast preset: min = 500 ms.
        let rules = EditRules::new()
            .with_pacing(PacingPreset::VeryFast)
            .with_custom_rule("test", "true");
        let engine = RulesEngine::new(rules);

        let min_ms = PacingPreset::VeryFast.min_shot_duration_ms();

        // Create cuts that are 100 ms apart (way below the 500 ms minimum).
        let mut cuts: Vec<CutPoint> = (0..10).map(|i| make_cut(i * 100)).collect();

        engine
            .apply_rules(&mut cuts)
            .expect("apply_rules should succeed");

        // All consecutive pairs must be >= min_ms apart.
        for window in cuts.windows(2) {
            let gap = window[1].timestamp.pts - window[0].timestamp.pts;
            assert!(
                gap >= min_ms,
                "gap {gap} ms is less than min_shot_duration {min_ms} ms"
            );
        }
    }

    /// Cuts that already satisfy all constraints should pass through unchanged.
    #[test]
    fn test_rules_engine_no_op_on_valid_cuts() {
        // Medium preset: min = 1500 ms, max = 8000 ms.
        let rules = EditRules::new().with_pacing(PacingPreset::Medium);
        let engine = RulesEngine::new(rules);

        // Cuts spaced 2000 ms apart — comfortably within [1500, 8000].
        let original_pts: Vec<i64> = (0..5).map(|i| i * 2000).collect();
        let mut cuts: Vec<CutPoint> = original_pts.iter().copied().map(make_cut).collect();

        engine
            .apply_rules(&mut cuts)
            .expect("apply_rules should succeed");

        // Cuts must be preserved (in timestamp order — apply_rules sorts).
        assert_eq!(cuts.len(), original_pts.len(), "no cuts should be removed");
        for (cut, &pts) in cuts.iter().zip(original_pts.iter()) {
            assert_eq!(cut.timestamp.pts, pts, "timestamps must be unchanged");
        }
    }

    /// `calculate_target_duration` clamps to the configured min/max bounds.
    #[test]
    fn test_calculate_target_duration_clamped() {
        let rules = EditRules::new().with_pacing(PacingPreset::Fast);
        let engine = RulesEngine::new(rules);

        let min = PacingPreset::Fast.min_shot_duration_ms();
        let max = PacingPreset::Fast.max_shot_duration_ms();

        // Tiny total: result should be clamped to min.
        let tiny = engine.calculate_target_duration(10, 100);
        assert_eq!(tiny, min);

        // Giant total with 1 clip: result should be clamped to max.
        let huge = engine.calculate_target_duration(1_000_000, 1);
        assert_eq!(huge, max);
    }
}
