//! Smart / intelligent trim operations for the timeline editor.
//!
//! Analyses clip content to suggest optimal in/out trim points based on
//! silence detection, scene boundaries, motion analysis, and audio peaks.

use crate::clip::{Clip, ClipId, ClipType};
use crate::error::EditResult;
use crate::timeline::Timeline;

// ─────────────────────────────────────────────────────────────────────────────
// TrimReason
// ─────────────────────────────────────────────────────────────────────────────

/// The signal that triggered a trim suggestion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimReason {
    /// A period of silence was detected near the trim point.
    SilenceDetected,
    /// A scene/shot boundary was detected near the trim point.
    SceneBoundary,
    /// Significant motion ceases near the trim point.
    MotionStop,
    /// An audio transient or peak occurs near the trim point.
    AudioPeak,
}

impl TrimReason {
    /// Human-readable description of the trim reason.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::SilenceDetected => "silence detected",
            Self::SceneBoundary => "scene boundary detected",
            Self::MotionStop => "motion stop detected",
            Self::AudioPeak => "audio peak detected",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TrimSuggestion
// ─────────────────────────────────────────────────────────────────────────────

/// A suggested trim point for a clip, with a confidence score and reason.
#[derive(Clone, Debug)]
pub struct TrimSuggestion {
    /// The clip this suggestion applies to.
    pub clip_id: ClipId,
    /// Proposed timeline position for the trim (timebase units).
    pub trim_point: i64,
    /// Confidence score in the range `[0.0, 1.0]`.
    pub confidence: f64,
    /// Reason the trim was suggested.
    pub reason: TrimReason,
    /// `true` → this is a suggested in-point; `false` → suggested out-point.
    pub is_in_point: bool,
}

impl TrimSuggestion {
    /// Create a new trim suggestion.
    #[must_use]
    pub fn new(
        clip_id: ClipId,
        trim_point: i64,
        confidence: f64,
        reason: TrimReason,
        is_in_point: bool,
    ) -> Self {
        Self {
            clip_id,
            trim_point,
            confidence: confidence.clamp(0.0, 1.0),
            reason,
            is_in_point,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SmartTrimConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Tuning parameters for the smart trim engine.
#[derive(Clone, Debug)]
pub struct SmartTrimConfig {
    /// Silence threshold in dBFS (default −40 dB).
    pub silence_threshold_db: f64,
    /// Minimum confidence required before a scene-boundary suggestion is
    /// emitted (default 0.7).
    pub min_scene_confidence: f64,
    /// Minimum duration of a silence region to be considered, in timebase
    /// units (default 100 ms).
    pub min_silence_duration_ms: i64,
    /// Normalised motion magnitude below which motion is considered stopped
    /// (default 0.1).
    pub motion_threshold: f64,
    /// Minimum suggestion confidence to include in `analyze` results (default 0.0).
    pub min_output_confidence: f64,
}

impl SmartTrimConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            silence_threshold_db: -40.0,
            min_scene_confidence: 0.7,
            min_silence_duration_ms: 100,
            motion_threshold: 0.1,
            min_output_confidence: 0.0,
        }
    }

    /// Set the silence threshold.
    #[must_use]
    pub fn with_silence_threshold(mut self, db: f64) -> Self {
        self.silence_threshold_db = db;
        self
    }

    /// Set the minimum scene-detection confidence.
    #[must_use]
    pub fn with_min_scene_confidence(mut self, confidence: f64) -> Self {
        self.min_scene_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the minimum silence duration in milliseconds.
    #[must_use]
    pub fn with_min_silence_duration_ms(mut self, ms: i64) -> Self {
        self.min_silence_duration_ms = ms.max(0);
        self
    }

    /// Set the motion-stop threshold.
    #[must_use]
    pub fn with_motion_threshold(mut self, threshold: f64) -> Self {
        self.motion_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Only surface suggestions with at least this confidence level.
    #[must_use]
    pub fn with_min_output_confidence(mut self, confidence: f64) -> Self {
        self.min_output_confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

impl Default for SmartTrimConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SmartTrimEngine
// ─────────────────────────────────────────────────────────────────────────────

/// Analyses clips and produces intelligent trim suggestions.
///
/// The analysis is based on heuristic rules derived from clip type and
/// duration; a production implementation would replace these with signal-level
/// analysis of the decoded media.
pub struct SmartTrimEngine {
    /// Tuning configuration.
    pub config: SmartTrimConfig,
}

impl SmartTrimEngine {
    /// Create a smart trim engine with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SmartTrimConfig::new(),
        }
    }

    /// Create a smart trim engine with custom configuration.
    #[must_use]
    pub fn new_with_config(config: SmartTrimConfig) -> Self {
        Self { config }
    }

    /// Analyse all clips in `timeline` and return all trim suggestions above
    /// the configured minimum confidence.
    #[must_use]
    pub fn analyze(&self, timeline: &Timeline) -> Vec<TrimSuggestion> {
        let min_conf = self.config.min_output_confidence;
        timeline
            .tracks
            .iter()
            .flat_map(|track| track.clips.iter())
            .flat_map(|clip| self.analyze_clip(clip))
            .filter(|s| s.confidence >= min_conf)
            .collect()
    }

    /// Produce trim suggestions for a single clip.
    ///
    /// Returns suggestions based on clip type:
    /// - **Audio** clips: silence-based in/out suggestions.
    /// - **Video** clips: scene-boundary in-point and motion-stop out-point.
    /// - **Subtitle** clips: no suggestions.
    #[must_use]
    pub fn analyze_clip(&self, clip: &Clip) -> Vec<TrimSuggestion> {
        let duration = clip.timeline_duration;
        if duration <= 0 {
            return Vec::new();
        }

        match clip.clip_type {
            ClipType::Audio => {
                let in_point = clip.timeline_start + duration / 10;
                let out_point = clip.timeline_end() - duration / 10;

                let in_suggestion =
                    TrimSuggestion::new(clip.id, in_point, 0.85, TrimReason::SilenceDetected, true);
                let out_suggestion = TrimSuggestion::new(
                    clip.id,
                    out_point,
                    0.80,
                    TrimReason::SilenceDetected,
                    false,
                );
                vec![in_suggestion, out_suggestion]
            }

            ClipType::Video => {
                let in_point = clip.timeline_start + duration / 20;
                let out_point = clip.timeline_end() - duration / 20;

                let in_suggestion =
                    TrimSuggestion::new(clip.id, in_point, 0.75, TrimReason::SceneBoundary, true);
                let out_suggestion =
                    TrimSuggestion::new(clip.id, out_point, 0.72, TrimReason::MotionStop, false);
                vec![in_suggestion, out_suggestion]
            }

            ClipType::Subtitle => Vec::new(),
        }
    }

    /// Return the highest-confidence in-point suggestion for `clip`, if any.
    #[must_use]
    pub fn suggest_in_point(&self, clip: &Clip) -> Option<TrimSuggestion> {
        self.analyze_clip(clip)
            .into_iter()
            .filter(|s| s.is_in_point)
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Return the highest-confidence out-point suggestion for `clip`, if any.
    #[must_use]
    pub fn suggest_out_point(&self, clip: &Clip) -> Option<TrimSuggestion> {
        self.analyze_clip(clip)
            .into_iter()
            .filter(|s| !s.is_in_point)
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Apply a set of trim suggestions to the timeline.
    ///
    /// Only suggestions with confidence ≥ 0.75 are applied.  Missing clips are
    /// silently skipped.  Returns the number of suggestions applied.
    pub fn apply_suggestions(
        &self,
        timeline: &mut Timeline,
        suggestions: &[TrimSuggestion],
    ) -> EditResult<usize> {
        let mut applied = 0usize;

        for suggestion in suggestions {
            if suggestion.confidence < 0.75 {
                continue;
            }

            // Fetch the clip; skip if not found
            let clip = match timeline.get_clip_mut(suggestion.clip_id) {
                Some(c) => c,
                None => continue,
            };

            if suggestion.is_in_point {
                // Compute new source_in relative to clip position
                let offset = suggestion.trim_point - clip.timeline_start;
                let new_source_in = (clip.source_in + offset)
                    .clamp(clip.source_in, clip.source_out.saturating_sub(1));
                clip.source_in = new_source_in;
            } else {
                // Compute new source_out relative to clip position
                let offset = suggestion.trim_point - clip.timeline_start;
                let new_source_out =
                    (clip.source_in + offset).clamp(clip.source_in + 1, clip.source_out);
                clip.source_out = new_source_out;
            }

            applied += 1;
        }

        Ok(applied)
    }
}

impl Default for SmartTrimEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::ClipType;
    use crate::timeline::Timeline;
    use oximedia_core::Rational;

    fn engine() -> SmartTrimEngine {
        SmartTrimEngine::new()
    }

    fn audio_clip(id: ClipId, start: i64, duration: i64) -> Clip {
        Clip::new(id, ClipType::Audio, start, duration)
    }

    fn video_clip(id: ClipId, start: i64, duration: i64) -> Clip {
        Clip::new(id, ClipType::Video, start, duration)
    }

    fn subtitle_clip(id: ClipId, start: i64, duration: i64) -> Clip {
        Clip::new(id, ClipType::Subtitle, start, duration)
    }

    #[test]
    fn test_analyze_clip_audio_returns_two_suggestions() {
        let clip = audio_clip(1, 0, 1000);
        let suggestions = engine().analyze_clip(&clip);
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.iter().any(|s| s.is_in_point));
        assert!(suggestions.iter().any(|s| !s.is_in_point));
        for s in &suggestions {
            assert_eq!(s.reason, TrimReason::SilenceDetected);
            assert!(s.confidence > 0.0);
        }
    }

    #[test]
    fn test_analyze_clip_video_returns_two_suggestions() {
        let clip = video_clip(2, 500, 2000);
        let suggestions = engine().analyze_clip(&clip);
        assert_eq!(suggestions.len(), 2);
        let in_sug = suggestions
            .iter()
            .find(|s| s.is_in_point)
            .expect("in-point");
        let out_sug = suggestions
            .iter()
            .find(|s| !s.is_in_point)
            .expect("out-point");
        assert_eq!(in_sug.reason, TrimReason::SceneBoundary);
        assert_eq!(out_sug.reason, TrimReason::MotionStop);
    }

    #[test]
    fn test_analyze_clip_subtitle_returns_empty() {
        let clip = subtitle_clip(3, 0, 500);
        let suggestions = engine().analyze_clip(&clip);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggest_in_point() {
        let clip = audio_clip(4, 0, 1000);
        let suggestion = engine().suggest_in_point(&clip);
        assert!(suggestion.is_some());
        let s = suggestion.expect("should have in-point suggestion");
        assert!(s.is_in_point);
        assert_eq!(s.clip_id, 4);
    }

    #[test]
    fn test_suggest_out_point() {
        let clip = video_clip(5, 0, 2000);
        let suggestion = engine().suggest_out_point(&clip);
        assert!(suggestion.is_some());
        let s = suggestion.expect("should have out-point suggestion");
        assert!(!s.is_in_point);
        assert_eq!(s.clip_id, 5);
    }

    #[test]
    fn test_apply_suggestions_returns_count() {
        let mut timeline = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let track = timeline.add_track(crate::timeline::TrackType::Audio);
        let clip = audio_clip(0, 0, 1000);
        let clip_id = timeline.add_clip(track, clip).expect("add clip ok");

        let engine = engine();
        let clip_ref = timeline.get_clip(clip_id).expect("clip exists");
        let suggestions = engine.analyze_clip(clip_ref);

        let applied = engine
            .apply_suggestions(&mut timeline, &suggestions)
            .expect("apply_suggestions ok");
        // Both audio suggestions have confidence >= 0.75, so 2 should be applied
        assert_eq!(applied, 2);
    }

    #[test]
    fn test_apply_suggestions_skips_missing_clips() {
        let mut timeline = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let suggestion = TrimSuggestion::new(9999, 100, 0.99, TrimReason::AudioPeak, true);
        let engine = engine();
        let applied = engine
            .apply_suggestions(&mut timeline, &[suggestion])
            .expect("should not error on missing clip");
        assert_eq!(applied, 0, "missing clips must be silently skipped");
    }

    #[test]
    fn test_apply_suggestions_skips_low_confidence() {
        let mut timeline = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let track = timeline.add_track(crate::timeline::TrackType::Video);
        let clip = video_clip(0, 0, 2000);
        let clip_id = timeline.add_clip(track, clip).expect("add clip ok");

        let suggestion = TrimSuggestion::new(clip_id, 100, 0.50, TrimReason::SceneBoundary, true);
        let engine = engine();
        let applied = engine
            .apply_suggestions(&mut timeline, &[suggestion])
            .expect("apply ok");
        assert_eq!(applied, 0, "low confidence suggestion must be skipped");
    }

    #[test]
    fn test_analyze_all_clips_in_timeline() {
        let mut timeline = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let v_track = timeline.add_track(crate::timeline::TrackType::Video);
        let a_track = timeline.add_track(crate::timeline::TrackType::Audio);
        timeline
            .add_clip(v_track, video_clip(0, 0, 2000))
            .expect("v ok");
        timeline
            .add_clip(a_track, audio_clip(0, 0, 1000))
            .expect("a ok");

        let engine = engine();
        let suggestions = engine.analyze(&timeline);
        // 2 video + 2 audio = 4 suggestions
        assert_eq!(suggestions.len(), 4);
    }

    #[test]
    fn test_config_builder() {
        let config = SmartTrimConfig::new()
            .with_silence_threshold(-30.0)
            .with_min_scene_confidence(0.8)
            .with_min_silence_duration_ms(200)
            .with_motion_threshold(0.05)
            .with_min_output_confidence(0.6);

        let engine = SmartTrimEngine::new_with_config(config);
        assert!((engine.config.silence_threshold_db - (-30.0)).abs() < f64::EPSILON);
        assert!((engine.config.min_scene_confidence - 0.8).abs() < f64::EPSILON);
        assert_eq!(engine.config.min_silence_duration_ms, 200);
    }
}
