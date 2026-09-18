//! Automated pacing and edit decision module.
//!
//! Provides tools for analyzing clip durations and suggesting edits to achieve
//! a target pacing style (action, drama, documentary, etc.).

#![allow(dead_code)]

/// Target pacing parameters for a given editing style.
#[derive(Debug, Clone)]
pub struct PacingTarget {
    /// Average desired shot duration in frames.
    pub avg_shot_duration_frames: f64,
    /// Shots shorter than this (frames) are considered fast cuts.
    pub fast_cut_threshold: u64,
    /// Shots longer than this (frames) are considered slow burns.
    pub slow_burn_threshold: u64,
}

impl PacingTarget {
    /// Action pacing: short, punchy cuts.
    #[must_use]
    pub fn action() -> Self {
        Self {
            avg_shot_duration_frames: 24.0,
            fast_cut_threshold: 12,
            slow_burn_threshold: 48,
        }
    }

    /// Drama pacing: medium shots with room to breathe.
    #[must_use]
    pub fn drama() -> Self {
        Self {
            avg_shot_duration_frames: 72.0,
            fast_cut_threshold: 36,
            slow_burn_threshold: 120,
        }
    }

    /// Documentary pacing: long, observational shots.
    #[must_use]
    pub fn documentary() -> Self {
        Self {
            avg_shot_duration_frames: 150.0,
            fast_cut_threshold: 60,
            slow_burn_threshold: 240,
        }
    }
}

/// A single edit suggestion produced by pacing analysis.
#[derive(Debug, Clone)]
pub struct EditSuggestion {
    /// Frame index at which the cut should occur.
    pub cut_at_frame: u64,
    /// Human-readable reason for the suggestion.
    pub reason: String,
    /// Confidence level in [0.0, 1.0].
    pub confidence: f64,
}

impl EditSuggestion {
    /// Create a new edit suggestion.
    #[must_use]
    pub fn new(cut_at_frame: u64, reason: impl Into<String>, confidence: f64) -> Self {
        Self {
            cut_at_frame,
            reason: reason.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Suggest cut points based on how clip durations deviate from the target pacing.
///
/// Returns a list of [`EditSuggestion`]s, one per clip that deviates significantly.
#[must_use]
pub fn suggest_cuts_from_pacing(
    durations: &[u64],
    fps: f64,
    target: &PacingTarget,
) -> Vec<EditSuggestion> {
    let mut suggestions = Vec::new();
    let mut frame_cursor: u64 = 0;

    for &dur in durations {
        if dur > target.slow_burn_threshold {
            // Shot is too long: suggest a cut in the middle.
            let cut_at = frame_cursor + dur / 2;
            let excess = dur.saturating_sub(target.slow_burn_threshold);
            let confidence = (excess as f64 / target.avg_shot_duration_frames).min(1.0);
            suggestions.push(EditSuggestion::new(
                cut_at,
                format!(
                    "Shot of {dur} frames exceeds slow-burn threshold of {} frames ({:.1}s excess)",
                    target.slow_burn_threshold,
                    excess as f64 / fps
                ),
                confidence,
            ));
        } else if dur < target.fast_cut_threshold {
            // Shot is too short: suggest merging by marking the start frame.
            let confidence = (target.fast_cut_threshold.saturating_sub(dur) as f64
                / target.avg_shot_duration_frames)
                .min(1.0);
            suggestions.push(EditSuggestion::new(
                frame_cursor,
                format!(
                    "Shot of {dur} frames is below fast-cut threshold of {} frames",
                    target.fast_cut_threshold
                ),
                confidence,
            ));
        }
        frame_cursor += dur;
    }

    suggestions
}

/// Compute a speed-factor adjustment per clip so the average duration approaches `target_avg`.
///
/// Returns `(frame_index, speed_factor)` pairs.  A speed factor > 1.0 means speed up;
/// < 1.0 means slow down.
#[must_use]
pub fn auto_pacing_adjustment(durations: &[u64], target_avg: f64) -> Vec<(u64, f64)> {
    if durations.is_empty() || target_avg <= 0.0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(durations.len());
    let mut frame_cursor: u64 = 0;

    for &dur in durations {
        let speed_factor = if dur == 0 {
            1.0
        } else {
            dur as f64 / target_avg
        };
        result.push((frame_cursor, speed_factor));
        frame_cursor += dur;
    }

    result
}

/// High-level editor that wraps pacing analysis and accumulates suggestions.
#[derive(Debug, Clone)]
pub struct PacingEditor {
    /// The pacing target configuration.
    pub target: PacingTarget,
    /// Accumulated edit suggestions from the most recent analysis.
    pub suggestions: Vec<EditSuggestion>,
}

impl PacingEditor {
    /// Create a new `PacingEditor` with the given target.
    #[must_use]
    pub fn new(target: PacingTarget) -> Self {
        Self {
            target,
            suggestions: Vec::new(),
        }
    }

    /// Analyse clip durations at the given frame rate and populate `self.suggestions`.
    pub fn analyze(&mut self, durations: &[u64], fps: f64) {
        self.suggestions = suggest_cuts_from_pacing(durations, fps, &self.target);
    }

    /// Apply high-confidence suggestions to a mutable list of clip durations.
    ///
    /// Clips that are too long are split at the midpoint; clips that are too short
    /// are merged with the next clip (if one exists).
    pub fn apply_suggestions(&self, clips: &mut Vec<u64>) {
        // Process in reverse order so index arithmetic stays valid.
        let mut i = 0usize;
        while i < clips.len() {
            let dur = clips[i];
            if dur > self.target.slow_burn_threshold {
                let half = dur / 2;
                let remainder = dur - half;
                clips[i] = half;
                clips.insert(i + 1, remainder);
                i += 2; // skip over both halves
            } else if dur < self.target.fast_cut_threshold && i + 1 < clips.len() {
                let next = clips[i + 1];
                clips[i] = dur + next;
                clips.remove(i + 1);
                // re-examine the merged clip
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_pacing_target() {
        let t = PacingTarget::action();
        assert!(t.avg_shot_duration_frames < 50.0);
        assert!(t.fast_cut_threshold < t.slow_burn_threshold);
    }

    #[test]
    fn test_drama_pacing_target() {
        let t = PacingTarget::drama();
        assert!(t.avg_shot_duration_frames > 50.0);
        assert!(t.fast_cut_threshold < t.slow_burn_threshold);
    }

    #[test]
    fn test_documentary_pacing_target() {
        let t = PacingTarget::documentary();
        assert!(t.avg_shot_duration_frames > 100.0);
        assert!(t.fast_cut_threshold < t.slow_burn_threshold);
    }

    #[test]
    fn test_suggest_cuts_too_long() {
        let target = PacingTarget::action();
        let durations = vec![100u64]; // well above slow_burn_threshold of 48
        let suggestions = suggest_cuts_from_pacing(&durations, 24.0, &target);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].cut_at_frame, 50); // midpoint
    }

    #[test]
    fn test_suggest_cuts_too_short() {
        let target = PacingTarget::action();
        let durations = vec![5u64]; // below fast_cut_threshold of 12
        let suggestions = suggest_cuts_from_pacing(&durations, 24.0, &target);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].cut_at_frame, 0);
    }

    #[test]
    fn test_suggest_cuts_no_suggestion_within_range() {
        let target = PacingTarget::action();
        let durations = vec![24u64]; // exactly avg, within thresholds
        let suggestions = suggest_cuts_from_pacing(&durations, 24.0, &target);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggest_cuts_multiple_clips() {
        let target = PacingTarget::drama();
        let durations = vec![200u64, 72, 10]; // long, ok, short
        let suggestions = suggest_cuts_from_pacing(&durations, 24.0, &target);
        // Should flag the 200-frame clip (too long) and the 10-frame clip (too short)
        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn test_auto_pacing_adjustment_empty() {
        let result = auto_pacing_adjustment(&[], 30.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_auto_pacing_adjustment_speed_up() {
        let durations = vec![60u64, 60, 60];
        let adjustments = auto_pacing_adjustment(&durations, 30.0);
        assert_eq!(adjustments.len(), 3);
        // Each clip is twice target_avg, so speed factor should be 2.0
        for (_, factor) in &adjustments {
            assert!((factor - 2.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_auto_pacing_adjustment_frame_cursor() {
        let durations = vec![10u64, 20, 30];
        let adjustments = auto_pacing_adjustment(&durations, 10.0);
        assert_eq!(adjustments[0].0, 0);
        assert_eq!(adjustments[1].0, 10);
        assert_eq!(adjustments[2].0, 30);
    }

    #[test]
    fn test_pacing_editor_new() {
        let editor = PacingEditor::new(PacingTarget::action());
        assert!(editor.suggestions.is_empty());
    }

    #[test]
    fn test_pacing_editor_analyze() {
        let mut editor = PacingEditor::new(PacingTarget::action());
        editor.analyze(&[100, 24, 5], 24.0);
        // 100 is too long, 5 is too short -> 2 suggestions
        assert_eq!(editor.suggestions.len(), 2);
    }

    #[test]
    fn test_pacing_editor_apply_splits_long_clip() {
        let editor = PacingEditor::new(PacingTarget::action());
        let mut clips = vec![100u64]; // above slow_burn_threshold of 48
        editor.apply_suggestions(&mut clips);
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0] + clips[1], 100);
    }

    #[test]
    fn test_pacing_editor_apply_merges_short_clip() {
        let editor = PacingEditor::new(PacingTarget::action());
        let mut clips = vec![5u64, 20]; // 5 below fast_cut_threshold
        editor.apply_suggestions(&mut clips);
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0], 25);
    }

    #[test]
    fn test_edit_suggestion_confidence_clamp() {
        let s = EditSuggestion::new(0, "test", 2.5);
        assert!((s.confidence - 1.0).abs() < f64::EPSILON);
        let s2 = EditSuggestion::new(0, "test", -0.5);
        assert!((s2.confidence).abs() < f64::EPSILON);
    }
}
