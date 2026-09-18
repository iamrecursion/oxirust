//! Scene transition analysis.
//!
//! This module provides tools for detecting and classifying transitions between
//! scenes in video content (cuts, dissolves, fades, wipes, etc.).

/// The type of transition between two scenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// A hard cut: instantaneous change between scenes.
    Cut,
    /// A dissolve: one scene fades into another.
    Dissolve,
    /// A fade: scene fades to or from a solid color.
    Fade,
    /// A wipe: one scene replaces another with a moving boundary.
    Wipe,
    /// A cross-fade: audio/video crossfade between scenes.
    CrossFade,
    /// A dip: scene briefly goes to black/white before next scene.
    Dip,
    /// Unknown or unclassified transition.
    Unknown,
}

impl TransitionType {
    /// Return a human-readable label for the transition type.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Cut => "cut",
            Self::Dissolve => "dissolve",
            Self::Fade => "fade",
            Self::Wipe => "wipe",
            Self::CrossFade => "cross-fade",
            Self::Dip => "dip",
            Self::Unknown => "unknown",
        }
    }

    /// Return whether this transition is gradual (takes multiple frames).
    #[must_use]
    pub fn is_gradual(&self) -> bool {
        matches!(
            self,
            Self::Dissolve | Self::Fade | Self::CrossFade | Self::Dip
        )
    }
}

/// A detected transition between two scenes.
#[derive(Debug, Clone)]
pub struct SceneTransition {
    /// Index of the scene before the transition.
    pub from_scene: u64,
    /// Index of the scene after the transition.
    pub to_scene: u64,
    /// Classification of the transition type.
    pub transition_type: TransitionType,
    /// First frame of the transition.
    pub start_frame: u64,
    /// Last frame of the transition (inclusive).
    pub end_frame: u64,
    /// Confidence of the classification (0.0–1.0).
    pub confidence: f64,
}

impl SceneTransition {
    /// Create a new `SceneTransition`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(from: u64, to: u64, ttype: TransitionType, start: u64, end: u64, conf: f64) -> Self {
        Self {
            from_scene: from,
            to_scene: to,
            transition_type: ttype,
            start_frame: start,
            end_frame: end,
            confidence: conf.clamp(0.0, 1.0),
        }
    }

    /// Return the duration of the transition in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Return whether this transition is a hard cut (zero duration).
    #[must_use]
    pub fn is_hard_cut(&self) -> bool {
        matches!(self.transition_type, TransitionType::Cut) || self.duration_frames() == 0
    }
}

/// Analyze a slice of per-frame difference values to classify the transition type.
///
/// `frames_diff` should contain normalised difference scores (0.0–1.0) between
/// consecutive frames spanning the transition region.
#[must_use]
pub fn detect_transition_type(frames_diff: &[f64]) -> TransitionType {
    if frames_diff.is_empty() {
        return TransitionType::Unknown;
    }

    let n = frames_diff.len();

    // A single spike → hard cut
    if n == 1 {
        return if frames_diff[0] > 0.5 {
            TransitionType::Cut
        } else {
            TransitionType::Unknown
        };
    }

    let max_diff = frames_diff
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_diff = frames_diff.iter().copied().fold(f64::INFINITY, f64::min);
    let mean_diff: f64 = frames_diff.iter().sum::<f64>() / n as f64;

    // Hard cut: single dominant spike, short duration
    if max_diff > 0.8 && (max_diff - min_diff) > 0.5 && n <= 3 {
        return TransitionType::Cut;
    }

    // Dip: goes low (near black) then recovers
    let mid_idx = n / 2;
    let mid_val = frames_diff[mid_idx];
    let ends_mean = (frames_diff[0] + frames_diff[n - 1]) / 2.0;
    if mid_val < 0.15 && ends_mean > 0.4 {
        return TransitionType::Dip;
    }

    // Fade: monotonically decreasing or increasing differences with low variance
    let is_monotone_up = frames_diff.windows(2).all(|w| w[1] >= w[0] - 0.05);
    let is_monotone_down = frames_diff.windows(2).all(|w| w[1] <= w[0] + 0.05);
    let variance: f64 = frames_diff
        .iter()
        .map(|&x| (x - mean_diff).powi(2))
        .sum::<f64>()
        / n as f64;

    if (is_monotone_up || is_monotone_down) && variance < 0.02 {
        return TransitionType::Fade;
    }

    // Dissolve: sustained mid-range differences
    if mean_diff > 0.2 && mean_diff < 0.6 && variance < 0.04 {
        return TransitionType::Dissolve;
    }

    // Wipe: sharp spatial boundary that moves; approximated by an abrupt step
    if (max_diff - min_diff) > 0.4 && n > 3 {
        return TransitionType::Wipe;
    }

    // Cross-fade: gradual, sustained with slight bell curve
    if n > 5 && mean_diff > 0.1 {
        return TransitionType::CrossFade;
    }

    TransitionType::Unknown
}

/// Accumulates detected transitions and computes aggregate statistics.
#[derive(Debug, Default)]
pub struct TransitionAnalyzer {
    transitions: Vec<SceneTransition>,
}

impl TransitionAnalyzer {
    /// Create a new, empty `TransitionAnalyzer`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a detected transition.
    pub fn add(&mut self, t: SceneTransition) {
        self.transitions.push(t);
    }

    /// Return the most common transition type across all stored transitions.
    #[must_use]
    pub fn most_common(&self) -> Option<TransitionType> {
        if self.transitions.is_empty() {
            return None;
        }

        let types = [
            TransitionType::Cut,
            TransitionType::Dissolve,
            TransitionType::Fade,
            TransitionType::Wipe,
            TransitionType::CrossFade,
            TransitionType::Dip,
            TransitionType::Unknown,
        ];

        types
            .iter()
            .max_by_key(|&&t| {
                self.transitions
                    .iter()
                    .filter(|tr| tr.transition_type == t)
                    .count()
            })
            .copied()
    }

    /// Return the average transition duration in frames.
    #[must_use]
    pub fn avg_duration(&self) -> f64 {
        if self.transitions.is_empty() {
            return 0.0;
        }
        let total: u64 = self
            .transitions
            .iter()
            .map(SceneTransition::duration_frames)
            .sum();
        total as f64 / self.transitions.len() as f64
    }

    /// Return the ratio of hard cuts to total transitions (0.0–1.0).
    #[must_use]
    pub fn cut_ratio(&self) -> f64 {
        if self.transitions.is_empty() {
            return 0.0;
        }
        let cuts = self.transitions.iter().filter(|t| t.is_hard_cut()).count();
        cuts as f64 / self.transitions.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_type_label() {
        assert_eq!(TransitionType::Cut.label(), "cut");
        assert_eq!(TransitionType::Dissolve.label(), "dissolve");
        assert_eq!(TransitionType::Fade.label(), "fade");
        assert_eq!(TransitionType::Wipe.label(), "wipe");
        assert_eq!(TransitionType::CrossFade.label(), "cross-fade");
        assert_eq!(TransitionType::Dip.label(), "dip");
        assert_eq!(TransitionType::Unknown.label(), "unknown");
    }

    #[test]
    fn test_transition_type_is_gradual() {
        assert!(!TransitionType::Cut.is_gradual());
        assert!(TransitionType::Dissolve.is_gradual());
        assert!(TransitionType::Fade.is_gradual());
        assert!(!TransitionType::Wipe.is_gradual());
        assert!(TransitionType::CrossFade.is_gradual());
        assert!(TransitionType::Dip.is_gradual());
        assert!(!TransitionType::Unknown.is_gradual());
    }

    #[test]
    fn test_scene_transition_new() {
        let t = SceneTransition::new(0, 1, TransitionType::Cut, 100, 100, 0.95);
        assert_eq!(t.from_scene, 0);
        assert_eq!(t.to_scene, 1);
        assert_eq!(t.transition_type, TransitionType::Cut);
        assert_eq!(t.start_frame, 100);
        assert_eq!(t.end_frame, 100);
        assert!((t.confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scene_transition_confidence_clamping() {
        let t = SceneTransition::new(0, 1, TransitionType::Cut, 0, 0, 1.5);
        assert!((t.confidence - 1.0).abs() < f64::EPSILON);
        let t2 = SceneTransition::new(0, 1, TransitionType::Cut, 0, 0, -0.5);
        assert!((t2.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_duration_frames() {
        let t = SceneTransition::new(0, 1, TransitionType::Dissolve, 100, 115, 0.8);
        assert_eq!(t.duration_frames(), 15);
    }

    #[test]
    fn test_is_hard_cut_true() {
        let t = SceneTransition::new(0, 1, TransitionType::Cut, 50, 50, 0.9);
        assert!(t.is_hard_cut());
    }

    #[test]
    fn test_is_hard_cut_false_for_dissolve() {
        let t = SceneTransition::new(0, 1, TransitionType::Dissolve, 50, 60, 0.9);
        assert!(!t.is_hard_cut());
    }

    #[test]
    fn test_detect_transition_type_cut() {
        let diffs = vec![0.9];
        assert_eq!(detect_transition_type(&diffs), TransitionType::Cut);
    }

    #[test]
    fn test_detect_transition_type_empty() {
        assert_eq!(detect_transition_type(&[]), TransitionType::Unknown);
    }

    #[test]
    fn test_detect_transition_type_fade() {
        // Monotonically increasing with low variance (tight ramp → variance < 0.02)
        let diffs = vec![0.30, 0.31, 0.32, 0.33, 0.34, 0.35];
        assert_eq!(detect_transition_type(&diffs), TransitionType::Fade);
    }

    #[test]
    fn test_detect_transition_type_dissolve() {
        // Non-monotone oscillation in mid range (not caught by fade branch)
        // mean=0.40, variance≈0.0012 (<0.04), not monotone → Dissolve
        let diffs = vec![0.40, 0.35, 0.45, 0.38, 0.42];
        assert_eq!(detect_transition_type(&diffs), TransitionType::Dissolve);
    }

    #[test]
    fn test_analyzer_most_common_empty() {
        let analyzer = TransitionAnalyzer::new();
        assert!(analyzer.most_common().is_none());
    }

    #[test]
    fn test_analyzer_most_common() {
        let mut analyzer = TransitionAnalyzer::new();
        analyzer.add(SceneTransition::new(0, 1, TransitionType::Cut, 10, 10, 0.9));
        analyzer.add(SceneTransition::new(1, 2, TransitionType::Cut, 20, 20, 0.9));
        analyzer.add(SceneTransition::new(
            2,
            3,
            TransitionType::Dissolve,
            30,
            45,
            0.8,
        ));
        assert_eq!(analyzer.most_common(), Some(TransitionType::Cut));
    }

    #[test]
    fn test_analyzer_avg_duration() {
        let mut analyzer = TransitionAnalyzer::new();
        analyzer.add(SceneTransition::new(0, 1, TransitionType::Cut, 10, 10, 0.9));
        analyzer.add(SceneTransition::new(
            1,
            2,
            TransitionType::Dissolve,
            20,
            30,
            0.8,
        ));
        // durations: 0 and 10 → avg = 5
        assert!((analyzer.avg_duration() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_cut_ratio() {
        let mut analyzer = TransitionAnalyzer::new();
        analyzer.add(SceneTransition::new(0, 1, TransitionType::Cut, 10, 10, 0.9));
        analyzer.add(SceneTransition::new(
            1,
            2,
            TransitionType::Dissolve,
            20,
            30,
            0.8,
        ));
        assert!((analyzer.cut_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_cut_ratio_empty() {
        let analyzer = TransitionAnalyzer::new();
        assert!((analyzer.cut_ratio() - 0.0).abs() < f64::EPSILON);
    }
}
