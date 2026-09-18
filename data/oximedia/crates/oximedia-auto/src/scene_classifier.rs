//! Scene classification for automated video analysis.
//!
//! Provides `SceneClass`, `ScoreThreshold`, `SceneClassification`, and
//! `SceneClassifier` for labelling video scenes by their visual content.

#![allow(dead_code)]

/// High-level scene categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneClass {
    /// Outdoor daylight scene.
    OutdoorDay,
    /// Outdoor night scene.
    OutdoorNight,
    /// Indoor scene.
    Indoor,
    /// Close-up face shot.
    FaceCloseUp,
    /// Crowd or group shot.
    Crowd,
    /// Sports action.
    Sports,
    /// Animated or graphic content.
    Animation,
    /// Text / title card.
    TitleCard,
    /// Black frame / cut.
    BlackFrame,
    /// None of the above.
    Unknown,
}

impl SceneClass {
    /// Return a human-readable label for this class.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::OutdoorDay => "outdoor-day",
            Self::OutdoorNight => "outdoor-night",
            Self::Indoor => "indoor",
            Self::FaceCloseUp => "face-close-up",
            Self::Crowd => "crowd",
            Self::Sports => "sports",
            Self::Animation => "animation",
            Self::TitleCard => "title-card",
            Self::BlackFrame => "black-frame",
            Self::Unknown => "unknown",
        }
    }

    /// Return `true` if the scene typically contains faces.
    #[must_use]
    pub const fn contains_faces(self) -> bool {
        matches!(self, Self::FaceCloseUp | Self::Crowd)
    }

    /// Return `true` if the scene is an outdoor scene.
    #[must_use]
    pub const fn is_outdoor(self) -> bool {
        matches!(self, Self::OutdoorDay | Self::OutdoorNight)
    }
}

/// A minimum score threshold used to decide whether a classification passes.
#[derive(Debug, Clone, Copy)]
pub struct ScoreThreshold {
    /// The threshold value in 0.0–1.0.
    pub value: f64,
}

impl ScoreThreshold {
    /// Create a new threshold.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
        }
    }

    /// Return `true` if `score` meets or exceeds the threshold.
    #[must_use]
    pub fn passes(&self, score: f64) -> bool {
        score >= self.value
    }
}

impl Default for ScoreThreshold {
    fn default() -> Self {
        Self::new(0.5)
    }
}

/// Per-class score for a single frame or segment.
#[derive(Debug, Clone)]
pub struct ClassScore {
    /// Scene class.
    pub class: SceneClass,
    /// Confidence score in 0.0–1.0.
    pub score: f64,
}

/// The full classification result for one scene.
#[derive(Debug, Clone)]
pub struct SceneClassification {
    /// All per-class scores, sorted descending by score.
    pub scores: Vec<ClassScore>,
    /// Frame index or scene identifier.
    pub scene_id: u64,
}

impl SceneClassification {
    /// Create a new `SceneClassification`, sorting `scores` descending.
    #[must_use]
    pub fn new(scene_id: u64, mut scores: Vec<ClassScore>) -> Self {
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { scores, scene_id }
    }

    /// Return the class with the highest score.
    #[must_use]
    pub fn top_class(&self) -> SceneClass {
        self.scores
            .first()
            .map_or(SceneClass::Unknown, |cs| cs.class)
    }

    /// Return the score for the top class.
    #[must_use]
    pub fn top_score(&self) -> f64 {
        self.scores.first().map_or(0.0, |cs| cs.score)
    }

    /// Return all classes whose score passes `threshold`.
    #[must_use]
    pub fn classes_above_threshold(&self, threshold: &ScoreThreshold) -> Vec<SceneClass> {
        self.scores
            .iter()
            .filter(|cs| threshold.passes(cs.score))
            .map(|cs| cs.class)
            .collect()
    }
}

/// Classifies scenes using a simple score-table approach.
///
/// In production this would delegate to a neural network; here it provides a
/// deterministic stub suitable for testing pipeline logic.
#[derive(Debug, Clone)]
pub struct SceneClassifier {
    /// Minimum score to report a class.
    threshold: ScoreThreshold,
}

impl SceneClassifier {
    /// Create a new `SceneClassifier` with the given threshold.
    #[must_use]
    pub fn new(threshold: ScoreThreshold) -> Self {
        Self { threshold }
    }

    /// Classify a single scene described by `feature_vec`.
    ///
    /// `feature_vec` is a slice of `f64` values in `[0.0, 1.0]` where each
    /// index corresponds to a `SceneClass` variant (by declaration order).
    /// Missing or extra entries are handled gracefully.
    #[must_use]
    pub fn classify(&self, scene_id: u64, feature_vec: &[f64]) -> SceneClassification {
        let classes = [
            SceneClass::OutdoorDay,
            SceneClass::OutdoorNight,
            SceneClass::Indoor,
            SceneClass::FaceCloseUp,
            SceneClass::Crowd,
            SceneClass::Sports,
            SceneClass::Animation,
            SceneClass::TitleCard,
            SceneClass::BlackFrame,
            SceneClass::Unknown,
        ];
        let scores: Vec<ClassScore> = classes
            .iter()
            .enumerate()
            .map(|(i, &class)| ClassScore {
                class,
                score: feature_vec.get(i).copied().unwrap_or(0.0),
            })
            .filter(|cs| self.threshold.passes(cs.score))
            .collect();
        SceneClassification::new(scene_id, scores)
    }

    /// Classify multiple scenes, returning one `SceneClassification` per entry.
    #[must_use]
    pub fn batch_classify(&self, scenes: &[(u64, Vec<f64>)]) -> Vec<SceneClassification> {
        scenes
            .iter()
            .map(|(id, feat)| self.classify(*id, feat))
            .collect()
    }
}

impl Default for SceneClassifier {
    fn default() -> Self {
        Self::new(ScoreThreshold::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_class_label_nonempty() {
        for class in [
            SceneClass::OutdoorDay,
            SceneClass::OutdoorNight,
            SceneClass::Indoor,
            SceneClass::FaceCloseUp,
            SceneClass::Crowd,
            SceneClass::Sports,
            SceneClass::Animation,
            SceneClass::TitleCard,
            SceneClass::BlackFrame,
            SceneClass::Unknown,
        ] {
            assert!(
                !class.label().is_empty(),
                "label should not be empty for {class:?}"
            );
        }
    }

    #[test]
    fn test_scene_class_contains_faces() {
        assert!(SceneClass::FaceCloseUp.contains_faces());
        assert!(SceneClass::Crowd.contains_faces());
        assert!(!SceneClass::Indoor.contains_faces());
    }

    #[test]
    fn test_scene_class_is_outdoor() {
        assert!(SceneClass::OutdoorDay.is_outdoor());
        assert!(SceneClass::OutdoorNight.is_outdoor());
        assert!(!SceneClass::Indoor.is_outdoor());
    }

    #[test]
    fn test_score_threshold_passes() {
        let t = ScoreThreshold::new(0.6);
        assert!(t.passes(0.6));
        assert!(t.passes(0.9));
        assert!(!t.passes(0.5));
    }

    #[test]
    fn test_score_threshold_clamps_above_one() {
        let t = ScoreThreshold::new(1.5);
        assert!((t.value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_score_threshold_clamps_below_zero() {
        let t = ScoreThreshold::new(-0.5);
        assert!((t.value - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_classification_top_class() {
        let scores = vec![
            ClassScore {
                class: SceneClass::Indoor,
                score: 0.4,
            },
            ClassScore {
                class: SceneClass::Sports,
                score: 0.9,
            },
            ClassScore {
                class: SceneClass::Crowd,
                score: 0.6,
            },
        ];
        let clf = SceneClassification::new(1, scores);
        assert_eq!(clf.top_class(), SceneClass::Sports);
    }

    #[test]
    fn test_classification_top_score() {
        let scores = vec![ClassScore {
            class: SceneClass::Animation,
            score: 0.75,
        }];
        let clf = SceneClassification::new(2, scores);
        assert!((clf.top_score() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_classification_empty_top_class() {
        let clf = SceneClassification::new(3, vec![]);
        assert_eq!(clf.top_class(), SceneClass::Unknown);
        assert_eq!(clf.top_score(), 0.0);
    }

    #[test]
    fn test_classification_classes_above_threshold() {
        let scores = vec![
            ClassScore {
                class: SceneClass::OutdoorDay,
                score: 0.8,
            },
            ClassScore {
                class: SceneClass::Indoor,
                score: 0.3,
            },
            ClassScore {
                class: SceneClass::FaceCloseUp,
                score: 0.7,
            },
        ];
        let clf = SceneClassification::new(4, scores);
        let t = ScoreThreshold::new(0.5);
        let above = clf.classes_above_threshold(&t);
        assert_eq!(above.len(), 2);
        assert!(above.contains(&SceneClass::OutdoorDay));
        assert!(above.contains(&SceneClass::FaceCloseUp));
    }

    #[test]
    fn test_classifier_classify_basic() {
        let clf = SceneClassifier::default();
        // Index 0 → OutdoorDay, score 0.9
        let mut feat = vec![0.0f64; 10];
        feat[0] = 0.9;
        let result = clf.classify(10, &feat);
        assert_eq!(result.top_class(), SceneClass::OutdoorDay);
        assert!((result.top_score() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_classifier_classify_filters_below_threshold() {
        let clf = SceneClassifier::new(ScoreThreshold::new(0.8));
        let feat = vec![0.5, 0.3, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = clf.classify(11, &feat);
        // Only Indoor (index 2, score 0.9) passes threshold 0.8
        assert_eq!(result.scores.len(), 1);
        assert_eq!(result.top_class(), SceneClass::Indoor);
    }

    #[test]
    fn test_classifier_batch_classify() {
        let clf = SceneClassifier::default();
        let mut f1 = vec![0.0f64; 10];
        f1[1] = 0.85; // OutdoorNight
        let mut f2 = vec![0.0f64; 10];
        f2[5] = 0.95; // Sports
        let scenes = vec![(1u64, f1), (2u64, f2)];
        let results = clf.batch_classify(&scenes);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].top_class(), SceneClass::OutdoorNight);
        assert_eq!(results[1].top_class(), SceneClass::Sports);
    }
}
