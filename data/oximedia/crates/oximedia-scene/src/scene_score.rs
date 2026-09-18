#![allow(dead_code)]
//! Scene quality scoring: per-category scores and aggregate scene score.
//!
//! `SceneScore` aggregates multiple `CategoryScore` values into a single
//! weighted overall score, enabling ranking and filtering of video scenes
//! by quality, relevance, or aesthetic merit.

// ─── ScoreCategory ────────────────────────────────────────────────────────────

/// The aspect of a scene that a score describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScoreCategory {
    /// Overall visual quality (sharpness, noise, exposure).
    VisualQuality,
    /// Aesthetic appeal (composition, colour harmony, rule of thirds).
    Aesthetic,
    /// Degree of motion activity in the scene.
    Motion,
    /// Prominence and size of detected faces in the frame.
    FacePresence,
    /// Relevance of the scene to a user-supplied search query.
    Relevance,
    /// Spoken audio quality (clarity, noise level).
    AudioClarity,
    /// Overall emotional impact or engagement of the scene.
    Engagement,
}

impl ScoreCategory {
    /// Human-readable label for this category.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::VisualQuality => "visual_quality",
            Self::Aesthetic => "aesthetic",
            Self::Motion => "motion",
            Self::FacePresence => "face_presence",
            Self::Relevance => "relevance",
            Self::AudioClarity => "audio_clarity",
            Self::Engagement => "engagement",
        }
    }

    /// Returns `true` for categories that relate to video (not audio) content.
    #[must_use]
    pub fn is_video_category(self) -> bool {
        !matches!(self, Self::AudioClarity)
    }
}

// ─── CategoryScore ────────────────────────────────────────────────────────────

/// A score in [0.0, 1.0] for a particular [`ScoreCategory`], together with an
/// associated weight used when computing an aggregate.
#[derive(Debug, Clone, Copy)]
pub struct CategoryScore {
    /// The category this score belongs to.
    pub category: ScoreCategory,
    /// Normalised score value in [0.0, 1.0].
    pub score: f32,
    /// Non-negative weight for this category in the aggregate calculation.
    pub weight: f32,
}

impl CategoryScore {
    /// Create a new `CategoryScore`.
    ///
    /// `score` and `weight` are clamped to [0.0, 1.0] and [0.0, ∞) respectively.
    #[must_use]
    pub fn new(category: ScoreCategory, score: f32, weight: f32) -> Self {
        Self {
            category,
            score: score.clamp(0.0, 1.0),
            weight: weight.max(0.0),
        }
    }

    /// Create a `CategoryScore` with a default weight of 1.0.
    #[must_use]
    pub fn unit(category: ScoreCategory, score: f32) -> Self {
        Self::new(category, score, 1.0)
    }

    /// Returns `true` if this score exceeds `threshold`.
    #[must_use]
    pub fn passes(&self, threshold: f32) -> bool {
        self.score >= threshold
    }

    /// Weighted contribution of this score to an aggregate: `score * weight`.
    #[must_use]
    pub fn weighted_value(self) -> f32 {
        self.score * self.weight
    }
}

// ─── SceneScore ───────────────────────────────────────────────────────────────

/// A collection of per-category scores that together describe the quality
/// or relevance of a single scene.
#[derive(Debug, Clone, Default)]
pub struct SceneScore {
    scores: Vec<CategoryScore>,
}

impl SceneScore {
    /// Create an empty `SceneScore`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a [`CategoryScore`] to this scene score.
    pub fn add(&mut self, score: CategoryScore) {
        self.scores.push(score);
    }

    /// Number of category scores stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns `true` if no category scores have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Compute the weighted-average aggregate score across all categories.
    ///
    /// Returns `0.0` when no scores are present or total weight is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aggregate(&self) -> f32 {
        let total_weight: f32 = self.scores.iter().map(|s| s.weight).sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        let weighted_sum: f32 = self.scores.iter().map(|s| s.weighted_value()).sum();
        weighted_sum / total_weight
    }

    /// Return the category score for a specific [`ScoreCategory`], if present.
    #[must_use]
    pub fn get(&self, category: ScoreCategory) -> Option<&CategoryScore> {
        self.scores.iter().find(|s| s.category == category)
    }

    /// Returns the highest individual score value, or `None` when empty.
    #[must_use]
    pub fn max_score(&self) -> Option<f32> {
        self.scores.iter().map(|s| s.score).reduce(f32::max)
    }

    /// Returns the lowest individual score value, or `None` when empty.
    #[must_use]
    pub fn min_score(&self) -> Option<f32> {
        self.scores.iter().map(|s| s.score).reduce(f32::min)
    }

    /// Returns `true` if every category score passes `threshold`.
    #[must_use]
    pub fn all_pass(&self, threshold: f32) -> bool {
        self.scores.iter().all(|s| s.passes(threshold))
    }

    /// Returns `true` if any category score passes `threshold`.
    #[must_use]
    pub fn any_pass(&self, threshold: f32) -> bool {
        self.scores.iter().any(|s| s.passes(threshold))
    }

    /// Count of category scores that pass `threshold`.
    #[must_use]
    pub fn pass_count(&self, threshold: f32) -> usize {
        self.scores.iter().filter(|s| s.passes(threshold)).count()
    }

    /// Drain and return all stored category scores.
    pub fn into_scores(self) -> Vec<CategoryScore> {
        self.scores
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScoreCategory ──

    #[test]
    fn test_labels_non_empty() {
        for cat in [
            ScoreCategory::VisualQuality,
            ScoreCategory::Aesthetic,
            ScoreCategory::Motion,
            ScoreCategory::FacePresence,
            ScoreCategory::Relevance,
            ScoreCategory::AudioClarity,
            ScoreCategory::Engagement,
        ] {
            assert!(!cat.label().is_empty(), "label empty for {cat:?}");
        }
    }

    #[test]
    fn test_is_video_category() {
        assert!(ScoreCategory::VisualQuality.is_video_category());
        assert!(ScoreCategory::Aesthetic.is_video_category());
        assert!(!ScoreCategory::AudioClarity.is_video_category());
    }

    // ── CategoryScore ──

    #[test]
    fn test_score_clamped_high() {
        let cs = CategoryScore::new(ScoreCategory::Aesthetic, 1.5, 1.0);
        assert!((cs.score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_clamped_low() {
        let cs = CategoryScore::new(ScoreCategory::Motion, -0.5, 1.0);
        assert!((cs.score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_weight_clamped_negative() {
        let cs = CategoryScore::new(ScoreCategory::Relevance, 0.5, -1.0);
        assert!(cs.weight >= 0.0);
    }

    #[test]
    fn test_unit_weight_one() {
        let cs = CategoryScore::unit(ScoreCategory::Engagement, 0.7);
        assert!((cs.weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_passes() {
        let cs = CategoryScore::unit(ScoreCategory::VisualQuality, 0.8);
        assert!(cs.passes(0.5));
        assert!(!cs.passes(0.9));
    }

    #[test]
    fn test_weighted_value() {
        let cs = CategoryScore::new(ScoreCategory::Aesthetic, 0.5, 2.0);
        assert!((cs.weighted_value() - 1.0).abs() < f32::EPSILON);
    }

    // ── SceneScore ──

    #[test]
    fn test_empty_aggregate_is_zero() {
        let ss = SceneScore::new();
        assert!((ss.aggregate() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_single_score_aggregate() {
        let mut ss = SceneScore::new();
        ss.add(CategoryScore::unit(ScoreCategory::VisualQuality, 0.6));
        assert!((ss.aggregate() - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_weight_average() {
        let mut ss = SceneScore::new();
        ss.add(CategoryScore::unit(ScoreCategory::VisualQuality, 0.4));
        ss.add(CategoryScore::unit(ScoreCategory::Aesthetic, 0.6));
        assert!((ss.aggregate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_average() {
        let mut ss = SceneScore::new();
        // weight 1 × score 1.0  +  weight 3 × score 0.0  → 0.25
        ss.add(CategoryScore::new(ScoreCategory::VisualQuality, 1.0, 1.0));
        ss.add(CategoryScore::new(ScoreCategory::Aesthetic, 0.0, 3.0));
        assert!((ss.aggregate() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_get_category_found() {
        let mut ss = SceneScore::new();
        ss.add(CategoryScore::unit(ScoreCategory::Motion, 0.7));
        let cs = ss
            .get(ScoreCategory::Motion)
            .expect("should succeed in test");
        assert!((cs.score - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_get_category_not_found() {
        let ss = SceneScore::new();
        assert!(ss.get(ScoreCategory::FacePresence).is_none());
    }

    #[test]
    fn test_max_min_scores() {
        let mut ss = SceneScore::new();
        ss.add(CategoryScore::unit(ScoreCategory::VisualQuality, 0.9));
        ss.add(CategoryScore::unit(ScoreCategory::Motion, 0.2));
        assert!((ss.max_score().expect("should succeed in test") - 0.9).abs() < f32::EPSILON);
        assert!((ss.min_score().expect("should succeed in test") - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_all_pass() {
        let mut ss = SceneScore::new();
        ss.add(CategoryScore::unit(ScoreCategory::VisualQuality, 0.8));
        ss.add(CategoryScore::unit(ScoreCategory::Aesthetic, 0.9));
        assert!(ss.all_pass(0.7));
        assert!(!ss.all_pass(0.85));
    }

    #[test]
    fn test_any_pass() {
        let mut ss = SceneScore::new();
        ss.add(CategoryScore::unit(ScoreCategory::VisualQuality, 0.3));
        ss.add(CategoryScore::unit(ScoreCategory::Aesthetic, 0.9));
        assert!(ss.any_pass(0.8));
        assert!(!ss.any_pass(0.95));
    }

    #[test]
    fn test_pass_count() {
        let mut ss = SceneScore::new();
        ss.add(CategoryScore::unit(ScoreCategory::VisualQuality, 0.9));
        ss.add(CategoryScore::unit(ScoreCategory::Motion, 0.4));
        ss.add(CategoryScore::unit(ScoreCategory::Engagement, 0.8));
        assert_eq!(ss.pass_count(0.7), 2);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut ss = SceneScore::new();
        assert!(ss.is_empty());
        ss.add(CategoryScore::unit(ScoreCategory::Relevance, 0.5));
        assert_eq!(ss.len(), 1);
        assert!(!ss.is_empty());
    }

    #[test]
    fn test_into_scores() {
        let mut ss = SceneScore::new();
        ss.add(CategoryScore::unit(ScoreCategory::Aesthetic, 0.5));
        ss.add(CategoryScore::unit(ScoreCategory::Engagement, 0.6));
        let scores = ss.into_scores();
        assert_eq!(scores.len(), 2);
    }
}
