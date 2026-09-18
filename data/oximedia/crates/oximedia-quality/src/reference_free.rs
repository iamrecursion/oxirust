//! Reference-free quality metrics aggregation.
//!
//! This module provides tools for aggregating multiple no-reference quality
//! metrics into a unified quality assessment without needing a reference frame.

#![allow(dead_code)]

/// Aggregated suite of reference-free (no-reference) quality metrics.
///
/// Combines BRISQUE, NIQE, noise, and blur scores into a single structure
/// for unified quality assessment.
#[derive(Clone, Debug)]
pub struct ReferenceFreeSuite {
    /// BRISQUE score (lower is better, 0–100)
    pub brisque_score: f32,
    /// NIQE score (lower is better, typically 0–15)
    pub niqe_score: f32,
    /// Estimated noise level (0.0–1.0)
    pub noise_level: f32,
    /// Estimated blur level (0.0–1.0)
    pub blur_level: f32,
}

impl ReferenceFreeSuite {
    /// Creates a new `ReferenceFreeSuite` from individual metric scores.
    #[must_use]
    pub fn new(brisque_score: f32, niqe_score: f32, noise_level: f32, blur_level: f32) -> Self {
        Self {
            brisque_score,
            niqe_score,
            noise_level,
            blur_level,
        }
    }

    /// Computes a weighted overall quality score (lower is better).
    ///
    /// Weights: BRISQUE 40%, NIQE 30%, noise 15%, blur 15%.
    /// NIQE is normalized from its typical 0–15 range to 0–100.
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        let niqe_normalized = (self.niqe_score / 15.0 * 100.0).min(100.0);
        let noise_normalized = self.noise_level * 100.0;
        let blur_normalized = self.blur_level * 100.0;

        0.40 * self.brisque_score
            + 0.30 * niqe_normalized
            + 0.15 * noise_normalized
            + 0.15 * blur_normalized
    }

    /// Returns `true` if the overall score is below `max_score` (i.e., acceptable).
    #[must_use]
    pub fn is_acceptable(&self, max_score: f32) -> bool {
        self.overall_score() <= max_score
    }
}

/// Perceptual quality grade derived from a numeric score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityGrade {
    /// Score 0–20: perceptually transparent.
    Excellent,
    /// Score 20–40: minor artefacts, not annoying.
    Good,
    /// Score 40–60: artefacts visible but tolerable.
    Acceptable,
    /// Score 60–80: artefacts clearly visible and annoying.
    Poor,
    /// Score 80–100: severe degradation.
    Unacceptable,
}

impl QualityGrade {
    /// Returns the grade corresponding to the given score (lower = better).
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        if score < 20.0 {
            Self::Excellent
        } else if score < 40.0 {
            Self::Good
        } else if score < 60.0 {
            Self::Acceptable
        } else if score < 80.0 {
            Self::Poor
        } else {
            Self::Unacceptable
        }
    }

    /// Returns a short human-readable description of the grade.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Excellent => "Excellent – perceptually transparent",
            Self::Good => "Good – minor artefacts, not annoying",
            Self::Acceptable => "Acceptable – artefacts visible but tolerable",
            Self::Poor => "Poor – artefacts clearly visible and annoying",
            Self::Unacceptable => "Unacceptable – severe degradation",
        }
    }
}

/// Grades a frame described by its `ReferenceFreeSuite`.
#[must_use]
pub fn grade_frame(suite: &ReferenceFreeSuite) -> QualityGrade {
    QualityGrade::from_score(suite.overall_score())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_suite() -> ReferenceFreeSuite {
        ReferenceFreeSuite::new(0.0, 0.0, 0.0, 0.0)
    }

    fn poor_suite() -> ReferenceFreeSuite {
        ReferenceFreeSuite::new(80.0, 12.0, 0.8, 0.9)
    }

    #[test]
    fn test_new_stores_fields() {
        let s = ReferenceFreeSuite::new(10.0, 3.0, 0.1, 0.2);
        assert!((s.brisque_score - 10.0).abs() < f32::EPSILON);
        assert!((s.niqe_score - 3.0).abs() < f32::EPSILON);
        assert!((s.noise_level - 0.1).abs() < f32::EPSILON);
        assert!((s.blur_level - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_overall_score_perfect_is_zero() {
        assert!((perfect_suite().overall_score()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_overall_score_bounded_above() {
        let worst = ReferenceFreeSuite::new(100.0, 15.0, 1.0, 1.0);
        let score = worst.overall_score();
        assert!(score <= 100.0, "score={score}");
    }

    #[test]
    fn test_overall_score_positive() {
        let s = ReferenceFreeSuite::new(30.0, 5.0, 0.3, 0.2);
        assert!(s.overall_score() > 0.0);
    }

    #[test]
    fn test_is_acceptable_true_when_below_max() {
        let s = ReferenceFreeSuite::new(10.0, 2.0, 0.05, 0.05);
        assert!(s.is_acceptable(100.0));
    }

    #[test]
    fn test_is_acceptable_false_when_above_max() {
        let s = poor_suite();
        assert!(!s.is_acceptable(10.0));
    }

    #[test]
    fn test_is_acceptable_at_boundary() {
        let s = ReferenceFreeSuite::new(0.0, 0.0, 0.0, 0.0);
        assert!(s.is_acceptable(0.0));
    }

    #[test]
    fn test_grade_excellent() {
        assert_eq!(QualityGrade::from_score(0.0), QualityGrade::Excellent);
        assert_eq!(QualityGrade::from_score(19.9), QualityGrade::Excellent);
    }

    #[test]
    fn test_grade_good() {
        assert_eq!(QualityGrade::from_score(20.0), QualityGrade::Good);
        assert_eq!(QualityGrade::from_score(39.9), QualityGrade::Good);
    }

    #[test]
    fn test_grade_acceptable() {
        assert_eq!(QualityGrade::from_score(40.0), QualityGrade::Acceptable);
        assert_eq!(QualityGrade::from_score(59.9), QualityGrade::Acceptable);
    }

    #[test]
    fn test_grade_poor() {
        assert_eq!(QualityGrade::from_score(60.0), QualityGrade::Poor);
        assert_eq!(QualityGrade::from_score(79.9), QualityGrade::Poor);
    }

    #[test]
    fn test_grade_unacceptable() {
        assert_eq!(QualityGrade::from_score(80.0), QualityGrade::Unacceptable);
        assert_eq!(QualityGrade::from_score(100.0), QualityGrade::Unacceptable);
    }

    #[test]
    fn test_description_not_empty() {
        for grade in [
            QualityGrade::Excellent,
            QualityGrade::Good,
            QualityGrade::Acceptable,
            QualityGrade::Poor,
            QualityGrade::Unacceptable,
        ] {
            assert!(!grade.description().is_empty());
        }
    }

    #[test]
    fn test_grade_frame_excellent_for_perfect() {
        assert_eq!(grade_frame(&perfect_suite()), QualityGrade::Excellent);
    }

    #[test]
    fn test_grade_frame_poor_for_bad_input() {
        let grade = grade_frame(&poor_suite());
        assert!(matches!(
            grade,
            QualityGrade::Poor | QualityGrade::Unacceptable
        ));
    }

    #[test]
    fn test_niqe_clamped_at_100() {
        // NIQE value way above 15 should not push score above 100
        let s = ReferenceFreeSuite::new(0.0, 1000.0, 0.0, 0.0);
        assert!(s.overall_score() <= 100.0);
    }
}
