//! Shot quality rating and scoring.
//!
//! Provides a multi-dimensional quality model for scoring individual shots
//! across technical and aesthetic axes, producing an aggregate rating that
//! can be used for shot selection or filtering.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::fmt;

// ──────────────────────────────────────────────────────────────────────────────
// Individual dimension scores
// ──────────────────────────────────────────────────────────────────────────────

/// A bounded quality score in the range \[0.0, 1.0\].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Score(f32);

impl Score {
    /// Create a score, clamping to \[0.0, 1.0\].
    #[must_use]
    pub fn new(v: f32) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    /// Return the raw `f32` value.
    #[must_use]
    pub fn value(self) -> f32 {
        self.0
    }

    /// Return `true` if the score is at or above the threshold.
    #[must_use]
    pub fn passes(self, threshold: f32) -> bool {
        self.0 >= threshold
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Star rating
// ──────────────────────────────────────────────────────────────────────────────

/// A 1–5 star editorial rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StarRating {
    /// Reject / unusable.
    One,
    /// Below average.
    Two,
    /// Average / acceptable.
    Three,
    /// Good.
    Four,
    /// Excellent / hero take.
    Five,
}

impl StarRating {
    /// Convert a normalised score (0–1) to a star rating.
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        if score >= 0.9 {
            Self::Five
        } else if score >= 0.7 {
            Self::Four
        } else if score >= 0.5 {
            Self::Three
        } else if score >= 0.3 {
            Self::Two
        } else {
            Self::One
        }
    }

    /// Return the numeric value (1–5).
    #[must_use]
    pub fn stars(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Five => 5,
        }
    }

    /// Return `true` if this rating is considered selectable (≥ 3 stars).
    #[must_use]
    pub fn is_selectable(self) -> bool {
        self >= Self::Three
    }
}

impl fmt::Display for StarRating {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} star(s)", self.stars())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Quality dimensions
// ──────────────────────────────────────────────────────────────────────────────

/// Technical quality dimension of a shot.
#[derive(Debug, Clone, Copy)]
pub struct TechnicalQuality {
    /// Focus sharpness (0 – 1).
    pub focus: Score,
    /// Exposure correctness (0 – 1).
    pub exposure: Score,
    /// Motion blur / stability (0 – 1, 1 = no blur/shake).
    pub stability: Score,
    /// Noise level (0 – 1, 1 = clean).
    pub noise: Score,
}

impl TechnicalQuality {
    /// Compute an aggregate technical score as a weighted mean.
    ///
    /// Weights: focus 35 %, exposure 30 %, stability 25 %, noise 10 %.
    #[must_use]
    pub fn aggregate(&self) -> Score {
        let v = self.focus.value() * 0.35
            + self.exposure.value() * 0.30
            + self.stability.value() * 0.25
            + self.noise.value() * 0.10;
        Score::new(v)
    }
}

/// Aesthetic quality dimension of a shot.
#[derive(Debug, Clone, Copy)]
pub struct AestheticQuality {
    /// Composition score (rule of thirds, balance, etc.) (0 – 1).
    pub composition: Score,
    /// Lighting quality (0 – 1).
    pub lighting: Score,
    /// Performance / expression (0 – 1; 0 if not applicable).
    pub performance: Score,
    /// Story / narrative value (0 – 1).
    pub narrative: Score,
}

impl AestheticQuality {
    /// Compute an aggregate aesthetic score as a simple mean.
    #[must_use]
    pub fn aggregate(&self) -> Score {
        let v = (self.composition.value()
            + self.lighting.value()
            + self.performance.value()
            + self.narrative.value())
            / 4.0;
        Score::new(v)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Full shot rating
// ──────────────────────────────────────────────────────────────────────────────

/// Full multi-dimensional rating for a single shot.
#[derive(Debug, Clone, Copy)]
pub struct ShotRating {
    /// Technical quality.
    pub technical: TechnicalQuality,
    /// Aesthetic quality.
    pub aesthetic: AestheticQuality,
    /// Override from a human reviewer (if present).
    pub manual_override: Option<Score>,
}

impl ShotRating {
    /// Create a shot rating with default (neutral 0.5) scores.
    #[must_use]
    pub fn neutral() -> Self {
        let half = Score::new(0.5);
        Self {
            technical: TechnicalQuality {
                focus: half,
                exposure: half,
                stability: half,
                noise: half,
            },
            aesthetic: AestheticQuality {
                composition: half,
                lighting: half,
                performance: half,
                narrative: half,
            },
            manual_override: None,
        }
    }

    /// Compute the overall quality score.
    ///
    /// If a manual override is present, it is used directly.  Otherwise
    /// technical (60 %) and aesthetic (40 %) scores are blended.
    #[must_use]
    pub fn overall(&self) -> Score {
        if let Some(ov) = self.manual_override {
            return ov;
        }
        let v = self.technical.aggregate().value() * 0.6 + self.aesthetic.aggregate().value() * 0.4;
        Score::new(v)
    }

    /// Return the star rating corresponding to the overall score.
    #[must_use]
    pub fn stars(&self) -> StarRating {
        StarRating::from_score(self.overall().value())
    }

    /// Return `true` if this shot meets the minimum quality bar.
    #[must_use]
    pub fn passes(&self, min_score: f32) -> bool {
        self.overall().passes(min_score)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Rating comparison helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compare two ratings and return which is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonResult {
    /// First shot is better.
    First,
    /// Second shot is better.
    Second,
    /// Both shots are equally rated.
    Equal,
}

/// Compare two `ShotRating` values and return which is better.
#[must_use]
pub fn compare_ratings(a: &ShotRating, b: &ShotRating) -> ComparisonResult {
    let sa = a.overall().value();
    let sb = b.overall().value();
    if (sa - sb).abs() < 0.01 {
        ComparisonResult::Equal
    } else if sa > sb {
        ComparisonResult::First
    } else {
        ComparisonResult::Second
    }
}

/// Filter a slice of ratings, returning indices that pass the minimum score.
#[must_use]
pub fn filter_selectable(ratings: &[ShotRating], min_score: f32) -> Vec<usize> {
    ratings
        .iter()
        .enumerate()
        .filter_map(|(i, r)| if r.passes(min_score) { Some(i) } else { None })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_clamping_high() {
        let s = Score::new(1.5);
        assert_eq!(s.value(), 1.0);
    }

    #[test]
    fn test_score_clamping_low() {
        let s = Score::new(-0.5);
        assert_eq!(s.value(), 0.0);
    }

    #[test]
    fn test_score_passes() {
        let s = Score::new(0.8);
        assert!(s.passes(0.7));
        assert!(!s.passes(0.9));
    }

    #[test]
    fn test_score_display() {
        let s = Score::new(0.75);
        assert_eq!(s.to_string(), "0.75");
    }

    #[test]
    fn test_star_rating_from_score_five() {
        assert_eq!(StarRating::from_score(0.95), StarRating::Five);
    }

    #[test]
    fn test_star_rating_from_score_one() {
        assert_eq!(StarRating::from_score(0.1), StarRating::One);
    }

    #[test]
    fn test_star_rating_stars_value() {
        assert_eq!(StarRating::Four.stars(), 4);
        assert_eq!(StarRating::Two.stars(), 2);
    }

    #[test]
    fn test_star_rating_is_selectable() {
        assert!(StarRating::Three.is_selectable());
        assert!(!StarRating::Two.is_selectable());
    }

    #[test]
    fn test_technical_quality_aggregate() {
        let tq = TechnicalQuality {
            focus: Score::new(1.0),
            exposure: Score::new(1.0),
            stability: Score::new(1.0),
            noise: Score::new(1.0),
        };
        assert!((tq.aggregate().value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_aesthetic_quality_aggregate_zero() {
        let aq = AestheticQuality {
            composition: Score::new(0.0),
            lighting: Score::new(0.0),
            performance: Score::new(0.0),
            narrative: Score::new(0.0),
        };
        assert_eq!(aq.aggregate().value(), 0.0);
    }

    #[test]
    fn test_shot_rating_neutral_overall() {
        let r = ShotRating::neutral();
        // All scores are 0.5 → overall should be ~0.5.
        assert!((r.overall().value() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_shot_rating_manual_override() {
        let mut r = ShotRating::neutral();
        r.manual_override = Some(Score::new(0.9));
        assert!((r.overall().value() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_shot_rating_stars() {
        let mut r = ShotRating::neutral();
        r.manual_override = Some(Score::new(0.95));
        assert_eq!(r.stars(), StarRating::Five);
    }

    #[test]
    fn test_compare_ratings_first_better() {
        let mut a = ShotRating::neutral();
        a.manual_override = Some(Score::new(0.9));
        let b = ShotRating::neutral();
        assert_eq!(compare_ratings(&a, &b), ComparisonResult::First);
    }

    #[test]
    fn test_compare_ratings_equal() {
        let a = ShotRating::neutral();
        let b = ShotRating::neutral();
        assert_eq!(compare_ratings(&a, &b), ComparisonResult::Equal);
    }

    #[test]
    fn test_filter_selectable() {
        let mut high = ShotRating::neutral();
        high.manual_override = Some(Score::new(0.8));
        let low = ShotRating::neutral();
        let ratings = vec![high, low];
        let selected = filter_selectable(&ratings, 0.7);
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn test_star_rating_display() {
        assert_eq!(StarRating::Five.to_string(), "5 star(s)");
    }
}
