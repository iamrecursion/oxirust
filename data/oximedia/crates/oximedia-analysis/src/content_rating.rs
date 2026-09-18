//! Content sensitivity flagging and rating classification.

#![allow(dead_code)]

/// Individual content flags that may apply to a piece of media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentFlag {
    /// Graphic or realistic violence.
    Violence,
    /// Adult sexual content.
    AdultContent,
    /// Profane or offensive language.
    ProfaneLanguage,
    /// Drug or substance use.
    DrugUse,
    /// Content that may disturb sensitive viewers.
    DisturbingImagery,
    /// Gambling-related content.
    Gambling,
    /// Mild language or suggestive themes.
    MildContent,
}

impl ContentFlag {
    /// `true` for flags that typically require a content warning.
    #[must_use]
    pub fn is_sensitive(&self) -> bool {
        matches!(
            self,
            Self::Violence | Self::AdultContent | Self::DisturbingImagery | Self::DrugUse
        )
    }

    /// Weight of this flag in the rating calculation (0–10).
    #[must_use]
    pub fn weight(&self) -> u32 {
        match self {
            Self::AdultContent => 10,
            Self::Violence => 8,
            Self::DisturbingImagery => 7,
            Self::DrugUse => 6,
            Self::ProfaneLanguage => 4,
            Self::Gambling => 3,
            Self::MildContent => 1,
        }
    }
}

/// Aggregate content rating derived from accumulated flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContentRating {
    /// Suitable for all audiences.
    G,
    /// Parental guidance suggested.
    Pg,
    /// Parents strongly cautioned.
    Pg13,
    /// Restricted – under 17 requires accompaniment.
    R,
    /// No one under 17 admitted.
    Nc17,
}

impl ContentRating {
    /// Short display label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::G => "G",
            Self::Pg => "PG",
            Self::Pg13 => "PG-13",
            Self::R => "R",
            Self::Nc17 => "NC-17",
        }
    }

    /// Minimum combined flag weight that maps to this rating.
    #[must_use]
    fn min_score(&self) -> u32 {
        match self {
            Self::G => 0,
            Self::Pg => 1,
            Self::Pg13 => 4,
            Self::R => 8,
            Self::Nc17 => 12,
        }
    }

    /// Derive a rating from an accumulated score.
    #[must_use]
    pub fn from_score(score: u32) -> Self {
        if score >= Self::Nc17.min_score() {
            Self::Nc17
        } else if score >= Self::R.min_score() {
            Self::R
        } else if score >= Self::Pg13.min_score() {
            Self::Pg13
        } else if score >= Self::Pg.min_score() {
            Self::Pg
        } else {
            Self::G
        }
    }
}

/// Report produced by a `ContentRater`.
#[derive(Debug, Clone)]
pub struct ContentRatingReport {
    /// Final content rating.
    pub rating: ContentRating,
    /// Combined flag weight score.
    pub score: u32,
    /// List of flags that were detected.
    pub flags: Vec<ContentFlag>,
}

impl ContentRatingReport {
    /// `true` when the content requires human review before distribution.
    #[must_use]
    pub fn requires_review(&self) -> bool {
        self.rating >= ContentRating::R || self.flags.iter().any(ContentFlag::is_sensitive)
    }

    /// `true` when no flags were recorded.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.flags.is_empty()
    }
}

/// Accumulates content flags and computes a rating.
#[derive(Debug, Default)]
pub struct ContentRater {
    flags: Vec<ContentFlag>,
}

impl ContentRater {
    /// Create a new rater with no flags.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a flag (duplicates are allowed and each occurrence increases the score).
    pub fn add_flag(&mut self, flag: ContentFlag) {
        self.flags.push(flag);
    }

    /// Compute the current rating from all accumulated flags.
    #[must_use]
    pub fn compute_rating(&self) -> ContentRatingReport {
        let score: u32 = self.flags.iter().map(ContentFlag::weight).sum();
        let rating = ContentRating::from_score(score);
        ContentRatingReport {
            rating,
            score,
            flags: self.flags.clone(),
        }
    }

    /// Clear all recorded flags.
    pub fn reset(&mut self) {
        self.flags.clear();
    }

    /// Number of flags currently recorded.
    #[must_use]
    pub fn flag_count(&self) -> usize {
        self.flags.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_violence_is_sensitive() {
        assert!(ContentFlag::Violence.is_sensitive());
    }

    #[test]
    fn test_mild_content_not_sensitive() {
        assert!(!ContentFlag::MildContent.is_sensitive());
    }

    #[test]
    fn test_profane_language_not_sensitive() {
        assert!(!ContentFlag::ProfaneLanguage.is_sensitive());
    }

    #[test]
    fn test_adult_content_highest_weight() {
        assert_eq!(ContentFlag::AdultContent.weight(), 10);
    }

    #[test]
    fn test_mild_content_lowest_weight() {
        assert_eq!(ContentFlag::MildContent.weight(), 1);
    }

    #[test]
    fn test_rating_label_g() {
        assert_eq!(ContentRating::G.label(), "G");
    }

    #[test]
    fn test_rating_label_nc17() {
        assert_eq!(ContentRating::Nc17.label(), "NC-17");
    }

    #[test]
    fn test_from_score_zero_is_g() {
        assert_eq!(ContentRating::from_score(0), ContentRating::G);
    }

    #[test]
    fn test_from_score_adult_is_nc17() {
        assert_eq!(ContentRating::from_score(12), ContentRating::Nc17);
    }

    #[test]
    fn test_rater_empty_is_clean() {
        let rater = ContentRater::new();
        let report = rater.compute_rating();
        assert!(report.is_clean());
        assert_eq!(report.rating, ContentRating::G);
    }

    #[test]
    fn test_rater_add_flag_increments_count() {
        let mut rater = ContentRater::new();
        rater.add_flag(ContentFlag::MildContent);
        assert_eq!(rater.flag_count(), 1);
    }

    #[test]
    fn test_rater_violence_raises_rating() {
        let mut rater = ContentRater::new();
        rater.add_flag(ContentFlag::Violence);
        let report = rater.compute_rating();
        assert!(report.rating >= ContentRating::R);
    }

    #[test]
    fn test_report_requires_review_for_sensitive_flag() {
        let mut rater = ContentRater::new();
        rater.add_flag(ContentFlag::DisturbingImagery);
        let report = rater.compute_rating();
        assert!(report.requires_review());
    }

    #[test]
    fn test_report_no_review_for_mild() {
        let mut rater = ContentRater::new();
        rater.add_flag(ContentFlag::MildContent);
        let report = rater.compute_rating();
        assert!(!report.requires_review());
    }

    #[test]
    fn test_rater_reset_clears_flags() {
        let mut rater = ContentRater::new();
        rater.add_flag(ContentFlag::Violence);
        rater.reset();
        assert_eq!(rater.flag_count(), 0);
    }

    #[test]
    fn test_multiple_flags_accumulate_score() {
        let mut rater = ContentRater::new();
        rater.add_flag(ContentFlag::MildContent);
        rater.add_flag(ContentFlag::ProfaneLanguage);
        let report = rater.compute_rating();
        assert_eq!(report.score, 5);
    }
}
