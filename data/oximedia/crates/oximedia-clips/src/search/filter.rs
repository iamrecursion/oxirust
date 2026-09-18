//! Advanced filtering for clips.

use crate::clip::Clip;
use crate::logging::Rating;
use chrono::{DateTime, Utc};

/// Filter criteria for clips.
#[derive(Debug, Clone, Default)]
pub struct FilterCriteria {
    /// Filter by rating.
    pub rating: Option<RatingFilter>,

    /// Filter by favorite status.
    pub is_favorite: Option<bool>,

    /// Filter by rejected status.
    pub is_rejected: Option<bool>,

    /// Filter by keywords.
    pub keywords: Vec<String>,

    /// Filter by date range.
    pub date_range: Option<DateRange>,

    /// Filter by duration range.
    pub duration_range: Option<DurationRange>,

    /// Filter by file extension.
    pub file_extension: Option<String>,

    /// Filter by having markers.
    pub has_markers: Option<bool>,
}

/// Rating filter options.
#[derive(Debug, Clone, Copy)]
pub enum RatingFilter {
    /// Exact rating.
    Exact(Rating),
    /// Minimum rating.
    Minimum(Rating),
    /// Maximum rating.
    Maximum(Rating),
    /// Rating range.
    Range(Rating, Rating),
}

/// Date range filter.
#[derive(Debug, Clone, Copy)]
pub struct DateRange {
    /// Start date.
    pub start: DateTime<Utc>,
    /// End date.
    pub end: DateTime<Utc>,
}

/// Duration range filter (in frames).
#[derive(Debug, Clone, Copy)]
pub struct DurationRange {
    /// Minimum duration.
    pub min: i64,
    /// Maximum duration.
    pub max: i64,
}

/// Filter for clips.
#[derive(Debug, Clone, Default)]
pub struct ClipFilter {
    criteria: FilterCriteria,
}

impl ClipFilter {
    /// Creates a new clip filter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            criteria: FilterCriteria::default(),
        }
    }

    /// Sets the rating filter.
    #[must_use]
    pub fn with_rating(mut self, rating: RatingFilter) -> Self {
        self.criteria.rating = Some(rating);
        self
    }

    /// Sets the favorite filter.
    #[must_use]
    pub fn with_favorite(mut self, is_favorite: bool) -> Self {
        self.criteria.is_favorite = Some(is_favorite);
        self
    }

    /// Sets the rejected filter.
    #[must_use]
    pub fn with_rejected(mut self, is_rejected: bool) -> Self {
        self.criteria.is_rejected = Some(is_rejected);
        self
    }

    /// Adds a keyword filter.
    #[must_use]
    pub fn with_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.criteria.keywords.push(keyword.into());
        self
    }

    /// Sets the date range filter.
    #[must_use]
    pub fn with_date_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.criteria.date_range = Some(DateRange { start, end });
        self
    }

    /// Sets the duration range filter.
    #[must_use]
    pub fn with_duration_range(mut self, min: i64, max: i64) -> Self {
        self.criteria.duration_range = Some(DurationRange { min, max });
        self
    }

    /// Sets the file extension filter.
    #[must_use]
    pub fn with_extension(mut self, extension: impl Into<String>) -> Self {
        self.criteria.file_extension = Some(extension.into());
        self
    }

    /// Sets the has markers filter.
    #[must_use]
    pub fn with_has_markers(mut self, has_markers: bool) -> Self {
        self.criteria.has_markers = Some(has_markers);
        self
    }

    /// Applies the filter to a list of clips.
    #[must_use]
    pub fn apply<'a>(&self, clips: &'a [Clip]) -> Vec<&'a Clip> {
        clips.iter().filter(|clip| self.matches(clip)).collect()
    }

    /// Checks if a clip matches the filter criteria.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn matches(&self, clip: &Clip) -> bool {
        // Rating filter
        if let Some(rating_filter) = self.criteria.rating {
            match rating_filter {
                RatingFilter::Exact(rating) => {
                    if clip.rating != rating {
                        return false;
                    }
                }
                RatingFilter::Minimum(min) => {
                    if clip.rating < min {
                        return false;
                    }
                }
                RatingFilter::Maximum(max) => {
                    if clip.rating > max {
                        return false;
                    }
                }
                RatingFilter::Range(min, max) => {
                    if clip.rating < min || clip.rating > max {
                        return false;
                    }
                }
            }
        }

        // Favorite filter
        if let Some(is_favorite) = self.criteria.is_favorite {
            if clip.is_favorite != is_favorite {
                return false;
            }
        }

        // Rejected filter
        if let Some(is_rejected) = self.criteria.is_rejected {
            if clip.is_rejected != is_rejected {
                return false;
            }
        }

        // Keywords filter
        if !self.criteria.keywords.is_empty() {
            let has_all_keywords = self
                .criteria
                .keywords
                .iter()
                .all(|kw| clip.keywords.contains(kw));
            if !has_all_keywords {
                return false;
            }
        }

        // Date range filter
        if let Some(date_range) = self.criteria.date_range {
            if clip.created_at < date_range.start || clip.created_at > date_range.end {
                return false;
            }
        }

        // Duration range filter
        if let Some(duration_range) = self.criteria.duration_range {
            if let Some(duration) = clip.effective_duration() {
                if duration < duration_range.min || duration > duration_range.max {
                    return false;
                }
            } else {
                return false;
            }
        }

        // File extension filter
        if let Some(ext) = &self.criteria.file_extension {
            if let Some(clip_ext) = clip.file_path.extension() {
                if clip_ext.to_string_lossy().to_lowercase() != ext.to_lowercase() {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Has markers filter
        if let Some(has_markers) = self.criteria.has_markers {
            if clip.markers.is_empty() == has_markers {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_rating_filter() {
        let mut clip1 = Clip::new(PathBuf::from("/test1.mov"));
        clip1.set_rating(Rating::FiveStars);

        let mut clip2 = Clip::new(PathBuf::from("/test2.mov"));
        clip2.set_rating(Rating::ThreeStars);

        let clips = vec![clip1, clip2];

        let filter = ClipFilter::new().with_rating(RatingFilter::Minimum(Rating::FourStars));
        let results = filter.apply(&clips);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rating, Rating::FiveStars);
    }

    #[test]
    fn test_favorite_filter() {
        let mut clip1 = Clip::new(PathBuf::from("/test1.mov"));
        clip1.set_favorite(true);

        let clip2 = Clip::new(PathBuf::from("/test2.mov"));

        let clips = vec![clip1, clip2];

        let filter = ClipFilter::new().with_favorite(true);
        let results = filter.apply(&clips);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_favorite);
    }

    #[test]
    fn test_keyword_filter() {
        let mut clip1 = Clip::new(PathBuf::from("/test1.mov"));
        clip1.add_keyword("interview");
        clip1.add_keyword("john");

        let mut clip2 = Clip::new(PathBuf::from("/test2.mov"));
        clip2.add_keyword("interview");

        let clips = vec![clip1, clip2];

        let filter = ClipFilter::new()
            .with_keyword("interview")
            .with_keyword("john");
        let results = filter.apply(&clips);
        assert_eq!(results.len(), 1);
    }
}
