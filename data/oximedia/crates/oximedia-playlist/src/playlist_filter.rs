#![allow(dead_code)]
//! Playlist filtering by metadata criteria.
//!
//! Provides [`FilterCriteria`] for expressing filter conditions and
//! [`PlaylistFilter`] for applying those conditions to a sequence of items.

use std::time::Duration;

// ---------------------------------------------------------------------------
// Filter criteria
// ---------------------------------------------------------------------------

/// A single filtering predicate that can be applied to a playlist item.
#[derive(Debug, Clone)]
pub enum FilterCriteria {
    /// Keep items whose title contains the given substring (case-insensitive).
    TitleContains(String),
    /// Keep items whose genre matches exactly (case-insensitive).
    Genre(String),
    /// Keep items with duration at or above the given threshold.
    MinDuration(Duration),
    /// Keep items with duration at or below the given threshold.
    MaxDuration(Duration),
    /// Keep items with a rating (0–100) at or above the given value.
    MinRating(u8),
    /// Keep items tagged with all of the given tags.
    HasAllTags(Vec<String>),
    /// Keep items tagged with at least one of the given tags.
    HasAnyTag(Vec<String>),
    /// Logical conjunction: all inner criteria must match.
    And(Vec<FilterCriteria>),
    /// Logical disjunction: at least one inner criterion must match.
    Or(Vec<FilterCriteria>),
    /// Logical negation: inner criterion must *not* match.
    Not(Box<FilterCriteria>),
}

// ---------------------------------------------------------------------------
// Filterable item
// ---------------------------------------------------------------------------

/// A playlist item with enough metadata for filtering.
#[derive(Debug, Clone)]
pub struct FilterableItem {
    /// Display title.
    pub title: String,
    /// Optional genre label.
    pub genre: Option<String>,
    /// Optional duration.
    pub duration: Option<Duration>,
    /// Editorial rating 0–100.
    pub rating: u8,
    /// Arbitrary tags attached to this item.
    pub tags: Vec<String>,
}

impl FilterableItem {
    /// Creates a minimal item with a title and no metadata.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            genre: None,
            duration: None,
            rating: 0,
            tags: Vec::new(),
        }
    }

    /// Attaches a genre.
    pub fn with_genre(mut self, genre: impl Into<String>) -> Self {
        self.genre = Some(genre.into());
        self
    }

    /// Attaches a duration.
    pub fn with_duration(mut self, dur: Duration) -> Self {
        self.duration = Some(dur);
        self
    }

    /// Sets the rating (clamped to 0–100).
    pub fn with_rating(mut self, rating: u8) -> Self {
        self.rating = rating.min(100);
        self
    }

    /// Adds a single tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Adds multiple tags at once.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

impl FilterCriteria {
    /// Evaluates whether `item` satisfies this criterion.
    pub fn matches(&self, item: &FilterableItem) -> bool {
        match self {
            Self::TitleContains(sub) => item.title.to_lowercase().contains(&sub.to_lowercase()),
            Self::Genre(g) => item
                .genre
                .as_deref()
                .is_some_and(|genre| genre.eq_ignore_ascii_case(g)),
            Self::MinDuration(min) => item.duration.is_some_and(|d| d >= *min),
            Self::MaxDuration(max) => item.duration.is_some_and(|d| d <= *max),
            Self::MinRating(min) => item.rating >= *min,
            Self::HasAllTags(required) => required
                .iter()
                .all(|t| item.tags.iter().any(|tag| tag.eq_ignore_ascii_case(t))),
            Self::HasAnyTag(any) => any
                .iter()
                .any(|t| item.tags.iter().any(|tag| tag.eq_ignore_ascii_case(t))),
            Self::And(criteria) => criteria.iter().all(|c| c.matches(item)),
            Self::Or(criteria) => criteria.iter().any(|c| c.matches(item)),
            Self::Not(inner) => !inner.matches(item),
        }
    }
}

// ---------------------------------------------------------------------------
// PlaylistFilter
// ---------------------------------------------------------------------------

/// Applies a [`FilterCriteria`] to a collection of [`FilterableItem`]s.
#[derive(Debug, Clone)]
pub struct PlaylistFilter {
    criteria: FilterCriteria,
}

impl PlaylistFilter {
    /// Creates a new filter from the given criteria.
    pub fn new(criteria: FilterCriteria) -> Self {
        Self { criteria }
    }

    /// Applies the filter, returning only the items that match.
    pub fn apply<'a>(&self, items: &'a [FilterableItem]) -> Vec<&'a FilterableItem> {
        items
            .iter()
            .filter(|item| self.criteria.matches(item))
            .collect()
    }

    /// Applies the filter in-place, retaining only the matching items.
    pub fn apply_owned(&self, items: Vec<FilterableItem>) -> Vec<FilterableItem> {
        items
            .into_iter()
            .filter(|item| self.criteria.matches(item))
            .collect()
    }

    /// Returns the number of items that pass the filter without cloning.
    pub fn count_matching(&self, items: &[FilterableItem]) -> usize {
        items
            .iter()
            .filter(|item| self.criteria.matches(item))
            .count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_items() -> Vec<FilterableItem> {
        vec![
            FilterableItem::new("Jazz Evening")
                .with_genre("Jazz")
                .with_duration(Duration::from_secs(240))
                .with_rating(85)
                .with_tags(["live", "instrumental"]),
            FilterableItem::new("Rock Classics")
                .with_genre("Rock")
                .with_duration(Duration::from_secs(180))
                .with_rating(70)
                .with_tags(["studio"]),
            FilterableItem::new("Evening Jazz Special")
                .with_genre("Jazz")
                .with_duration(Duration::from_secs(300))
                .with_rating(90)
                .with_tags(["live", "featured"]),
            FilterableItem::new("Ambient Chill")
                .with_duration(Duration::from_secs(60))
                .with_rating(50)
                .with_tags(["background"]),
        ]
    }

    #[test]
    fn test_title_contains_case_insensitive() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::TitleContains("jazz".to_string()));
        let results = f.apply(&items);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_genre_match() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::Genre("Jazz".to_string()));
        assert_eq!(f.count_matching(&items), 2);
    }

    #[test]
    fn test_genre_no_match_missing_genre() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::Genre("Classical".to_string()));
        assert_eq!(f.count_matching(&items), 0);
    }

    #[test]
    fn test_min_duration() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::MinDuration(Duration::from_secs(200)));
        // Jazz Evening (240s), Evening Jazz Special (300s)
        assert_eq!(f.count_matching(&items), 2);
    }

    #[test]
    fn test_max_duration() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::MaxDuration(Duration::from_secs(180)));
        // Rock Classics (180s), Ambient Chill (60s)
        assert_eq!(f.count_matching(&items), 2);
    }

    #[test]
    fn test_min_rating() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::MinRating(80));
        // Jazz Evening (85), Evening Jazz Special (90)
        assert_eq!(f.count_matching(&items), 2);
    }

    #[test]
    fn test_has_all_tags() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::HasAllTags(vec![
            "live".to_string(),
            "instrumental".to_string(),
        ]));
        assert_eq!(f.count_matching(&items), 1);
    }

    #[test]
    fn test_has_any_tag() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::HasAnyTag(vec![
            "featured".to_string(),
            "studio".to_string(),
        ]));
        assert_eq!(f.count_matching(&items), 2);
    }

    #[test]
    fn test_and_criteria() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::And(vec![
            FilterCriteria::Genre("Jazz".to_string()),
            FilterCriteria::MinRating(88),
        ]));
        // Only Evening Jazz Special (Jazz, rating 90)
        assert_eq!(f.count_matching(&items), 1);
    }

    #[test]
    fn test_or_criteria() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::Or(vec![
            FilterCriteria::Genre("Rock".to_string()),
            FilterCriteria::MinRating(88),
        ]));
        // Rock Classics + Evening Jazz Special
        assert_eq!(f.count_matching(&items), 2);
    }

    #[test]
    fn test_not_criteria() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::Not(Box::new(FilterCriteria::Genre(
            "Jazz".to_string(),
        ))));
        // Rock Classics + Ambient Chill
        assert_eq!(f.count_matching(&items), 2);
    }

    #[test]
    fn test_apply_returns_references() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::TitleContains("Rock".to_string()));
        let results = f.apply(&items);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Rock Classics");
    }

    #[test]
    fn test_apply_owned_consumes_items() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::MinRating(80));
        let owned = f.apply_owned(items);
        assert_eq!(owned.len(), 2);
    }

    #[test]
    fn test_filter_no_duration_min_duration() {
        // Item with no duration should not pass MinDuration filter
        let items = vec![FilterableItem::new("No Duration")];
        let f = PlaylistFilter::new(FilterCriteria::MinDuration(Duration::from_secs(1)));
        assert_eq!(f.count_matching(&items), 0);
    }

    #[test]
    fn test_filter_rating_clamped_to_100() {
        let item = FilterableItem::new("x").with_rating(255);
        assert_eq!(item.rating, 100);
    }

    #[test]
    fn test_has_all_tags_empty_required_matches_all() {
        let items = make_items();
        let f = PlaylistFilter::new(FilterCriteria::HasAllTags(vec![]));
        assert_eq!(f.count_matching(&items), items.len());
    }

    #[test]
    fn test_nested_and_or() {
        let items = make_items();
        // (Jazz AND live) OR (Rock)
        let f = PlaylistFilter::new(FilterCriteria::Or(vec![
            FilterCriteria::And(vec![
                FilterCriteria::Genre("Jazz".to_string()),
                FilterCriteria::HasAnyTag(vec!["live".to_string()]),
            ]),
            FilterCriteria::Genre("Rock".to_string()),
        ]));
        // Jazz Evening + Evening Jazz Special + Rock Classics = 3
        assert_eq!(f.count_matching(&items), 3);
    }
}
