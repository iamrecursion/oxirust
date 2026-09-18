//! Content filtering for media access control.
//!
//! Provides `FilterCriteria`, `ContentFilter`, and `ContentFilterChain`.

#![allow(dead_code)]

/// A single filtering criterion that can be applied to a media item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterCriteria {
    /// Only allow content with the given MIME type prefix (e.g. "video/").
    MimeType(String),
    /// Only allow content with a specific language tag (e.g. "en").
    Language(String),
    /// Reject content whose size in bytes exceeds the limit.
    MaxSizeBytes(u64),
    /// Only allow content tagged with this category label.
    Category(String),
    /// Reject content with an age rating above this level (0–18).
    MaxAgeRating(u8),
    /// Only allow content from the specified region code (ISO 3166-1 alpha-2).
    Region(String),
}

impl FilterCriteria {
    /// Returns `true` if this criterion *excludes* content (block-list semantics)
    /// rather than selecting it (allow-list semantics).
    #[must_use]
    pub fn is_exclusive(&self) -> bool {
        matches!(
            self,
            FilterCriteria::MaxSizeBytes(_) | FilterCriteria::MaxAgeRating(_)
        )
    }

    /// Human-readable label for this criterion.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            FilterCriteria::MimeType(m) => format!("MIME type: {m}"),
            FilterCriteria::Language(l) => format!("Language: {l}"),
            FilterCriteria::MaxSizeBytes(n) => format!("Max size: {n} bytes"),
            FilterCriteria::Category(c) => format!("Category: {c}"),
            FilterCriteria::MaxAgeRating(r) => format!("Max age rating: {r}"),
            FilterCriteria::Region(r) => format!("Region: {r}"),
        }
    }
}

/// A minimal representation of a media item used for filtering decisions.
#[derive(Debug, Clone, Default)]
pub struct MediaItem {
    /// MIME type of the item (e.g. "video/mp4").
    pub mime_type: String,
    /// BCP-47 language tag (e.g. "en-US").
    pub language: String,
    /// Size of the item in bytes.
    pub size_bytes: u64,
    /// Category labels (e.g. "documentary", "sports").
    pub categories: Vec<String>,
    /// Age rating 0–18.
    pub age_rating: u8,
    /// ISO 3166-1 alpha-2 region code (e.g. "US").
    pub region: String,
}

impl MediaItem {
    /// Create a minimal media item.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set MIME type.
    pub fn with_mime(mut self, mime: impl Into<String>) -> Self {
        self.mime_type = mime.into();
        self
    }

    /// Set language.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Set size in bytes.
    #[must_use]
    pub fn with_size(mut self, bytes: u64) -> Self {
        self.size_bytes = bytes;
        self
    }

    /// Add a category.
    pub fn with_category(mut self, cat: impl Into<String>) -> Self {
        self.categories.push(cat.into());
        self
    }

    /// Set age rating.
    #[must_use]
    pub fn with_age_rating(mut self, rating: u8) -> Self {
        self.age_rating = rating;
        self
    }

    /// Set region code.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }
}

/// A filter with a set of criteria; all criteria must pass for an item to match.
#[derive(Debug, Default)]
pub struct ContentFilter {
    criteria: Vec<FilterCriteria>,
}

impl ContentFilter {
    /// Create a new empty filter (passes everything by default).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a criterion to this filter.
    pub fn add_criterion(&mut self, criterion: FilterCriteria) {
        self.criteria.push(criterion);
    }

    /// Returns `true` if the item passes all criteria in this filter.
    #[must_use]
    pub fn matches(&self, item: &MediaItem) -> bool {
        self.criteria
            .iter()
            .all(|c| Self::criterion_matches(c, item))
    }

    fn criterion_matches(criterion: &FilterCriteria, item: &MediaItem) -> bool {
        match criterion {
            FilterCriteria::MimeType(expected) => item.mime_type.starts_with(expected.as_str()),
            FilterCriteria::Language(lang) => item.language.starts_with(lang.as_str()),
            FilterCriteria::MaxSizeBytes(max) => item.size_bytes <= *max,
            FilterCriteria::Category(cat) => item.categories.iter().any(|c| c == cat),
            FilterCriteria::MaxAgeRating(max) => item.age_rating <= *max,
            FilterCriteria::Region(region) => item.region == *region,
        }
    }

    /// Number of criteria in this filter.
    #[must_use]
    pub fn criterion_count(&self) -> usize {
        self.criteria.len()
    }
}

/// A chain of `ContentFilter`s applied in sequence; all must pass.
#[derive(Debug, Default)]
pub struct ContentFilterChain {
    filters: Vec<ContentFilter>,
}

impl ContentFilterChain {
    /// Create a new empty filter chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a filter to the chain.
    pub fn push(&mut self, filter: ContentFilter) {
        self.filters.push(filter);
    }

    /// Apply all filters in the chain to the item.
    ///
    /// Returns `true` only if every filter passes.
    #[must_use]
    pub fn apply(&self, item: &MediaItem) -> bool {
        self.filters.iter().all(|f| f.matches(item))
    }

    /// Number of filters in the chain.
    #[must_use]
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video_item() -> MediaItem {
        MediaItem::new()
            .with_mime("video/mp4")
            .with_language("en-US")
            .with_size(50_000_000)
            .with_category("documentary")
            .with_age_rating(12)
            .with_region("US")
    }

    #[test]
    fn filter_criteria_is_exclusive_max_size() {
        assert!(FilterCriteria::MaxSizeBytes(100).is_exclusive());
    }

    #[test]
    fn filter_criteria_is_exclusive_age_rating() {
        assert!(FilterCriteria::MaxAgeRating(18).is_exclusive());
    }

    #[test]
    fn filter_criteria_not_exclusive_mime() {
        assert!(!FilterCriteria::MimeType("video/".into()).is_exclusive());
    }

    #[test]
    fn filter_criteria_label_mime() {
        let label = FilterCriteria::MimeType("video/mp4".into()).label();
        assert!(label.contains("MIME"));
    }

    #[test]
    fn filter_criteria_label_age_rating() {
        let label = FilterCriteria::MaxAgeRating(16).label();
        assert!(label.contains("16"));
    }

    #[test]
    fn content_filter_empty_passes_all() {
        let filter = ContentFilter::new();
        assert!(filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_mime_type_match() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::MimeType("video/".into()));
        assert!(filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_mime_type_no_match() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::MimeType("audio/".into()));
        assert!(!filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_language_match() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::Language("en".into()));
        assert!(filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_max_size_pass() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::MaxSizeBytes(100_000_000));
        assert!(filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_max_size_fail() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::MaxSizeBytes(1_000));
        assert!(!filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_category_match() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::Category("documentary".into()));
        assert!(filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_age_rating_pass() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::MaxAgeRating(18));
        assert!(filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_age_rating_fail() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::MaxAgeRating(10));
        assert!(!filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_region_match() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::Region("US".into()));
        assert!(filter.matches(&video_item()));
    }

    #[test]
    fn content_filter_region_no_match() {
        let mut filter = ContentFilter::new();
        filter.add_criterion(FilterCriteria::Region("GB".into()));
        assert!(!filter.matches(&video_item()));
    }

    #[test]
    fn filter_chain_all_pass() {
        let mut chain = ContentFilterChain::new();
        let mut f1 = ContentFilter::new();
        f1.add_criterion(FilterCriteria::MimeType("video/".into()));
        let mut f2 = ContentFilter::new();
        f2.add_criterion(FilterCriteria::Region("US".into()));
        chain.push(f1);
        chain.push(f2);
        assert!(chain.apply(&video_item()));
        assert_eq!(chain.filter_count(), 2);
    }

    #[test]
    fn filter_chain_one_fails() {
        let mut chain = ContentFilterChain::new();
        let mut f1 = ContentFilter::new();
        f1.add_criterion(FilterCriteria::MimeType("video/".into()));
        let mut f2 = ContentFilter::new();
        f2.add_criterion(FilterCriteria::Region("GB".into()));
        chain.push(f1);
        chain.push(f2);
        assert!(!chain.apply(&video_item()));
    }
}
