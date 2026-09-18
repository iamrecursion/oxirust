//! Catalog search: filters, entries, and result sets for the MAM catalog.
//!
//! Provides `SearchFilter` for building query criteria, `CatalogEntry` for
//! catalog records, `CatalogSearcher` for executing queries, and
//! `SearchResultSet` for working with results.

#![allow(dead_code)]

/// Filter criteria for a catalog search.
#[derive(Debug, Default, Clone)]
pub struct SearchFilter {
    /// Optional keyword to match against title/description.
    pub keyword: Option<String>,
    /// Optional media type filter (e.g. `"video"`, `"audio"`, `"image"`).
    pub media_type: Option<String>,
    /// Optional creator/owner filter.
    pub creator: Option<String>,
    /// Optional minimum duration in seconds.
    pub min_duration_secs: Option<f64>,
    /// Optional maximum duration in seconds.
    pub max_duration_secs: Option<f64>,
    /// Optional earliest creation date (Unix timestamp).
    pub date_from: Option<i64>,
    /// Optional latest creation date (Unix timestamp).
    pub date_to: Option<i64>,
}

impl SearchFilter {
    /// Create an empty (no-op) filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if all filter fields are `None` (i.e. no restrictions).
    pub fn is_empty(&self) -> bool {
        self.keyword.is_none()
            && self.media_type.is_none()
            && self.creator.is_none()
            && self.min_duration_secs.is_none()
            && self.max_duration_secs.is_none()
            && self.date_from.is_none()
            && self.date_to.is_none()
    }

    /// Builder: set keyword filter.
    pub fn with_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.keyword = Some(keyword.into());
        self
    }

    /// Builder: set media type filter.
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        self.media_type = Some(media_type.into());
        self
    }

    /// Builder: set creator filter.
    pub fn with_creator(mut self, creator: impl Into<String>) -> Self {
        self.creator = Some(creator.into());
        self
    }

    /// Builder: set date range.
    pub fn with_date_range(mut self, from: i64, to: i64) -> Self {
        self.date_from = Some(from);
        self.date_to = Some(to);
        self
    }
}

/// A single entry in the media catalog.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    /// Unique asset identifier.
    pub id: u64,
    /// Asset title.
    pub title: String,
    /// Media type (e.g. `"video"`, `"audio"`, `"image"`).
    pub media_type: String,
    /// Creator/owner name.
    pub creator: String,
    /// Duration in seconds (0.0 for non-temporal assets).
    pub duration_secs: f64,
    /// Creation Unix timestamp.
    pub created_at: i64,
    /// Brief description.
    pub description: String,
}

impl CatalogEntry {
    /// Create a new catalog entry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        title: impl Into<String>,
        media_type: impl Into<String>,
        creator: impl Into<String>,
        duration_secs: f64,
        created_at: i64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            media_type: media_type.into(),
            creator: creator.into(),
            duration_secs,
            created_at,
            description: description.into(),
        }
    }

    /// Returns `true` if this entry matches all criteria in `filter`.
    pub fn matches_filter(&self, filter: &SearchFilter) -> bool {
        if let Some(kw) = &filter.keyword {
            let kw_lower = kw.to_lowercase();
            let title_match = self.title.to_lowercase().contains(&kw_lower);
            let desc_match = self.description.to_lowercase().contains(&kw_lower);
            if !title_match && !desc_match {
                return false;
            }
        }
        if let Some(mt) = &filter.media_type {
            if &self.media_type != mt {
                return false;
            }
        }
        if let Some(creator) = &filter.creator {
            if &self.creator != creator {
                return false;
            }
        }
        if let Some(min_dur) = filter.min_duration_secs {
            if self.duration_secs < min_dur {
                return false;
            }
        }
        if let Some(max_dur) = filter.max_duration_secs {
            if self.duration_secs > max_dur {
                return false;
            }
        }
        if let Some(from) = filter.date_from {
            if self.created_at < from {
                return false;
            }
        }
        if let Some(to) = filter.date_to {
            if self.created_at > to {
                return false;
            }
        }
        true
    }
}

/// Executes searches over an in-memory catalog of entries.
#[derive(Debug, Default)]
pub struct CatalogSearcher {
    entries: Vec<CatalogEntry>,
}

impl CatalogSearcher {
    /// Create a new searcher with no entries.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the catalog.
    pub fn add_entry(&mut self, entry: CatalogEntry) {
        self.entries.push(entry);
    }

    /// Add multiple entries.
    pub fn add_entries(&mut self, entries: impl IntoIterator<Item = CatalogEntry>) {
        self.entries.extend(entries);
    }

    /// Number of entries in the catalog.
    pub fn catalog_size(&self) -> usize {
        self.entries.len()
    }

    /// Execute a search and return a `SearchResultSet`.
    pub fn search(&self, filter: &SearchFilter) -> SearchResultSet {
        let matched: Vec<CatalogEntry> = self
            .entries
            .iter()
            .filter(|e| e.matches_filter(filter))
            .cloned()
            .collect();
        SearchResultSet { results: matched }
    }
}

/// The result set returned by `CatalogSearcher::search`.
#[derive(Debug, Default, Clone)]
pub struct SearchResultSet {
    results: Vec<CatalogEntry>,
}

impl SearchResultSet {
    /// Create an empty result set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of results.
    pub fn count(&self) -> usize {
        self.results.len()
    }

    /// Returns `true` if there are no results.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Sort results by creation date, newest first.
    pub fn sort_by_date(&mut self) {
        self.results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    }

    /// Sort results by title (ascending, case-insensitive).
    pub fn sort_by_title(&mut self) {
        self.results
            .sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
    }

    /// Iterate over results.
    pub fn iter(&self) -> impl Iterator<Item = &CatalogEntry> {
        self.results.iter()
    }

    /// Return the first result, if any.
    pub fn first(&self) -> Option<&CatalogEntry> {
        self.results.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(
        id: u64,
        title: &str,
        media_type: &str,
        creator: &str,
        dur: f64,
        ts: i64,
    ) -> CatalogEntry {
        CatalogEntry::new(id, title, media_type, creator, dur, ts, "desc")
    }

    #[test]
    fn test_search_filter_is_empty_default() {
        assert!(SearchFilter::new().is_empty());
    }

    #[test]
    fn test_search_filter_not_empty_after_keyword() {
        let f = SearchFilter::new().with_keyword("test");
        assert!(!f.is_empty());
    }

    #[test]
    fn test_search_filter_not_empty_after_media_type() {
        let f = SearchFilter::new().with_media_type("video");
        assert!(!f.is_empty());
    }

    #[test]
    fn test_catalog_entry_matches_empty_filter() {
        let e = sample_entry(1, "News", "video", "Alice", 120.0, 1000);
        assert!(e.matches_filter(&SearchFilter::new()));
    }

    #[test]
    fn test_catalog_entry_matches_keyword_in_title() {
        let e = sample_entry(1, "Breaking News", "video", "Alice", 120.0, 1000);
        let f = SearchFilter::new().with_keyword("breaking");
        assert!(e.matches_filter(&f));
    }

    #[test]
    fn test_catalog_entry_no_match_keyword() {
        let e = CatalogEntry::new(1, "Sports Clip", "video", "Bob", 60.0, 500, "football");
        let f = SearchFilter::new().with_keyword("cooking");
        assert!(!e.matches_filter(&f));
    }

    #[test]
    fn test_catalog_entry_matches_media_type() {
        let e = sample_entry(1, "Track", "audio", "DJ", 180.0, 2000);
        let f = SearchFilter::new().with_media_type("audio");
        assert!(e.matches_filter(&f));
    }

    #[test]
    fn test_catalog_entry_no_match_media_type() {
        let e = sample_entry(1, "Track", "audio", "DJ", 180.0, 2000);
        let f = SearchFilter::new().with_media_type("video");
        assert!(!e.matches_filter(&f));
    }

    #[test]
    fn test_catalog_entry_matches_creator() {
        let e = sample_entry(1, "Film", "video", "Alice", 7200.0, 3000);
        let f = SearchFilter::new().with_creator("Alice");
        assert!(e.matches_filter(&f));
    }

    #[test]
    fn test_catalog_entry_duration_filter() {
        let e = sample_entry(1, "Short", "video", "x", 30.0, 100);
        let mut f = SearchFilter::new();
        f.min_duration_secs = Some(60.0);
        assert!(!e.matches_filter(&f));
    }

    #[test]
    fn test_catalog_entry_date_range_filter() {
        let e = sample_entry(1, "Old", "video", "x", 60.0, 100);
        let f = SearchFilter::new().with_date_range(200, 500);
        assert!(!e.matches_filter(&f));
    }

    #[test]
    fn test_searcher_search_all_with_empty_filter() {
        let mut s = CatalogSearcher::new();
        s.add_entry(sample_entry(1, "A", "video", "x", 60.0, 100));
        s.add_entry(sample_entry(2, "B", "audio", "y", 30.0, 200));
        let results = s.search(&SearchFilter::new());
        assert_eq!(results.count(), 2);
    }

    #[test]
    fn test_searcher_search_filtered() {
        let mut s = CatalogSearcher::new();
        s.add_entry(sample_entry(1, "Video A", "video", "x", 60.0, 100));
        s.add_entry(sample_entry(2, "Audio B", "audio", "y", 30.0, 200));
        let results = s.search(&SearchFilter::new().with_media_type("video"));
        assert_eq!(results.count(), 1);
        assert_eq!(results.first().expect("should succeed in test").id, 1);
    }

    #[test]
    fn test_result_set_sort_by_date() {
        let mut s = CatalogSearcher::new();
        s.add_entry(sample_entry(1, "Old", "video", "x", 60.0, 100));
        s.add_entry(sample_entry(2, "New", "video", "x", 60.0, 999));
        let mut results = s.search(&SearchFilter::new());
        results.sort_by_date();
        assert_eq!(results.first().expect("should succeed in test").id, 2);
    }

    #[test]
    fn test_result_set_is_empty() {
        let s = CatalogSearcher::new();
        let results = s.search(&SearchFilter::new().with_keyword("nothing_here"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_searcher_catalog_size() {
        let mut s = CatalogSearcher::new();
        s.add_entries(vec![
            sample_entry(1, "A", "video", "x", 1.0, 1),
            sample_entry(2, "B", "video", "x", 1.0, 2),
            sample_entry(3, "C", "video", "x", 1.0, 3),
        ]);
        assert_eq!(s.catalog_size(), 3);
    }

    #[test]
    fn test_result_set_sort_by_title() {
        let mut s = CatalogSearcher::new();
        s.add_entry(sample_entry(1, "Zebra", "video", "x", 1.0, 1));
        s.add_entry(sample_entry(2, "Apple", "video", "x", 1.0, 2));
        let mut results = s.search(&SearchFilter::new());
        results.sort_by_title();
        assert_eq!(
            results.first().expect("should succeed in test").title,
            "Apple"
        );
    }
}
