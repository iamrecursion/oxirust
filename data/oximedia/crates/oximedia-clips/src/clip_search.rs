//! Clip search engine for querying clips by various fields.
//!
//! Provides `SearchField`, `ClipSearchQuery`, `ClipSearchEngine`, and `ClipSearchResult`.

#![allow(dead_code)]

/// Fields available for searching clips.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchField {
    /// Match against the clip name.
    Name,
    /// Match against the clip description or notes.
    Description,
    /// Match against keyword tags.
    Keywords,
    /// Match against the source file path.
    FilePath,
    /// Match against the clip rating (as a string like "5").
    Rating,
    /// Match against all text fields simultaneously.
    All,
}

impl SearchField {
    /// Human-readable label for this search field.
    pub fn label(&self) -> &'static str {
        match self {
            SearchField::Name => "Name",
            SearchField::Description => "Description",
            SearchField::Keywords => "Keywords",
            SearchField::FilePath => "File Path",
            SearchField::Rating => "Rating",
            SearchField::All => "All Fields",
        }
    }

    /// Returns `true` if this field targets metadata rather than content.
    pub fn is_metadata_field(&self) -> bool {
        matches!(self, SearchField::Rating | SearchField::FilePath)
    }
}

/// A single search criterion: a field combined with a query string.
#[derive(Debug, Clone)]
pub struct SearchCriterion {
    /// The field to search in.
    pub field: SearchField,
    /// The text to look for.
    pub value: String,
    /// Whether the match must be case-sensitive.
    pub case_sensitive: bool,
}

impl SearchCriterion {
    /// Create a new case-insensitive criterion.
    pub fn new(field: SearchField, value: impl Into<String>) -> Self {
        Self {
            field,
            value: value.into(),
            case_sensitive: false,
        }
    }

    /// Make this criterion case-sensitive.
    pub fn case_sensitive(mut self) -> Self {
        self.case_sensitive = true;
        self
    }
}

/// A composable query that aggregates multiple `SearchCriterion`s.
#[derive(Debug, Default, Clone)]
pub struct ClipSearchQuery {
    /// All criteria that must be satisfied (AND semantics).
    pub criteria: Vec<SearchCriterion>,
    /// Maximum number of results to return (0 = unlimited).
    pub limit: usize,
    /// Field to sort results by.
    pub sort_by: Option<SearchField>,
}

impl ClipSearchQuery {
    /// Create an empty query.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a criterion to the query.
    pub fn add(mut self, criterion: SearchCriterion) -> Self {
        self.criteria.push(criterion);
        self
    }

    /// Set the result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the sort field.
    pub fn sort_by(mut self, field: SearchField) -> Self {
        self.sort_by = Some(field);
        self
    }

    /// Returns `true` when at least one criterion has been added.
    pub fn has_criteria(&self) -> bool {
        !self.criteria.is_empty()
    }
}

/// A single clip record used by the search engine.
#[derive(Debug, Clone)]
pub struct ClipRecord {
    /// Identifier for the clip.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Optional description or notes text.
    pub description: String,
    /// Keywords / tags.
    pub keywords: Vec<String>,
    /// Source file path as a string.
    pub file_path: String,
    /// Rating as a numeric string (e.g. "4").
    pub rating: String,
}

impl ClipRecord {
    /// Create a minimal clip record.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            keywords: Vec::new(),
            file_path: String::new(),
            rating: "0".to_string(),
        }
    }

    /// Add a keyword tag.
    pub fn with_keyword(mut self, kw: impl Into<String>) -> Self {
        self.keywords.push(kw.into());
        self
    }

    /// Set description text.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set file path.
    pub fn with_file_path(mut self, path: impl Into<String>) -> Self {
        self.file_path = path.into();
        self
    }

    /// Set rating string.
    pub fn with_rating(mut self, rating: impl Into<String>) -> Self {
        self.rating = rating.into();
        self
    }
}

/// Search result container.
#[derive(Debug, Clone, Default)]
pub struct ClipSearchResult {
    /// Clips that matched the query.
    pub matches: Vec<ClipRecord>,
}

impl ClipSearchResult {
    /// Number of clips that matched.
    pub fn match_count(&self) -> usize {
        self.matches.len()
    }

    /// Returns `true` if no clips matched.
    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }
}

/// In-memory clip search engine.
#[derive(Debug, Default)]
pub struct ClipSearchEngine {
    records: Vec<ClipRecord>,
}

impl ClipSearchEngine {
    /// Create a new engine with no records.
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a clip record for future searches.
    pub fn index(&mut self, record: ClipRecord) {
        self.records.push(record);
    }

    /// Run a search query against all indexed records.
    pub fn search(&self, query: &ClipSearchQuery) -> ClipSearchResult {
        if !query.has_criteria() {
            return ClipSearchResult {
                matches: self.records.clone(),
            };
        }

        let mut matches: Vec<ClipRecord> = self
            .records
            .iter()
            .filter(|r| self.record_matches(r, query))
            .cloned()
            .collect();

        if let Some(ref sort_field) = query.sort_by {
            match sort_field {
                SearchField::Name => matches.sort_by(|a, b| a.name.cmp(&b.name)),
                SearchField::Rating => matches.sort_by(|a, b| a.rating.cmp(&b.rating)),
                _ => {}
            }
        }

        if query.limit > 0 {
            matches.truncate(query.limit);
        }

        ClipSearchResult { matches }
    }

    fn record_matches(&self, record: &ClipRecord, query: &ClipSearchQuery) -> bool {
        query
            .criteria
            .iter()
            .all(|c| self.criterion_matches(record, c))
    }

    fn criterion_matches(&self, record: &ClipRecord, criterion: &SearchCriterion) -> bool {
        let needle = if criterion.case_sensitive {
            criterion.value.clone()
        } else {
            criterion.value.to_lowercase()
        };

        let haystack_contains = |s: &str| -> bool {
            if criterion.case_sensitive {
                s.contains(&needle)
            } else {
                s.to_lowercase().contains(&needle)
            }
        };

        match &criterion.field {
            SearchField::Name => haystack_contains(&record.name),
            SearchField::Description => haystack_contains(&record.description),
            SearchField::Keywords => record.keywords.iter().any(|k| haystack_contains(k)),
            SearchField::FilePath => haystack_contains(&record.file_path),
            SearchField::Rating => haystack_contains(&record.rating),
            SearchField::All => {
                haystack_contains(&record.name)
                    || haystack_contains(&record.description)
                    || record.keywords.iter().any(|k| haystack_contains(k))
                    || haystack_contains(&record.file_path)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_engine() -> ClipSearchEngine {
        let mut engine = ClipSearchEngine::new();
        engine.index(
            ClipRecord::new("c1", "Interview Alpha")
                .with_description("Office interview")
                .with_keyword("interview")
                .with_keyword("indoor")
                .with_rating("5"),
        );
        engine.index(
            ClipRecord::new("c2", "Outdoor Landscape")
                .with_description("Mountain sunset")
                .with_keyword("outdoor")
                .with_keyword("nature")
                .with_rating("4")
                .with_file_path("/footage/landscape.mov"),
        );
        engine.index(
            ClipRecord::new("c3", "Interview Beta")
                .with_keyword("interview")
                .with_rating("3"),
        );
        engine
    }

    #[test]
    fn search_field_label_name() {
        assert_eq!(SearchField::Name.label(), "Name");
    }

    #[test]
    fn search_field_label_all() {
        assert_eq!(SearchField::All.label(), "All Fields");
    }

    #[test]
    fn search_field_is_metadata() {
        assert!(SearchField::Rating.is_metadata_field());
        assert!(!SearchField::Name.is_metadata_field());
    }

    #[test]
    fn query_has_criteria_empty() {
        let q = ClipSearchQuery::new();
        assert!(!q.has_criteria());
    }

    #[test]
    fn query_has_criteria_with_one() {
        let q = ClipSearchQuery::new().add(SearchCriterion::new(SearchField::Name, "test"));
        assert!(q.has_criteria());
    }

    #[test]
    fn search_by_name_finds_matches() {
        let engine = sample_engine();
        let q = ClipSearchQuery::new().add(SearchCriterion::new(SearchField::Name, "Interview"));
        let result = engine.search(&q);
        assert_eq!(result.match_count(), 2);
    }

    #[test]
    fn search_by_keyword_finds_matches() {
        let engine = sample_engine();
        let q = ClipSearchQuery::new().add(SearchCriterion::new(SearchField::Keywords, "outdoor"));
        let result = engine.search(&q);
        assert_eq!(result.match_count(), 1);
        assert_eq!(result.matches[0].id, "c2");
    }

    #[test]
    fn search_by_description() {
        let engine = sample_engine();
        let q =
            ClipSearchQuery::new().add(SearchCriterion::new(SearchField::Description, "sunset"));
        let result = engine.search(&q);
        assert_eq!(result.match_count(), 1);
    }

    #[test]
    fn search_by_file_path() {
        let engine = sample_engine();
        let q =
            ClipSearchQuery::new().add(SearchCriterion::new(SearchField::FilePath, "landscape"));
        let result = engine.search(&q);
        assert_eq!(result.match_count(), 1);
    }

    #[test]
    fn search_no_match_returns_empty() {
        let engine = sample_engine();
        let q = ClipSearchQuery::new().add(SearchCriterion::new(SearchField::Name, "zzznomatch"));
        let result = engine.search(&q);
        assert!(result.is_empty());
    }

    #[test]
    fn search_with_limit() {
        let engine = sample_engine();
        let q = ClipSearchQuery::new()
            .add(SearchCriterion::new(SearchField::All, "i"))
            .with_limit(1);
        let result = engine.search(&q);
        assert_eq!(result.match_count(), 1);
    }

    #[test]
    fn search_empty_query_returns_all() {
        let engine = sample_engine();
        let q = ClipSearchQuery::new();
        let result = engine.search(&q);
        assert_eq!(result.match_count(), 3);
    }

    #[test]
    fn search_case_insensitive_by_default() {
        let engine = sample_engine();
        let q = ClipSearchQuery::new().add(SearchCriterion::new(SearchField::Name, "INTERVIEW"));
        let result = engine.search(&q);
        assert_eq!(result.match_count(), 2);
    }

    #[test]
    fn search_case_sensitive_no_match() {
        let engine = sample_engine();
        let q = ClipSearchQuery::new()
            .add(SearchCriterion::new(SearchField::Name, "INTERVIEW").case_sensitive());
        let result = engine.search(&q);
        assert!(result.is_empty());
    }

    #[test]
    fn search_sorted_by_name() {
        let engine = sample_engine();
        let q = ClipSearchQuery::new()
            .add(SearchCriterion::new(SearchField::Keywords, "interview"))
            .sort_by(SearchField::Name);
        let result = engine.search(&q);
        assert_eq!(result.match_count(), 2);
        assert_eq!(result.matches[0].name, "Interview Alpha");
        assert_eq!(result.matches[1].name, "Interview Beta");
    }
}
