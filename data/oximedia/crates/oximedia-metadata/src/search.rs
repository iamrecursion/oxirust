//! Metadata search and indexing.
//!
//! Provides an in-memory index of metadata entries and query facilities
//! for full-text and field-based searching.

use std::collections::HashMap;

/// A single indexed metadata entry.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Path to the media file this metadata belongs to.
    pub path: String,
    /// Metadata fields stored as key-value strings.
    pub fields: HashMap<String, String>,
    /// Unix timestamp (seconds) when this entry was indexed.
    pub indexed_at: u64,
}

impl IndexEntry {
    /// Return whether any field value contains `query` (case-insensitive).
    fn matches_fulltext(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        // Also search the path
        if self.path.to_lowercase().contains(&q) {
            return true;
        }
        self.fields.values().any(|v| v.to_lowercase().contains(&q))
    }

    /// Return whether the field `key` has a value containing `value` (case-insensitive).
    fn matches_field(&self, key: &str, value: &str) -> bool {
        self.fields
            .get(key)
            .is_some_and(|v| v.to_lowercase().contains(&value.to_lowercase()))
    }
}

/// An in-memory index of metadata entries indexed by file path.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct MetadataIndex {
    entries: Vec<IndexEntry>,
}

impl MetadataIndex {
    /// Create a new, empty index.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add or update a metadata entry for the given path.
    /// If an entry for `path` already exists it is replaced.
    #[allow(dead_code)]
    pub fn add(&mut self, path: &str, fields: HashMap<String, String>) {
        // Replace existing entry if path matches
        if let Some(existing) = self.entries.iter_mut().find(|e| e.path == path) {
            existing.fields = fields;
            existing.indexed_at = current_timestamp();
            return;
        }
        self.entries.push(IndexEntry {
            path: path.to_string(),
            fields,
            indexed_at: current_timestamp(),
        });
    }

    /// Search for entries where field `key` contains `value`.
    #[allow(dead_code)]
    pub fn search_by_field<'a>(&'a self, key: &str, value: &str) -> Vec<&'a IndexEntry> {
        self.entries
            .iter()
            .filter(|e| e.matches_field(key, value))
            .collect()
    }

    /// Full-text search across all field values and paths.
    #[allow(dead_code)]
    pub fn search_fulltext<'a>(&'a self, query: &str) -> Vec<&'a IndexEntry> {
        if query.is_empty() {
            return self.entries.iter().collect();
        }
        self.entries
            .iter()
            .filter(|e| e.matches_fulltext(query))
            .collect()
    }

    /// Remove the entry for the given path.
    /// Returns `true` if an entry was found and removed.
    #[allow(dead_code)]
    pub fn remove(&mut self, path: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.path != path);
        self.entries.len() < before
    }

    /// Return the number of indexed entries.
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over all entries.
    #[allow(dead_code)]
    pub fn entries(&self) -> &[IndexEntry] {
        &self.entries
    }
}

/// A structured search query.
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct SearchQuery {
    /// Free-text terms; ALL must appear somewhere in the entry.
    pub terms: Vec<String>,
    /// Field key-value pairs that MUST all be present (substring match).
    pub must_have: Vec<(String, String)>,
    /// Paths that should be excluded from results.
    pub exclude_paths: Vec<String>,
}

impl SearchQuery {
    /// Create a new empty query.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a free-text term that must appear in results.
    #[allow(dead_code)]
    pub fn with_term(mut self, term: &str) -> Self {
        self.terms.push(term.to_string());
        self
    }

    /// Add a required field key-value constraint.
    #[allow(dead_code)]
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.must_have.push((key.to_string(), value.to_string()));
        self
    }

    /// Add a path to exclude from results.
    #[allow(dead_code)]
    pub fn exclude(mut self, path: &str) -> Self {
        self.exclude_paths.push(path.to_string());
        self
    }
}

/// Execute a `SearchQuery` against a `MetadataIndex` and return matching entries.
#[allow(dead_code)]
pub fn execute_query<'a>(index: &'a MetadataIndex, query: &SearchQuery) -> Vec<&'a IndexEntry> {
    index
        .entries()
        .iter()
        .filter(|entry| {
            // Exclude paths
            if query.exclude_paths.iter().any(|p| p == &entry.path) {
                return false;
            }
            // All free-text terms must match
            if !query.terms.iter().all(|t| entry.matches_fulltext(t)) {
                return false;
            }
            // All must_have field constraints must match
            if !query
                .must_have
                .iter()
                .all(|(k, v)| entry.matches_field(k, v))
            {
                return false;
            }
            true
        })
        .collect()
}

/// Return the current Unix timestamp in seconds.
/// Falls back to zero on platforms where the system clock is unavailable.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn build_index() -> MetadataIndex {
        let mut idx = MetadataIndex::new();
        idx.add(
            "/media/clip1.mp4",
            fields(&[("title", "Summer Festival"), ("genre", "Documentary")]),
        );
        idx.add(
            "/media/clip2.mp4",
            fields(&[("title", "Winter Wonderland"), ("genre", "Nature")]),
        );
        idx.add(
            "/media/clip3.mp4",
            fields(&[("title", "City Life"), ("genre", "Documentary")]),
        );
        idx
    }

    #[test]
    fn test_index_count() {
        let idx = build_index();
        assert_eq!(idx.count(), 3);
    }

    #[test]
    fn test_add_replaces_existing() {
        let mut idx = MetadataIndex::new();
        idx.add("/media/a.mp4", fields(&[("title", "Old")]));
        idx.add("/media/a.mp4", fields(&[("title", "New")]));
        assert_eq!(idx.count(), 1);
        let results = idx.search_by_field("title", "New");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_remove_existing() {
        let mut idx = build_index();
        let removed = idx.remove("/media/clip1.mp4");
        assert!(removed);
        assert_eq!(idx.count(), 2);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut idx = build_index();
        let removed = idx.remove("/media/does_not_exist.mp4");
        assert!(!removed);
        assert_eq!(idx.count(), 3);
    }

    #[test]
    fn test_search_by_field_found() {
        let idx = build_index();
        let results = idx.search_by_field("genre", "Documentary");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_field_not_found() {
        let idx = build_index();
        let results = idx.search_by_field("genre", "Horror");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_by_field_case_insensitive() {
        let idx = build_index();
        let results = idx.search_by_field("genre", "documentary");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_fulltext_found() {
        let idx = build_index();
        let results = idx.search_fulltext("Winter");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "/media/clip2.mp4");
    }

    #[test]
    fn test_search_fulltext_empty_query_returns_all() {
        let idx = build_index();
        let results = idx.search_fulltext("");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_fulltext_no_match() {
        let idx = build_index();
        let results = idx.search_fulltext("zzznonexistent");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_execute_query_term_filter() {
        let idx = build_index();
        let query = SearchQuery::new().with_term("City");
        let results = execute_query(&idx, &query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "/media/clip3.mp4");
    }

    #[test]
    fn test_execute_query_must_have_filter() {
        let idx = build_index();
        let query = SearchQuery::new().with_field("genre", "Nature");
        let results = execute_query(&idx, &query);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_execute_query_exclude_path() {
        let idx = build_index();
        let query = SearchQuery::new()
            .with_field("genre", "Documentary")
            .exclude("/media/clip1.mp4");
        let results = execute_query(&idx, &query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "/media/clip3.mp4");
    }

    #[test]
    fn test_execute_query_multiple_terms() {
        let idx = build_index();
        // "City" and "Documentary" must both match
        let query = SearchQuery::new()
            .with_term("City")
            .with_field("genre", "Documentary");
        let results = execute_query(&idx, &query);
        assert_eq!(results.len(), 1);
    }
}
