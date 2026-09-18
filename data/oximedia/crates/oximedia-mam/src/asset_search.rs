//! Asset search engine for MAM.
//!
//! Provides a structured query model with multiple search fields and an
//! in-memory `AssetSearchEngine` that filters a collection of asset
//! descriptors.

#![allow(dead_code)]

use std::collections::HashMap;

/// Fields available for filtering assets.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SearchField {
    /// Asset title / name.
    Title,
    /// Creator or author.
    Creator,
    /// Media type / MIME type string.
    MediaType,
    /// Comma-separated list of tags.
    Tags,
    /// Project or production name.
    Project,
    /// Status label (e.g. `"approved"`, `"draft"`).
    Status,
    /// File extension (without leading dot).
    Extension,
    /// An arbitrary custom metadata key.
    Custom(String),
}

/// A single filter criterion.
#[derive(Clone, Debug)]
pub struct FilterClause {
    /// The field to match against.
    pub field: SearchField,
    /// Substring to search for (case-insensitive).
    pub value: String,
    /// When `true`, the value must NOT be present.
    pub negate: bool,
}

impl FilterClause {
    /// Positive (inclusive) filter clause.
    #[must_use]
    pub fn include(field: SearchField, value: impl Into<String>) -> Self {
        Self {
            field,
            value: value.into(),
            negate: false,
        }
    }

    /// Negative (exclusive) filter clause.
    #[must_use]
    pub fn exclude(field: SearchField, value: impl Into<String>) -> Self {
        Self {
            field,
            value: value.into(),
            negate: true,
        }
    }
}

/// A structured asset search query.
#[derive(Clone, Debug, Default)]
pub struct SearchQuery {
    /// Clauses that are ANDed together.
    pub clauses: Vec<FilterClause>,
    /// Maximum number of results to return (0 = unlimited).
    pub limit: usize,
    /// Number of results to skip before returning.
    pub offset: usize,
}

impl SearchQuery {
    /// Create an empty query.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an include clause for a field/value pair.
    pub fn with(mut self, field: SearchField, value: impl Into<String>) -> Self {
        self.clauses.push(FilterClause::include(field, value));
        self
    }

    /// Add an exclude clause for a field/value pair.
    pub fn without(mut self, field: SearchField, value: impl Into<String>) -> Self {
        self.clauses.push(FilterClause::exclude(field, value));
        self
    }

    /// Set the maximum number of results.
    #[must_use]
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// Set the result offset.
    #[must_use]
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }
}

/// A lightweight descriptor for an asset stored in the engine.
#[derive(Clone, Debug)]
pub struct AssetDescriptor {
    /// Unique asset ID.
    pub id: u64,
    /// Asset title.
    pub title: String,
    /// Creator name.
    pub creator: String,
    /// Media type string (e.g. `"video/mp4"`).
    pub media_type: String,
    /// Tags associated with this asset.
    pub tags: Vec<String>,
    /// Project name.
    pub project: String,
    /// Workflow status.
    pub status: String,
    /// File extension.
    pub extension: String,
    /// Custom metadata key-value pairs.
    pub custom: HashMap<String, String>,
}

impl AssetDescriptor {
    /// Create a minimal asset descriptor.
    #[must_use]
    pub fn new(id: u64, title: impl Into<String>) -> Self {
        Self {
            id,
            title: title.into(),
            creator: String::new(),
            media_type: String::new(),
            tags: Vec::new(),
            project: String::new(),
            status: String::new(),
            extension: String::new(),
            custom: HashMap::new(),
        }
    }

    /// Retrieve the string value for a given [`SearchField`].
    #[must_use]
    fn field_value(&self, field: &SearchField) -> String {
        match field {
            SearchField::Title => self.title.clone(),
            SearchField::Creator => self.creator.clone(),
            SearchField::MediaType => self.media_type.clone(),
            SearchField::Tags => self.tags.join(","),
            SearchField::Project => self.project.clone(),
            SearchField::Status => self.status.clone(),
            SearchField::Extension => self.extension.clone(),
            SearchField::Custom(key) => self.custom.get(key).cloned().unwrap_or_default(),
        }
    }

    /// Return `true` if all clauses match this asset.
    #[must_use]
    fn matches(&self, query: &SearchQuery) -> bool {
        for clause in &query.clauses {
            let field_val = self.field_value(&clause.field).to_ascii_lowercase();
            let needle = clause.value.to_ascii_lowercase();
            let found = field_val.contains(&needle);
            if clause.negate == found {
                // negate=true & found=true  => exclusion failed
                // negate=false & found=false => inclusion failed
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Sort order for paginated results
// ---------------------------------------------------------------------------

/// Field used to sort results in paginated queries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SortField {
    /// Sort by asset ID (default).
    Id,
    /// Sort by title (lexicographic, case-insensitive).
    Title,
    /// Sort by creator name (lexicographic, case-insensitive).
    Creator,
    /// Sort by status label.
    Status,
}

/// Sort direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortDirection {
    /// Ascending (A-Z, 0-9).
    Ascending,
    /// Descending (Z-A, 9-0).
    Descending,
}

/// Sort specification used by [`SearchQuery`].
#[derive(Clone, Debug)]
pub struct SortSpec {
    /// The field to sort on.
    pub field: SortField,
    /// Sort direction.
    pub direction: SortDirection,
}

impl Default for SortSpec {
    fn default() -> Self {
        Self {
            field: SortField::Id,
            direction: SortDirection::Ascending,
        }
    }
}

// ---------------------------------------------------------------------------
// Cursor-based pagination
// ---------------------------------------------------------------------------

/// An opaque cursor representing a position in the result set.
///
/// Internally it stores the asset ID of the last item in the previous page.
/// The cursor is encoded as a simple string for ease of transport over APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cursor(String);

impl Cursor {
    /// Create a cursor from a raw string value.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Return the encoded cursor string.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.0
    }

    /// Encode a `u64` asset ID into a cursor.
    #[must_use]
    pub fn from_id(id: u64) -> Self {
        Self(id.to_string())
    }

    /// Decode the cursor back to an asset ID. Returns `None` on invalid data.
    #[must_use]
    pub fn to_id(&self) -> Option<u64> {
        self.0.parse().ok()
    }
}

impl std::fmt::Display for Cursor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A page of results returned by [`AssetSearchEngine::paginate`].
#[derive(Clone, Debug)]
pub struct SearchPage<'a> {
    /// The items in this page.
    pub items: Vec<&'a AssetDescriptor>,
    /// Total number of items matching the query (before pagination).
    pub total_count: usize,
    /// Whether there is a next page.
    pub has_next_page: bool,
    /// Whether there is a previous page.
    pub has_previous_page: bool,
    /// Cursor pointing to the first item of this page (for backward navigation).
    pub start_cursor: Option<Cursor>,
    /// Cursor pointing to the last item of this page (pass as `after` for next page).
    pub end_cursor: Option<Cursor>,
}

/// In-memory asset search engine.
///
/// # Example
/// ```
/// use oximedia_mam::asset_search::{AssetDescriptor, AssetSearchEngine, SearchQuery, SearchField};
///
/// let mut engine = AssetSearchEngine::new();
/// let mut asset = AssetDescriptor::new(1, "Holiday Reel");
/// asset.tags = vec!["holiday".to_string(), "promo".to_string()];
/// engine.add(asset);
///
/// let results = engine.filter(
///     &SearchQuery::new().with(SearchField::Tags, "holiday")
/// );
/// assert_eq!(results.len(), 1);
/// ```
#[derive(Default)]
pub struct AssetSearchEngine {
    assets: Vec<AssetDescriptor>,
}

impl AssetSearchEngine {
    /// Create an empty engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an asset to the engine.
    pub fn add(&mut self, asset: AssetDescriptor) {
        self.assets.push(asset);
    }

    /// Remove asset with the given ID.  Returns `true` if found.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.assets.len();
        self.assets.retain(|a| a.id != id);
        self.assets.len() < before
    }

    /// Return the number of indexed assets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Return `true` when no assets are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    /// Execute a [`SearchQuery`] and return matching asset references.
    ///
    /// Applies `offset` and `limit` (0 = no limit) to the result slice.
    #[must_use]
    pub fn filter<'a>(&'a self, query: &SearchQuery) -> Vec<&'a AssetDescriptor> {
        let matched: Vec<&AssetDescriptor> =
            self.assets.iter().filter(|a| a.matches(query)).collect();

        let start = query.offset.min(matched.len());
        let slice = &matched[start..];
        if query.limit == 0 {
            slice.to_vec()
        } else {
            slice.iter().copied().take(query.limit).collect()
        }
    }

    /// Look up an asset by exact ID.
    #[must_use]
    pub fn find_by_id(&self, id: u64) -> Option<&AssetDescriptor> {
        self.assets.iter().find(|a| a.id == id)
    }

    // -----------------------------------------------------------------------
    // Sorted filter
    // -----------------------------------------------------------------------

    /// Execute a query with explicit sort order, returning matching asset
    /// references sorted accordingly. Applies `offset` and `limit` after
    /// sorting.
    #[must_use]
    pub fn filter_sorted<'a>(
        &'a self,
        query: &SearchQuery,
        sort: &SortSpec,
    ) -> Vec<&'a AssetDescriptor> {
        let mut matched: Vec<&AssetDescriptor> =
            self.assets.iter().filter(|a| a.matches(query)).collect();

        Self::sort_assets(&mut matched, sort);

        let start = query.offset.min(matched.len());
        let slice = &matched[start..];
        if query.limit == 0 {
            slice.to_vec()
        } else {
            slice.iter().copied().take(query.limit).collect()
        }
    }

    // -----------------------------------------------------------------------
    // Cursor-based pagination
    // -----------------------------------------------------------------------

    /// Execute a query with cursor-based pagination.
    ///
    /// * `query`  -- filter clauses (offset/limit fields on query are ignored;
    ///   use `first` and `after` instead).
    /// * `sort`   -- sort order applied before pagination.
    /// * `first`  -- page size (max items to return).
    /// * `after`  -- cursor returned by a previous call (`end_cursor`). Pass
    ///   `None` for the first page.
    ///
    /// Returns a [`SearchPage`] with the items, cursors for forward/backward
    /// navigation, and total count.
    #[must_use]
    pub fn paginate<'a>(
        &'a self,
        query: &SearchQuery,
        sort: &SortSpec,
        first: usize,
        after: Option<&Cursor>,
    ) -> SearchPage<'a> {
        let mut matched: Vec<&AssetDescriptor> =
            self.assets.iter().filter(|a| a.matches(query)).collect();

        Self::sort_assets(&mut matched, sort);
        let total_count = matched.len();

        // Determine the starting index based on the cursor.
        let start_index = if let Some(cursor) = after {
            if let Some(after_id) = cursor.to_id() {
                // Find the position of the cursor asset and start after it.
                matched
                    .iter()
                    .position(|a| a.id == after_id)
                    .map(|pos| pos + 1)
                    .unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        let has_previous_page = start_index > 0;
        let remaining = if start_index < matched.len() {
            &matched[start_index..]
        } else {
            &[]
        };

        let page_items: Vec<&AssetDescriptor> = remaining.iter().copied().take(first).collect();
        let has_next_page = start_index + page_items.len() < total_count;

        let start_cursor = page_items.first().map(|a| Cursor::from_id(a.id));
        let end_cursor = page_items.last().map(|a| Cursor::from_id(a.id));

        SearchPage {
            items: page_items,
            total_count,
            has_next_page,
            has_previous_page,
            start_cursor,
            end_cursor,
        }
    }

    /// Paginate backwards: return the page *before* the given cursor.
    ///
    /// * `last`   -- page size.
    /// * `before` -- cursor of the first item on the current page
    ///   (`start_cursor`). Pass `None` to get the last page.
    #[must_use]
    pub fn paginate_backward<'a>(
        &'a self,
        query: &SearchQuery,
        sort: &SortSpec,
        last: usize,
        before: Option<&Cursor>,
    ) -> SearchPage<'a> {
        let mut matched: Vec<&AssetDescriptor> =
            self.assets.iter().filter(|a| a.matches(query)).collect();

        Self::sort_assets(&mut matched, sort);
        let total_count = matched.len();

        let end_index = if let Some(cursor) = before {
            if let Some(before_id) = cursor.to_id() {
                matched
                    .iter()
                    .position(|a| a.id == before_id)
                    .unwrap_or(total_count)
            } else {
                total_count
            }
        } else {
            total_count
        };

        let start_index = end_index.saturating_sub(last);
        let page_items: Vec<&AssetDescriptor> = matched[start_index..end_index].to_vec();

        let has_previous_page = start_index > 0;
        let has_next_page = end_index < total_count;

        let start_cursor = page_items.first().map(|a| Cursor::from_id(a.id));
        let end_cursor = page_items.last().map(|a| Cursor::from_id(a.id));

        SearchPage {
            items: page_items,
            total_count,
            has_next_page,
            has_previous_page,
            start_cursor,
            end_cursor,
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Sort a mutable slice of asset references according to the given spec.
    fn sort_assets(assets: &mut [&AssetDescriptor], sort: &SortSpec) {
        assets.sort_by(|a, b| {
            let cmp = match sort.field {
                SortField::Id => a.id.cmp(&b.id),
                SortField::Title => a
                    .title
                    .to_ascii_lowercase()
                    .cmp(&b.title.to_ascii_lowercase()),
                SortField::Creator => a
                    .creator
                    .to_ascii_lowercase()
                    .cmp(&b.creator.to_ascii_lowercase()),
                SortField::Status => a
                    .status
                    .to_ascii_lowercase()
                    .cmp(&b.status.to_ascii_lowercase()),
            };
            match sort.direction {
                SortDirection::Ascending => cmp,
                SortDirection::Descending => cmp.reverse(),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> AssetSearchEngine {
        let mut engine = AssetSearchEngine::new();

        let mut a1 = AssetDescriptor::new(1, "Summer Campaign");
        a1.creator = "Alice".to_string();
        a1.media_type = "video/mp4".to_string();
        a1.tags = vec!["summer".to_string(), "promo".to_string()];
        a1.project = "Alpha".to_string();
        a1.status = "approved".to_string();
        a1.extension = "mp4".to_string();

        let mut a2 = AssetDescriptor::new(2, "Winter Promo");
        a2.creator = "Bob".to_string();
        a2.media_type = "video/mov".to_string();
        a2.tags = vec!["winter".to_string(), "promo".to_string()];
        a2.project = "Beta".to_string();
        a2.status = "draft".to_string();
        a2.extension = "mov".to_string();

        let mut a3 = AssetDescriptor::new(3, "Spring Shoot");
        a3.creator = "Alice".to_string();
        a3.media_type = "image/tiff".to_string();
        a3.tags = vec!["spring".to_string()];
        a3.project = "Alpha".to_string();
        a3.status = "approved".to_string();
        a3.extension = "tiff".to_string();

        engine.add(a1);
        engine.add(a2);
        engine.add(a3);
        engine
    }

    #[test]
    fn test_engine_len() {
        let engine = make_engine();
        assert_eq!(engine.len(), 3);
    }

    #[test]
    fn test_filter_by_title() {
        let engine = make_engine();
        let q = SearchQuery::new().with(SearchField::Title, "promo");
        assert_eq!(engine.filter(&q).len(), 1);
    }

    #[test]
    fn test_filter_by_creator() {
        let engine = make_engine();
        let q = SearchQuery::new().with(SearchField::Creator, "alice");
        assert_eq!(engine.filter(&q).len(), 2);
    }

    #[test]
    fn test_filter_by_tag() {
        let engine = make_engine();
        let q = SearchQuery::new().with(SearchField::Tags, "promo");
        assert_eq!(engine.filter(&q).len(), 2);
    }

    #[test]
    fn test_filter_negate() {
        let engine = make_engine();
        let q = SearchQuery::new().without(SearchField::Status, "draft");
        assert_eq!(engine.filter(&q).len(), 2);
    }

    #[test]
    fn test_filter_combined_clauses() {
        let engine = make_engine();
        let q = SearchQuery::new()
            .with(SearchField::Creator, "alice")
            .with(SearchField::Status, "approved");
        assert_eq!(engine.filter(&q).len(), 2);
    }

    #[test]
    fn test_filter_limit() {
        let engine = make_engine();
        let q = SearchQuery::new().limit(1);
        assert_eq!(engine.filter(&q).len(), 1);
    }

    #[test]
    fn test_filter_offset() {
        let engine = make_engine();
        let q = SearchQuery::new().offset(2);
        assert_eq!(engine.filter(&q).len(), 1);
    }

    #[test]
    fn test_filter_no_match() {
        let engine = make_engine();
        let q = SearchQuery::new().with(SearchField::Project, "Gamma");
        assert!(engine.filter(&q).is_empty());
    }

    #[test]
    fn test_filter_empty_query_returns_all() {
        let engine = make_engine();
        assert_eq!(engine.filter(&SearchQuery::new()).len(), 3);
    }

    #[test]
    fn test_find_by_id() {
        let engine = make_engine();
        assert_eq!(
            engine.find_by_id(2).expect("should succeed in test").title,
            "Winter Promo"
        );
    }

    #[test]
    fn test_find_by_id_missing() {
        let engine = make_engine();
        assert!(engine.find_by_id(99).is_none());
    }

    #[test]
    fn test_remove_asset() {
        let mut engine = make_engine();
        assert!(engine.remove(1));
        assert_eq!(engine.len(), 2);
        assert!(engine.find_by_id(1).is_none());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut engine = make_engine();
        assert!(!engine.remove(999));
    }

    #[test]
    fn test_custom_field_search() {
        let mut engine = AssetSearchEngine::new();
        let mut a = AssetDescriptor::new(10, "Branded Content");
        a.custom
            .insert("client".to_string(), "Acme Corp".to_string());
        engine.add(a);
        let q = SearchQuery::new().with(SearchField::Custom("client".to_string()), "acme");
        assert_eq!(engine.filter(&q).len(), 1);
    }

    #[test]
    fn test_filter_extension() {
        let engine = make_engine();
        let q = SearchQuery::new().with(SearchField::Extension, "tiff");
        assert_eq!(engine.filter(&q).len(), 1);
    }

    // -----------------------------------------------------------------------
    // Cursor & pagination tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cursor_from_id_round_trip() {
        let cursor = Cursor::from_id(42);
        assert_eq!(cursor.to_id(), Some(42));
        assert_eq!(cursor.value(), "42");
        assert_eq!(cursor.to_string(), "42");
    }

    #[test]
    fn test_cursor_invalid_returns_none() {
        let cursor = Cursor::new("not-a-number");
        assert!(cursor.to_id().is_none());
    }

    #[test]
    fn test_sort_spec_default() {
        let sort = SortSpec::default();
        assert_eq!(sort.field, SortField::Id);
        assert_eq!(sort.direction, SortDirection::Ascending);
    }

    #[test]
    fn test_filter_sorted_by_title_asc() {
        let engine = make_engine();
        let sort = SortSpec {
            field: SortField::Title,
            direction: SortDirection::Ascending,
        };
        let results = engine.filter_sorted(&SearchQuery::new(), &sort);
        assert_eq!(results.len(), 3);
        // Spring Shoot < Summer Campaign < Winter Promo
        assert_eq!(results[0].title, "Spring Shoot");
        assert_eq!(results[1].title, "Summer Campaign");
        assert_eq!(results[2].title, "Winter Promo");
    }

    #[test]
    fn test_filter_sorted_by_title_desc() {
        let engine = make_engine();
        let sort = SortSpec {
            field: SortField::Title,
            direction: SortDirection::Descending,
        };
        let results = engine.filter_sorted(&SearchQuery::new(), &sort);
        assert_eq!(results[0].title, "Winter Promo");
        assert_eq!(results[2].title, "Spring Shoot");
    }

    #[test]
    fn test_filter_sorted_with_limit() {
        let engine = make_engine();
        let sort = SortSpec {
            field: SortField::Id,
            direction: SortDirection::Ascending,
        };
        let q = SearchQuery::new().limit(2);
        let results = engine.filter_sorted(&q, &sort);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 2);
    }

    #[test]
    fn test_paginate_first_page() {
        let engine = make_engine();
        let sort = SortSpec::default();
        let page = engine.paginate(&SearchQuery::new(), &sort, 2, None);
        assert_eq!(page.items.len(), 2);
        assert_eq!(page.total_count, 3);
        assert!(page.has_next_page);
        assert!(!page.has_previous_page);
        assert!(page.start_cursor.is_some());
        assert!(page.end_cursor.is_some());
    }

    #[test]
    fn test_paginate_second_page() {
        let engine = make_engine();
        let sort = SortSpec::default();
        let page1 = engine.paginate(&SearchQuery::new(), &sort, 2, None);
        let cursor = page1.end_cursor.as_ref().expect("should have end cursor");
        let page2 = engine.paginate(&SearchQuery::new(), &sort, 2, Some(cursor));
        assert_eq!(page2.items.len(), 1);
        assert!(!page2.has_next_page);
        assert!(page2.has_previous_page);
    }

    #[test]
    fn test_paginate_empty_result() {
        let engine = make_engine();
        let sort = SortSpec::default();
        let q = SearchQuery::new().with(SearchField::Project, "Nonexistent");
        let page = engine.paginate(&q, &sort, 10, None);
        assert!(page.items.is_empty());
        assert_eq!(page.total_count, 0);
        assert!(!page.has_next_page);
        assert!(!page.has_previous_page);
        assert!(page.start_cursor.is_none());
        assert!(page.end_cursor.is_none());
    }

    #[test]
    fn test_paginate_exact_page_size() {
        let engine = make_engine();
        let sort = SortSpec::default();
        let page = engine.paginate(&SearchQuery::new(), &sort, 3, None);
        assert_eq!(page.items.len(), 3);
        assert!(!page.has_next_page);
        assert!(!page.has_previous_page);
    }

    #[test]
    fn test_paginate_larger_than_total() {
        let engine = make_engine();
        let sort = SortSpec::default();
        let page = engine.paginate(&SearchQuery::new(), &sort, 100, None);
        assert_eq!(page.items.len(), 3);
        assert!(!page.has_next_page);
    }

    #[test]
    fn test_paginate_backward_last_page() {
        let engine = make_engine();
        let sort = SortSpec::default();
        let page = engine.paginate_backward(&SearchQuery::new(), &sort, 2, None);
        // Should get the last 2 items
        assert_eq!(page.items.len(), 2);
        assert_eq!(page.items[0].id, 2);
        assert_eq!(page.items[1].id, 3);
        assert!(page.has_previous_page);
        assert!(!page.has_next_page);
    }

    #[test]
    fn test_paginate_backward_from_cursor() {
        let engine = make_engine();
        let sort = SortSpec::default();
        // Get the cursor of item id=3 (last item) via start_cursor
        let cursor = Cursor::from_id(3);
        let page = engine.paginate_backward(&SearchQuery::new(), &sort, 2, Some(&cursor));
        assert_eq!(page.items.len(), 2);
        assert_eq!(page.items[0].id, 1);
        assert_eq!(page.items[1].id, 2);
    }

    #[test]
    fn test_paginate_full_traversal() {
        // Walk forward through all pages and collect every ID.
        let mut engine = AssetSearchEngine::new();
        for i in 1..=7_u64 {
            engine.add(AssetDescriptor::new(i, format!("Asset {i}")));
        }
        let sort = SortSpec::default();
        let mut all_ids = Vec::new();
        let mut cursor: Option<Cursor> = None;
        loop {
            let page = engine.paginate(&SearchQuery::new(), &sort, 3, cursor.as_ref());
            for item in &page.items {
                all_ids.push(item.id);
            }
            if !page.has_next_page {
                break;
            }
            cursor = page.end_cursor;
        }
        assert_eq!(all_ids, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_paginate_with_filter_and_sort() {
        let engine = make_engine();
        let sort = SortSpec {
            field: SortField::Title,
            direction: SortDirection::Ascending,
        };
        let q = SearchQuery::new().with(SearchField::Creator, "alice");
        let page = engine.paginate(&q, &sort, 10, None);
        assert_eq!(page.items.len(), 2);
        assert_eq!(page.total_count, 2);
        // Spring Shoot < Summer Campaign (alphabetical)
        assert_eq!(page.items[0].title, "Spring Shoot");
        assert_eq!(page.items[1].title, "Summer Campaign");
    }

    #[test]
    fn test_paginate_invalid_cursor_starts_from_beginning() {
        let engine = make_engine();
        let sort = SortSpec::default();
        let bad_cursor = Cursor::new("not-a-number");
        let page = engine.paginate(&SearchQuery::new(), &sort, 10, Some(&bad_cursor));
        // Should gracefully start from beginning
        assert_eq!(page.items.len(), 3);
    }

    #[test]
    fn test_filter_sorted_by_creator() {
        let engine = make_engine();
        let sort = SortSpec {
            field: SortField::Creator,
            direction: SortDirection::Ascending,
        };
        let results = engine.filter_sorted(&SearchQuery::new(), &sort);
        // Alice (x2) before Bob
        assert_eq!(results[0].creator, "Alice");
        assert_eq!(results[2].creator, "Bob");
    }

    #[test]
    fn test_filter_sorted_by_status() {
        let engine = make_engine();
        let sort = SortSpec {
            field: SortField::Status,
            direction: SortDirection::Ascending,
        };
        let results = engine.filter_sorted(&SearchQuery::new(), &sort);
        // "approved" before "draft"
        assert_eq!(results[0].status, "approved");
        assert_eq!(results[2].status, "draft");
    }
}
