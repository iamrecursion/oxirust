//! MAM full-text search index.
//!
//! A lightweight, in-process inverted index for searching asset metadata
//! without requiring an external search engine.
//!
//! This module contains two implementations:
//!
//! * [`SearchIndex`] — a simple, pure-Rust in-memory index suitable for small
//!   collections and unit tests.
//! * [`TantivySearchIndex`] — a Tantivy-backed index with incremental
//!   add/update/delete and index warming, designed for production use.

/// A single searchable field attached to an asset document.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct IndexedField {
    /// Field name (e.g. "title", "description", "tags").
    pub name: String,
    /// Field value as a string.
    pub value: String,
    /// Relative importance multiplier (higher = more relevant).
    pub weight: f32,
    /// Whether this field participates in full-text searches.
    pub searchable: bool,
}

impl IndexedField {
    /// Create a new indexed field.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        value: impl Into<String>,
        weight: f32,
        searchable: bool,
    ) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            weight,
            searchable,
        }
    }

    /// Returns `true` when the field's value contains the query (case-insensitive).
    #[must_use]
    pub fn matches(&self, query: &str) -> bool {
        if !self.searchable {
            return false;
        }
        self.value.to_lowercase().contains(&query.to_lowercase())
    }
}

/// A document representing one asset with all its indexed metadata.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AssetDocument {
    /// Unique asset identifier.
    pub id: String,
    /// All fields attached to this document.
    pub fields: Vec<IndexedField>,
}

impl AssetDocument {
    /// Create a new, empty asset document.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: Vec::new(),
        }
    }

    /// Append a field to this document.
    pub fn add_field(&mut self, field: IndexedField) {
        self.fields.push(field);
    }

    /// Retrieve the string value of the first field with `name`, if present.
    #[must_use]
    pub fn field_value(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.value.as_str())
    }

    /// Compute a relevance score for `query` across all searchable fields.
    ///
    /// Score is the sum of weights of matching fields.
    #[must_use]
    pub fn search_score(&self, query: &str) -> f32 {
        self.fields
            .iter()
            .filter(|f| f.matches(query))
            .map(|f| f.weight)
            .sum()
    }
}

/// In-process search index over a collection of asset documents.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct SearchIndex {
    /// All indexed documents.
    pub documents: Vec<AssetDocument>,
}

impl SearchIndex {
    /// Create a new, empty search index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a document to the index.
    pub fn add(&mut self, doc: AssetDocument) {
        self.documents.push(doc);
    }

    /// Remove the document with `id` from the index.
    ///
    /// Returns `true` if a document was found and removed.
    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.documents.len();
        self.documents.retain(|d| d.id != id);
        self.documents.len() < before
    }

    /// Search all documents for `query` and return `(doc, score)` pairs sorted
    /// by descending relevance score.  Documents with a score of `0.0` are excluded.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<(&AssetDocument, f32)> {
        let mut results: Vec<(&AssetDocument, f32)> = self
            .documents
            .iter()
            .filter_map(|doc| {
                let score = doc.search_score(query);
                if score > 0.0 {
                    Some((doc, score))
                } else {
                    None
                }
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Return the top `n` documents matching `query`.
    #[must_use]
    pub fn top_results(&self, query: &str, n: usize) -> Vec<&AssetDocument> {
        self.search(query)
            .into_iter()
            .take(n)
            .map(|(doc, _)| doc)
            .collect()
    }

    /// Returns the number of documents in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Returns `true` when the index contains no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

/// A structured search query with keyword terms and field-value filters.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SearchQuery {
    /// Free-text terms that must all appear in at least one field.
    pub terms: Vec<String>,
    /// Exact field filters as `(field_name, expected_value)` pairs.
    pub filters: Vec<(String, String)>,
}

impl SearchQuery {
    /// Parse a query string into terms and `field:value` filters.
    ///
    /// Tokens that contain a `:` are treated as field filters; all others
    /// are treated as free-text terms.
    #[must_use]
    pub fn parse(input: &str) -> Self {
        let mut terms = Vec::new();
        let mut filters = Vec::new();
        for token in input.split_whitespace() {
            if let Some((field, value)) = token.split_once(':') {
                filters.push((field.to_string(), value.to_string()));
            } else {
                terms.push(token.to_lowercase());
            }
        }
        Self { terms, filters }
    }

    /// Returns `true` when `doc` satisfies all terms and filters in this query.
    #[must_use]
    pub fn matches_document(&self, doc: &AssetDocument) -> bool {
        // All free-text terms must match at least one searchable field.
        let terms_ok = self
            .terms
            .iter()
            .all(|term| doc.fields.iter().any(|f| f.matches(term)));

        // All filters must match exactly.
        let filters_ok = self.filters.iter().all(|(field, expected)| {
            doc.field_value(field)
                .map(|v| v.to_lowercase() == expected.to_lowercase())
                .unwrap_or(false)
        });

        terms_ok && filters_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, title: &str, tags: &str) -> AssetDocument {
        let mut doc = AssetDocument::new(id);
        doc.add_field(IndexedField::new("title", title, 2.0, true));
        doc.add_field(IndexedField::new("tags", tags, 1.0, true));
        doc.add_field(IndexedField::new("id_field", id, 0.5, false));
        doc
    }

    fn populated_index() -> SearchIndex {
        let mut idx = SearchIndex::new();
        idx.add(make_doc("a1", "Breaking News London", "news uk"));
        idx.add(make_doc("a2", "Sports Highlights", "sports football"));
        idx.add(make_doc(
            "a3",
            "Corporate Announcement",
            "corporate business news",
        ));
        idx
    }

    // --- IndexedField tests ---

    #[test]
    fn test_field_matches_case_insensitive() {
        let f = IndexedField::new("title", "Breaking News", 1.0, true);
        assert!(f.matches("news"));
        assert!(f.matches("BREAKING"));
    }

    #[test]
    fn test_field_not_searchable_never_matches() {
        let f = IndexedField::new("internal", "secret", 1.0, false);
        assert!(!f.matches("secret"));
    }

    #[test]
    fn test_field_no_match() {
        let f = IndexedField::new("title", "Sports", 1.0, true);
        assert!(!f.matches("weather"));
    }

    // --- AssetDocument tests ---

    #[test]
    fn test_document_field_value_found() {
        let doc = make_doc("x", "My Title", "tag1");
        assert_eq!(doc.field_value("title"), Some("My Title"));
    }

    #[test]
    fn test_document_field_value_missing() {
        let doc = make_doc("x", "My Title", "tag1");
        assert!(doc.field_value("nonexistent").is_none());
    }

    #[test]
    fn test_document_search_score_positive() {
        let doc = make_doc("x", "Football Highlights", "sports");
        let score = doc.search_score("football");
        assert!(score > 0.0);
    }

    #[test]
    fn test_document_search_score_zero_no_match() {
        let doc = make_doc("x", "Football Highlights", "sports");
        let score = doc.search_score("cooking");
        assert!((score - 0.0).abs() < f32::EPSILON);
    }

    // --- SearchIndex tests ---

    #[test]
    fn test_index_add_and_len() {
        let mut idx = SearchIndex::new();
        idx.add(make_doc("d1", "Title One", "tag"));
        idx.add(make_doc("d2", "Title Two", "tag"));
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_index_remove_existing() {
        let mut idx = SearchIndex::new();
        idx.add(make_doc("d1", "Title", "tag"));
        assert!(idx.remove("d1"));
        assert!(idx.is_empty());
    }

    #[test]
    fn test_index_remove_nonexistent() {
        let mut idx = SearchIndex::new();
        assert!(!idx.remove("ghost"));
    }

    #[test]
    fn test_index_search_returns_relevant() {
        let idx = populated_index();
        let results = idx.search("news");
        assert!(!results.is_empty());
        assert!(results.iter().any(|(d, _)| d.id == "a1" || d.id == "a3"));
    }

    #[test]
    fn test_index_search_sorted_by_score() {
        let idx = populated_index();
        let results = idx.search("news");
        if results.len() >= 2 {
            assert!(results[0].1 >= results[1].1);
        }
    }

    #[test]
    fn test_index_top_results_limit() {
        let idx = populated_index();
        let top = idx.top_results("news", 1);
        assert_eq!(top.len(), 1);
    }

    // --- SearchQuery tests ---

    #[test]
    fn test_query_parse_terms() {
        let q = SearchQuery::parse("football highlights");
        assert_eq!(q.terms, vec!["football", "highlights"]);
        assert!(q.filters.is_empty());
    }

    #[test]
    fn test_query_parse_filters() {
        let q = SearchQuery::parse("type:video");
        assert_eq!(q.filters, vec![("type".to_string(), "video".to_string())]);
        assert!(q.terms.is_empty());
    }

    #[test]
    fn test_query_matches_document_terms() {
        let q = SearchQuery::parse("sports");
        let doc = make_doc("y", "Sports Reel", "sport events");
        assert!(q.matches_document(&doc));
    }

    #[test]
    fn test_query_no_match_missing_term() {
        let q = SearchQuery::parse("cooking");
        let doc = make_doc("y", "Sports Reel", "sport events");
        assert!(!q.matches_document(&doc));
    }
}

// =============================================================================
// TantivySearchIndex — incremental Tantivy-backed index (Wave 15)
// =============================================================================

use std::sync::{Arc, RwLock};
use tantivy::{
    collector::{Count, TopDocs},
    query::{QueryParser, TermQuery},
    schema::{Schema, SchemaBuilder, Term, Value, STORED, STRING, TEXT},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument,
};

/// Structured search fields for a single MAM asset.
///
/// Used as the canonical input type for [`TantivySearchIndex`] operations.
#[derive(Debug, Clone, Default)]
pub struct AssetSearchFields {
    /// Human-readable asset title.
    pub title: String,
    /// Free-text description.
    pub description: String,
    /// Space-separated tags / keywords.
    pub tags: String,
    /// MIME type string (e.g. `"video/mp4"`).
    pub mime_type: String,
}

/// Error type for [`TantivySearchIndex`] operations.
#[derive(Debug, thiserror::Error)]
pub enum SearchIndexError {
    /// An underlying Tantivy error.
    #[error("Tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    /// A Tantivy directory open error.
    #[error("Directory open error: {0}")]
    Directory(#[from] tantivy::directory::error::OpenDirectoryError),
    /// An I/O error (e.g. creating the index directory).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A query parse error.
    #[error("Query parse error: {0}")]
    QueryParse(#[from] tantivy::query::QueryParserError),
    /// The internal RwLock was poisoned by a previous panic.
    #[error("Index writer lock poisoned")]
    LockPoisoned,
    /// The requested schema field is missing — indicates a programming error.
    #[error("Schema field not found: {0}")]
    FieldNotFound(String),
}

/// Result alias for [`SearchIndexError`].
pub type SearchIndexResult<T> = std::result::Result<T, SearchIndexError>;

/// Tantivy-backed in-process search index with incremental update support.
///
/// All mutations (add / update / delete) are committed immediately and the
/// reader is reloaded so that subsequent calls to [`Self::search`] observe the
/// change.
///
/// # Thread safety
///
/// The [`IndexWriter`] is wrapped in an `Arc<RwLock<...>>` so that multiple
/// threads may issue concurrent reads while a single thread holds a write lock
/// for mutations.
pub struct TantivySearchIndex {
    /// The underlying Tantivy index (kept alive for the lifetime of this
    /// struct; fields and schema are resolved from here).
    index: Index,
    /// Live reader — reloaded after each commit so searches are current.
    reader: IndexReader,
    /// Writer wrapped in a lock to serialise concurrent mutation requests.
    writer: Arc<RwLock<IndexWriter>>,
    /// Schema built at construction time; used by all field lookups.
    schema: Schema,
    /// Query parser pre-configured for text fields.
    query_parser: QueryParser,
}

impl TantivySearchIndex {
    /// Build the Tantivy schema used by this index.
    fn build_schema() -> Schema {
        let mut builder: SchemaBuilder = Schema::builder();
        // doc_id is the primary key — exact-match only (STRING) and stored so
        // we can retrieve it from search results.
        builder.add_text_field("doc_id", STRING | STORED);
        // Text fields participate in full-text search.
        builder.add_text_field("title", TEXT | STORED);
        builder.add_text_field("description", TEXT);
        builder.add_text_field("tags", TEXT);
        builder.add_text_field("mime_type", STRING);
        builder.build()
    }

    /// Create a new `TantivySearchIndex` backed by `index`.
    ///
    /// The schema **must** have been built with [`TantivySearchIndex::build_schema`]
    /// (or an equivalent layout).  Use [`TantivySearchIndex::new_ram`] for the
    /// typical in-process / test use-case.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer or reader cannot be initialised, or if
    /// the initial warming fails.
    fn new_from_index(index: Index) -> SearchIndexResult<Self> {
        let schema = index.schema();

        let writer: IndexWriter = index.writer(15_000_000)?; // 15 MB heap

        let reader: IndexReader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        let doc_id_field = schema
            .get_field("doc_id")
            .map_err(|_| SearchIndexError::FieldNotFound("doc_id".to_string()))?;
        let title_field = schema
            .get_field("title")
            .map_err(|_| SearchIndexError::FieldNotFound("title".to_string()))?;
        let description_field = schema
            .get_field("description")
            .map_err(|_| SearchIndexError::FieldNotFound("description".to_string()))?;
        let tags_field = schema
            .get_field("tags")
            .map_err(|_| SearchIndexError::FieldNotFound("tags".to_string()))?;

        let query_parser = QueryParser::for_index(
            &index,
            vec![doc_id_field, title_field, description_field, tags_field],
        );

        let idx = Self {
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            schema,
            query_parser,
        };

        // Warm immediately so the reader is current.
        idx.warm()?;

        Ok(idx)
    }

    /// Create a new `TantivySearchIndex` that stores data in RAM only.
    ///
    /// This is the recommended constructor for tests and in-process use where
    /// durability is not required.
    ///
    /// # Errors
    ///
    /// Returns an error if Tantivy cannot initialise the RAM-backed index.
    pub fn new_ram() -> SearchIndexResult<Self> {
        let schema = Self::build_schema();
        let index = Index::create_in_ram(schema);
        Self::new_from_index(index)
    }

    /// Create a new `TantivySearchIndex` persisted in the directory at `path`.
    ///
    /// If the directory already contains an index it is opened; otherwise a new
    /// index is created.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be opened or the index cannot
    /// be initialised.
    pub fn new_in_dir(path: &std::path::Path) -> SearchIndexResult<Self> {
        let schema = Self::build_schema();
        let index = if path.exists() {
            let dir = tantivy::directory::MmapDirectory::open(path)?;
            Index::open(dir)?
        } else {
            std::fs::create_dir_all(path)?;
            Index::create_in_dir(path, schema)?
        };
        Self::new_from_index(index)
    }

    /// Build a Tantivy document from `doc_id` and `fields`.
    fn build_tantivy_doc(
        &self,
        doc_id: &str,
        fields: &AssetSearchFields,
    ) -> SearchIndexResult<TantivyDocument> {
        let mut doc = TantivyDocument::new();
        doc.add_text(
            self.schema
                .get_field("doc_id")
                .map_err(|_| SearchIndexError::FieldNotFound("doc_id".to_string()))?,
            doc_id,
        );
        doc.add_text(
            self.schema
                .get_field("title")
                .map_err(|_| SearchIndexError::FieldNotFound("title".to_string()))?,
            &fields.title,
        );
        doc.add_text(
            self.schema
                .get_field("description")
                .map_err(|_| SearchIndexError::FieldNotFound("description".to_string()))?,
            &fields.description,
        );
        doc.add_text(
            self.schema
                .get_field("tags")
                .map_err(|_| SearchIndexError::FieldNotFound("tags".to_string()))?,
            &fields.tags,
        );
        doc.add_text(
            self.schema
                .get_field("mime_type")
                .map_err(|_| SearchIndexError::FieldNotFound("mime_type".to_string()))?,
            &fields.mime_type,
        );
        Ok(doc)
    }

    /// Add a document to the index and commit immediately.
    ///
    /// After a successful call the document is visible to calls to [`Self::search`]
    /// (the reader is reloaded as part of the commit).
    ///
    /// # Errors
    ///
    /// Returns an error if the document cannot be built, the writer lock is
    /// poisoned, or the commit fails.
    pub fn add_document(&self, doc_id: &str, fields: &AssetSearchFields) -> SearchIndexResult<()> {
        let tantivy_doc = self.build_tantivy_doc(doc_id, fields)?;
        let mut writer = self
            .writer
            .write()
            .map_err(|_| SearchIndexError::LockPoisoned)?;
        writer.add_document(tantivy_doc)?;
        writer.commit()?;
        drop(writer); // release write lock before reloading reader
        self.reader.reload()?;
        Ok(())
    }

    /// Replace an existing document with updated content.
    ///
    /// The old entry is deleted by `doc_id` term before the new one is
    /// inserted, so there is at most one document per `doc_id` after the call.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete, insert, or commit fails.
    pub fn update_document(
        &self,
        doc_id: &str,
        fields: &AssetSearchFields,
    ) -> SearchIndexResult<()> {
        let id_field = self
            .schema
            .get_field("doc_id")
            .map_err(|_| SearchIndexError::FieldNotFound("doc_id".to_string()))?;
        let delete_term = Term::from_field_text(id_field, doc_id);
        let tantivy_doc = self.build_tantivy_doc(doc_id, fields)?;

        let mut writer = self
            .writer
            .write()
            .map_err(|_| SearchIndexError::LockPoisoned)?;
        writer.delete_term(delete_term);
        writer.add_document(tantivy_doc)?;
        writer.commit()?;
        drop(writer);
        self.reader.reload()?;
        Ok(())
    }

    /// Remove a document by `doc_id` and commit immediately.
    ///
    /// After a successful call the document no longer appears in [`Self::search`]
    /// results.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete or commit fails.
    pub fn delete_document(&self, doc_id: &str) -> SearchIndexResult<()> {
        let id_field = self
            .schema
            .get_field("doc_id")
            .map_err(|_| SearchIndexError::FieldNotFound("doc_id".to_string()))?;
        let term = Term::from_field_text(id_field, doc_id);
        let mut writer = self
            .writer
            .write()
            .map_err(|_| SearchIndexError::LockPoisoned)?;
        writer.delete_term(term);
        writer.commit()?;
        drop(writer);
        self.reader.reload()?;
        Ok(())
    }

    /// Warm the index by reloading the reader and running a lightweight probe
    /// query to prime OS page caches and segment metadata.
    ///
    /// This is called automatically by the constructors; you may also call it
    /// explicitly after a bulk load to ensure the reader is current.
    ///
    /// # Errors
    ///
    /// Returns an error if the reader cannot be reloaded.
    pub fn warm(&self) -> SearchIndexResult<()> {
        self.reader.reload()?;

        // Fire a low-overhead probe search to prime segment-level caches.
        // The `Count` collector avoids allocating a result vector.
        let searcher = self.reader.searcher();
        let probe_query = self.query_parser.parse_query("*").unwrap_or_else(|_| {
            // If `*` is unparseable fall back to a guaranteed-valid empty query.
            Box::new(tantivy::query::AllQuery)
        });
        let _ = searcher.search(&*probe_query, &Count);

        Ok(())
    }

    /// Search for documents matching `query_str` using full-text search across
    /// title, description, tags, and doc_id fields.
    ///
    /// Returns the `doc_id` strings of matching documents ordered by Tantivy
    /// relevance score (highest first), limited to `limit` results.
    ///
    /// # Errors
    ///
    /// Returns an error if the query cannot be parsed or the search fails.
    pub fn search(&self, query_str: &str, limit: usize) -> SearchIndexResult<Vec<String>> {
        let searcher = self.reader.searcher();
        let query = self.query_parser.parse_query(query_str)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;

        let doc_id_field = self
            .schema
            .get_field("doc_id")
            .map_err(|_| SearchIndexError::FieldNotFound("doc_id".to_string()))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (_score, doc_address) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(id_str) = retrieved.get_first(doc_id_field).and_then(|v| v.as_str()) {
                results.push(id_str.to_string());
            }
        }
        Ok(results)
    }

    /// Perform an exact-term lookup on the `doc_id` field.
    ///
    /// Returns `true` if at least one document with `doc_id == id` exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub fn contains(&self, id: &str) -> SearchIndexResult<bool> {
        let id_field = self
            .schema
            .get_field("doc_id")
            .map_err(|_| SearchIndexError::FieldNotFound("doc_id".to_string()))?;
        let term = Term::from_field_text(id_field, id);
        let term_query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let searcher = self.reader.searcher();
        let count = searcher.search(&term_query, &Count)?;
        Ok(count > 0)
    }

    /// Return the number of documents currently in the index.
    #[must_use]
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Return the number of indexed segments.
    ///
    /// Useful for diagnostics; each commit may produce one additional segment
    /// until Tantivy's background merger collapses them.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.index
            .searchable_segments()
            .map(|s| s.len())
            .unwrap_or(0)
    }
}
