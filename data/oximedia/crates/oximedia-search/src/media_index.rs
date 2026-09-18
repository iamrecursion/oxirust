//! Full-text media index with TF-IDF scoring, facet filters, and ranked results.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Duration bucket for faceted filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DurationBucket {
    /// 0–30 seconds.
    Short,
    /// 30 seconds – 5 minutes.
    Medium,
    /// 5–30 minutes.
    Long,
    /// Over 30 minutes.
    Feature,
}

impl DurationBucket {
    /// Returns the bucket for a duration in milliseconds.
    #[must_use]
    pub fn from_ms(ms: i64) -> Self {
        match ms {
            ms if ms < 30_000 => Self::Short,
            ms if ms < 300_000 => Self::Medium,
            ms if ms < 1_800_000 => Self::Long,
            _ => Self::Feature,
        }
    }
}

/// Facet filter for narrowing search results by category, tag, or duration bucket.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FacetFilter {
    /// Required categories (any match).
    pub categories: Vec<String>,
    /// Required tags (any match).
    pub tags: Vec<String>,
    /// Required duration buckets (any match).
    pub duration_buckets: Vec<DurationBucket>,
    /// Required codec values (any match).
    pub codecs: Vec<String>,
    /// Date range as (`start_unix`, `end_unix`).
    pub date_range: Option<(i64, i64)>,
    /// Duration range in milliseconds as (min, max).
    pub duration_range: Option<(i64, i64)>,
    /// Required resolution strings (any match).
    pub resolutions: Vec<String>,
}

impl FacetFilter {
    /// Creates a new empty facet filter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts to given categories.
    #[must_use]
    pub fn with_categories(
        mut self,
        categories: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.categories = categories
            .into_iter()
            .map(std::convert::Into::into)
            .collect();
        self
    }

    /// Restricts to given tags.
    #[must_use]
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(std::convert::Into::into).collect();
        self
    }

    /// Restricts to given duration buckets.
    #[must_use]
    pub fn with_duration_buckets(
        mut self,
        buckets: impl IntoIterator<Item = DurationBucket>,
    ) -> Self {
        self.duration_buckets = buckets.into_iter().collect();
        self
    }

    /// Restricts to given codecs.
    #[must_use]
    pub fn with_codecs(mut self, codecs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.codecs = codecs.into_iter().map(std::convert::Into::into).collect();
        self
    }

    /// Sets a date range filter.
    #[must_use]
    pub fn with_date_range(mut self, start: i64, end: i64) -> Self {
        self.date_range = Some((start, end));
        self
    }

    /// Sets a duration range filter in milliseconds.
    #[must_use]
    pub fn with_duration_range(mut self, min_ms: i64, max_ms: i64) -> Self {
        self.duration_range = Some((min_ms, max_ms));
        self
    }

    /// Checks whether a document entry passes this filter.
    #[must_use]
    pub fn matches(&self, doc: &MediaDocument) -> bool {
        if !self.categories.is_empty()
            && !doc.categories.iter().any(|c| self.categories.contains(c))
        {
            return false;
        }
        if !self.tags.is_empty() && !doc.tags.iter().any(|t| self.tags.contains(t)) {
            return false;
        }
        if !self.duration_buckets.is_empty() {
            match doc.duration_ms {
                None => return false,
                Some(ms) => {
                    if !self.duration_buckets.contains(&DurationBucket::from_ms(ms)) {
                        return false;
                    }
                }
            }
        }
        if !self.codecs.is_empty() {
            match &doc.codec {
                None => return false,
                Some(c) => {
                    if !self.codecs.contains(c) {
                        return false;
                    }
                }
            }
        }
        if let Some((start, end)) = self.date_range {
            if doc.created_at < start || doc.created_at > end {
                return false;
            }
        }
        if let Some((min, max)) = self.duration_range {
            match doc.duration_ms {
                None => return false,
                Some(ms) if ms < min || ms > max => return false,
                _ => {}
            }
        }
        if !self.resolutions.is_empty() {
            match &doc.resolution {
                None => return false,
                Some(r) => {
                    if !self.resolutions.contains(r) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Lightweight media document stored in the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaDocument {
    /// Unique media identifier.
    pub media_id: String,
    /// Title of the media asset.
    pub title: String,
    /// Free-text description.
    pub description: String,
    /// Associated tags.
    pub tags: Vec<String>,
    /// Category labels.
    pub categories: Vec<String>,
    /// Video codec, e.g. `"h264"`.
    pub codec: Option<String>,
    /// Resolution string, e.g. `"1920x1080"`.
    pub resolution: Option<String>,
    /// Duration in milliseconds.
    pub duration_ms: Option<i64>,
    /// Creation unix timestamp.
    pub created_at: i64,
}

/// Search query for the `MediaIndex`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Free-text query string.
    pub text: String,
    /// Optional facet filter applied after scoring.
    pub filter: Option<FacetFilter>,
    /// Maximum number of results to return.
    pub limit: usize,
}

impl SearchQuery {
    /// Creates a new query with the given text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            filter: None,
            limit: 20,
        }
    }

    /// Attaches a facet filter.
    #[must_use]
    pub fn with_filter(mut self, filter: FacetFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets the result limit.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// A single ranked search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The media document identifier.
    pub media_id: String,
    /// TF-IDF relevance score.
    pub score: f64,
    /// Fields in which the query terms matched.
    pub matched_fields: Vec<String>,
    /// Short metadata snippet (title + truncated description).
    pub snippet: String,
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverted index internals
// ──────────────────────────────────────────────────────────────────────────────

/// Per-field term occurrence (used to build the inverted index).
#[derive(Debug, Clone)]
struct Posting {
    media_id: String,
    field: String,
    /// Term frequency in this field for this document.
    tf: f64,
}

/// In-memory inverted index with TF-IDF scoring.
///
/// # Example
///
/// ```
/// use oximedia_search::media_index::{MediaIndex, MediaDocument, SearchQuery};
///
/// let mut index = MediaIndex::new();
/// let doc = MediaDocument {
///     media_id: "vid-001".to_string(),
///     title: "Sunset over the ocean".to_string(),
///     description: "Beautiful golden sunset".to_string(),
///     tags: vec!["nature".to_string()],
///     categories: vec!["travel".to_string()],
///     codec: Some("h264".to_string()),
///     resolution: Some("1920x1080".to_string()),
///     duration_ms: Some(90_000),
///     created_at: 1_700_000_000,
/// };
/// index.index("vid-001", doc);
/// let results = index.search(&SearchQuery::new("sunset"));
/// assert!(!results.is_empty());
/// ```
pub struct MediaIndex {
    /// All indexed documents keyed by `media_id`.
    documents: HashMap<String, MediaDocument>,
    /// Inverted index: term → list of postings.
    inverted: HashMap<String, Vec<Posting>>,
    /// Total number of documents indexed (for IDF denominator).
    doc_count: usize,
    /// Per-term document frequency (for IDF).
    doc_freq: HashMap<String, usize>,
}

impl Default for MediaIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MediaIndex {
    /// Creates a new empty `MediaIndex`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            inverted: HashMap::new(),
            doc_count: 0,
            doc_freq: HashMap::new(),
        }
    }

    /// Adds or replaces a document in the index.
    ///
    /// `id` is a stable identifier (e.g., a UUID or file path hash).
    /// `metadata` is the [`MediaDocument`] to index.
    pub fn index(&mut self, id: impl Into<String>, metadata: MediaDocument) {
        let id = id.into();

        // Remove old postings if the document already exists.
        if self.documents.contains_key(&id) {
            self.remove_postings(&id);
            self.doc_count = self.doc_count.saturating_sub(1);
        }

        // Build term → (field, tf) from indexable text fields.
        let field_texts: Vec<(&str, String)> = vec![
            ("title", metadata.title.clone()),
            ("description", metadata.description.clone()),
            ("tags", metadata.tags.join(" ")),
            ("categories", metadata.categories.join(" ")),
        ];

        // Accumulate per-field TF maps.
        let mut terms_seen_in_doc: HashMap<String, bool> = HashMap::new();

        for (field, text) in &field_texts {
            let tokens = tokenize(text);
            let total = tokens.len().max(1) as f64;
            let mut tf_map: HashMap<String, f64> = HashMap::new();
            for token in tokens {
                *tf_map.entry(token).or_insert(0.0) += 1.0 / total;
            }
            for (term, tf) in tf_map {
                self.inverted
                    .entry(term.clone())
                    .or_default()
                    .push(Posting {
                        media_id: id.clone(),
                        field: (*field).to_string(),
                        tf,
                    });
                if !terms_seen_in_doc.contains_key(&term) {
                    terms_seen_in_doc.insert(term.clone(), true);
                    *self.doc_freq.entry(term).or_insert(0) += 1;
                }
            }
        }

        self.documents.insert(id, metadata);
        self.doc_count += 1;
    }

    /// Searches the index and returns ranked [`SearchResult`]s.
    ///
    /// Applies TF-IDF scoring then optionally filters by the query's [`FacetFilter`].
    #[must_use]
    pub fn search(&self, query: &SearchQuery) -> Vec<SearchResult> {
        let tokens = tokenize(&query.text);
        if tokens.is_empty() {
            return vec![];
        }

        // Accumulate TF-IDF scores per document.
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut matched_fields: HashMap<String, Vec<String>> = HashMap::new();

        let n = self.doc_count as f64;

        for token in &tokens {
            if let Some(postings) = self.inverted.get(token) {
                let df = self.doc_freq.get(token).copied().unwrap_or(1) as f64;
                let idf = ((n + 1.0) / (df + 1.0)).ln() + 1.0;

                for posting in postings {
                    let tfidf = posting.tf * idf;
                    *scores.entry(posting.media_id.clone()).or_insert(0.0) += tfidf;
                    let fields = matched_fields.entry(posting.media_id.clone()).or_default();
                    if !fields.contains(&posting.field) {
                        fields.push(posting.field.clone());
                    }
                }
            }
        }

        // Build results, applying facet filter if present.
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .filter_map(|(id, score)| {
                let doc = self.documents.get(&id)?;
                if let Some(ref filter) = query.filter {
                    if !filter.matches(doc) {
                        return None;
                    }
                }
                let fields = matched_fields.get(&id).cloned().unwrap_or_default();
                let snippet = build_snippet(doc);
                Some(SearchResult {
                    media_id: id,
                    score,
                    matched_fields: fields,
                    snippet,
                })
            })
            .collect();

        // Sort by descending score.
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let limit = if query.limit == 0 {
            results.len()
        } else {
            query.limit
        };
        results.truncate(limit);
        results
    }

    /// Returns the number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.doc_count
    }

    /// Returns `true` if no documents have been indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Removes all postings for a given document id.
    fn remove_postings(&mut self, id: &str) {
        for postings in self.inverted.values_mut() {
            postings.retain(|p| p.media_id != id);
        }
        // Rebuild doc_freq from scratch (simple approach for an in-memory index).
        self.doc_freq.clear();
        let mut seen: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
        for (term, postings) in &self.inverted {
            for posting in postings {
                seen.entry(term.clone())
                    .or_default()
                    .insert(posting.media_id.clone());
            }
        }
        for (term, ids) in seen {
            self.doc_freq.insert(term, ids.len());
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Lowercases and splits text into tokens, stripping punctuation.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(String::from)
        .collect()
}

/// Builds a short snippet from title + description.
fn build_snippet(doc: &MediaDocument) -> String {
    let desc_preview: String = doc.description.chars().take(120).collect();
    if desc_preview.is_empty() {
        doc.title.clone()
    } else {
        format!("{} — {}", doc.title, desc_preview)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, title: &str, description: &str) -> (String, MediaDocument) {
        (
            id.to_string(),
            MediaDocument {
                media_id: id.to_string(),
                title: title.to_string(),
                description: description.to_string(),
                tags: vec!["sample".to_string()],
                categories: vec!["test".to_string()],
                codec: Some("h264".to_string()),
                resolution: Some("1920x1080".to_string()),
                duration_ms: Some(60_000),
                created_at: 1_700_000_000,
            },
        )
    }

    #[test]
    fn test_index_and_search_basic() {
        let mut index = MediaIndex::new();
        let (id, doc) = make_doc(
            "v1",
            "Sunset timelapse",
            "Beautiful golden sunset over hills",
        );
        index.index(id, doc);
        let results = index.search(&SearchQuery::new("sunset"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].media_id, "v1");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_search_no_match() {
        let mut index = MediaIndex::new();
        let (id, doc) = make_doc("v1", "Ocean waves", "Relaxing wave sound");
        index.index(id, doc);
        let results = index.search(&SearchQuery::new("volcano"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_ranking_order() {
        let mut index = MediaIndex::new();
        // "sunrise" appears once in doc2, twice in title+desc of doc1.
        index.index(
            "doc1",
            MediaDocument {
                media_id: "doc1".to_string(),
                title: "Sunrise sunrise".to_string(),
                description: "Morning sunrise glow".to_string(),
                tags: vec![],
                categories: vec![],
                codec: None,
                resolution: None,
                duration_ms: None,
                created_at: 0,
            },
        );
        index.index(
            "doc2",
            MediaDocument {
                media_id: "doc2".to_string(),
                title: "Waterfall".to_string(),
                description: "Cool sunrise view".to_string(),
                tags: vec![],
                categories: vec![],
                codec: None,
                resolution: None,
                duration_ms: None,
                created_at: 0,
            },
        );
        let results = index.search(&SearchQuery::new("sunrise"));
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_facet_filter_category() {
        let mut index = MediaIndex::new();
        for (i, cat) in ["nature", "sports", "music"].iter().enumerate() {
            index.index(
                format!("d{i}"),
                MediaDocument {
                    media_id: format!("d{i}"),
                    title: "sample video".to_string(),
                    description: "sample description".to_string(),
                    tags: vec![],
                    categories: vec![(*cat).to_string()],
                    codec: None,
                    resolution: None,
                    duration_ms: None,
                    created_at: 0,
                },
            );
        }
        let filter = FacetFilter::new().with_categories(["nature"]);
        let q = SearchQuery::new("sample").with_filter(filter);
        let results = index.search(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].media_id, "d0");
    }

    #[test]
    fn test_facet_filter_duration_bucket() {
        let mut index = MediaIndex::new();
        // Short: 10s, Long: 600s
        index.index(
            "short",
            MediaDocument {
                media_id: "short".to_string(),
                title: "clip".to_string(),
                description: "short clip description".to_string(),
                tags: vec![],
                categories: vec![],
                codec: None,
                resolution: None,
                duration_ms: Some(10_000), // 10 s → Short
                created_at: 0,
            },
        );
        index.index(
            "long",
            MediaDocument {
                media_id: "long".to_string(),
                title: "clip".to_string(),
                description: "long clip description".to_string(),
                tags: vec![],
                categories: vec![],
                codec: None,
                resolution: None,
                duration_ms: Some(600_000), // 10 min → Long
                created_at: 0,
            },
        );
        let filter = FacetFilter::new().with_duration_buckets([DurationBucket::Short]);
        let results = index.search(&SearchQuery::new("clip").with_filter(filter));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].media_id, "short");
    }

    #[test]
    fn test_index_len_and_is_empty() {
        let mut index = MediaIndex::new();
        assert!(index.is_empty());
        let (id, doc) = make_doc("v1", "Test", "Test description");
        index.index(id, doc);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_matched_fields_populated() {
        let mut index = MediaIndex::new();
        let (id, doc) = make_doc("v1", "Rocket launch", "Space rocket description");
        index.index(id, doc);
        let results = index.search(&SearchQuery::new("rocket"));
        assert!(!results.is_empty());
        assert!(!results[0].matched_fields.is_empty());
        assert!(
            results[0].matched_fields.contains(&"title".to_string())
                || results[0]
                    .matched_fields
                    .contains(&"description".to_string())
        );
    }

    #[test]
    fn test_snippet_contains_title() {
        let mut index = MediaIndex::new();
        let (id, doc) = make_doc("v1", "Mountain hike", "Epic mountain trail adventure");
        index.index(id, doc);
        let results = index.search(&SearchQuery::new("mountain"));
        assert!(!results.is_empty());
        assert!(results[0].snippet.contains("Mountain hike"));
    }

    #[test]
    fn test_limit_applied() {
        let mut index = MediaIndex::new();
        for i in 0..10 {
            index.index(
                format!("d{i}"),
                MediaDocument {
                    media_id: format!("d{i}"),
                    title: "test video".to_string(),
                    description: "test description content".to_string(),
                    tags: vec![],
                    categories: vec![],
                    codec: None,
                    resolution: None,
                    duration_ms: None,
                    created_at: 0,
                },
            );
        }
        let results = index.search(&SearchQuery::new("test").with_limit(3));
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_duration_bucket_classification() {
        assert_eq!(DurationBucket::from_ms(5_000), DurationBucket::Short);
        assert_eq!(DurationBucket::from_ms(60_000), DurationBucket::Medium);
        assert_eq!(DurationBucket::from_ms(900_000), DurationBucket::Long);
        assert_eq!(DurationBucket::from_ms(7_200_000), DurationBucket::Feature);
    }

    #[test]
    fn test_facet_filter_codec() {
        let mut index = MediaIndex::new();
        for (id, codec) in [("a", "h264"), ("b", "av1"), ("c", "vp9")] {
            index.index(
                id,
                MediaDocument {
                    media_id: id.to_string(),
                    title: "video clip".to_string(),
                    description: "video description".to_string(),
                    tags: vec![],
                    categories: vec![],
                    codec: Some(codec.to_string()),
                    resolution: None,
                    duration_ms: None,
                    created_at: 0,
                },
            );
        }
        let filter = FacetFilter::new().with_codecs(["av1"]);
        let results = index.search(&SearchQuery::new("video").with_filter(filter));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].media_id, "b");
    }
}
